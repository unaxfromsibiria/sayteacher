use futures_util::{SinkExt, StreamExt};
use log::{debug, error, info, warn};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tokio_util::codec::{Framed, LengthDelimitedCodec};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rodio::{buffer::SamplesBuffer, mixer::Mixer, stream::DeviceSinkBuilder};
use std::num::NonZero;

use super::settings::Settings;
use microagent_transport::{Bytes, Message, MessageKind, Uuid};

// Silence-detection tuning (peak amplitude, not RMS — avoids the sqrt).
const TAIL_SILENCE_MS: u64 = 1100; // trailing quiet required to end a recording
const MAX_RECORD_FACTOR: f64 = 8.0; // safety cap so an endless stream stops the recorder
const TAIL_POLL_MS: u64 = 160; // how often to re-check the tail for silence

// Minimum fraction of audio frames that must carry signal above threshold before we
// consider the interval useful. Keyboard clicks and other short transients fail this check.
const MIN_SIGNAL_FILL_RATIO: f32 = 0.10;

/// Speech recognition only needs ~16 kHz mono (whisper), so downsample before sending.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Linear-interpolation resample from `from_rate` to `to_rate`.
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.len() < 2 {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = ((samples.len() as f64) / ratio).floor() as usize + 1;
    (0..out_len).map(|i| resample_one(i, samples, ratio)).collect()
}

fn resample_one(i: usize, samples: &[f32], ratio: f64) -> f32 {
    let pos = i as f64 * ratio;
    let idx = pos as usize;
    if idx >= samples.len() - 1 {
        return samples.last().copied().unwrap_or(0.0);
    }

    let frac = (pos - idx as f64) as f32;
    samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
}
/// Downmix interleaved multichannel samples to mono by averaging the channels.
fn downmix_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let ch = channels.max(1) as usize;
    if ch == 1 {
        return samples.to_vec();
    }

    let frames = samples.len() / ch;
    (0..frames).map(|i| {
        let start = i * ch;
        samples[start..start + ch].iter().sum::<f32>() / ch as f32
    }).collect()
}

/// Peak absolute amplitude of a sample buffer — cheap silence test (no sqrt).
fn peak_amplitude(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0_f32, |m, s| m.max(s.abs()))
}

/// Fraction of ~10 ms windows where at least one sample reaches `threshold`.
/// Returns 0.0 when the buffer is entirely below threshold; 1.0 when every window has signal.
fn signal_fill_ratio(samples: &[f32], channels: u16, sample_rate: u32, threshold: f32) -> f32 {
    let ch = (channels.max(1)) as usize;
    // One 10 ms frame in device-rate multichannel samples.
    let window_len = (((sample_rate as u64 * 10) / 1000).max(1) as usize) * ch;

    if samples.len() < window_len {
        return if peak_amplitude(samples) >= threshold { 1.0 } else { 0.0 };
    }

    let mut active = 0u32;
    let total = (samples.len() + window_len - 1) / window_len; // >= 1 here due to early-return
    for chunk in samples.chunks(window_len) {
        // Per-frame max so no single channel masks the other.
        let frame_has_signal = (0..chunk.len() / ch.max(1)).any(|f| {
            let start = f * ch;
            chunk[start..start + ch].iter().any(|s| s.abs() >= threshold)
        });
        if frame_has_signal {
            active += 1;
        }
    }

    (active as f32 / total as f32).min(1.0)
}

/// Client audio queued for playback after the next record/send cycle.
struct PendingAudio {
    samples: Vec<f32>,
}

fn f32_vec_from_le_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Final output of one record interval.
struct IntervalResult {
    samples: Vec<f32>,
    pending_playback: Vec<PendingAudio>,
    /// True when the captured fragment is not worth sending (silent or mostly silence).
    silent: bool,
}

/// Runs on a dedicated OS thread so cpal::Stream (non-Send) stays there.
/// Records for at least `delay_secs`, then extends the capture while audio is still present
/// and stops once the tail goes quiet. Stops early if the client sends data back.
/// Mute mode: the record loop skips mic capture and uses a short service interval instead.
const MUTE_INTERVAL_SECS: f64 = 0.5;

fn record_on_thread(
    input_device: &cpal::Device,
    delay_secs: f64,
    threshold: f32,
    inbound: std::sync::mpsc::Receiver<Vec<u8>>,
    ui: super::ui::UiState,
) -> Result<IntervalResult, String> {
    let config = input_device.default_input_config().map_err(
        |e| format!("input config error: {}", e)
    )?;
    let channels = config.channels();
    let sample_rate = config.sample_rate();

    // Track when sound was last heard so the tail can be tested for silence.
    let started = std::time::Instant::now();
    let last_sound: Arc<Mutex<std::time::Instant>> = Arc::new(Mutex::new(started));

    info!("recording {}s @ {} Hz, {} ch (level threshold {:.4})", delay_secs, sample_rate, channels, threshold);
    // The audio callback keeps running until the stream is dropped below, so the buffer it
    // writes into must be shared rather than moved out of the closure.
    let recorded: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(
        (delay_secs * sample_rate as f64 * channels as f64) as usize,
    )));
    let buf_cb = Arc::clone(&recorded);
    let last_sound_cb = Arc::clone(&last_sound);

    let stream = input_device.build_input_stream(
        config.config(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if !data.is_empty() && peak_amplitude(data) >= threshold {
                *last_sound_cb.lock().expect("last_sound poisoned") = std::time::Instant::now();
            }
            if let Ok(mut g) = buf_cb.lock() {
                g.extend_from_slice(data);
            }
        },
        |err| warn!("input stream error: {}", err),
        None,
    ).map_err(|e| format!("build_input_stream failed: {}", e))?;
    stream.play().map_err(|e| format!("stream play failed: {}", e))?;

    let mut pending_playback: Vec<PendingAudio> = Vec::new();
    // Recording must not run forever if the audio keeps going.
    let delay_ms = (delay_secs * 1000.0) as u64;
    let max_record_ms = ((delay_secs * MAX_RECORD_FACTOR).max(delay_secs) * 1000.0) as u64;

    loop {
        let now = std::time::Instant::now();
        let elapsed_ms = now.duration_since(started).as_millis() as u64;

        // Stop recording as soon as the user presses Mute so we do not keep filling the buffer.
        if ui.is_muted() {
            info!("mute active, ending in-flight recording early (~{}s)", elapsed_ms / 1000);
            break;
        }

        // Before the minimum interval we keep recording unconditionally; after it we
        // poll more often so silence can end the capture, capped at max_record_secs.
        if elapsed_ms >= delay_ms {
            // Minimum interval reached: end on trailing silence or the safety cap.
            let last_sound_at = *last_sound.lock().expect("last_sound poisoned");
            let quiet_ms = now.saturating_duration_since(last_sound_at).as_millis() as u64;
            if quiet_ms >= TAIL_SILENCE_MS {
                info!(
                    "tail silent for {}ms, ending recording at ~{}s",
                    quiet_ms, elapsed_ms / 1000
                );
                break;
            }
            if elapsed_ms >= max_record_ms {
                warn!("recording hit safety cap of {}s, ending early", max_record_ms / 1000);
                break;
            }
        }

        // Wait for a client frame or re-poll the tail. Use a short timeout while in the
        // tail phase so silence can be detected even without inbound traffic.
        let poll_ms = if elapsed_ms >= delay_ms { TAIL_POLL_MS } else { u64::MAX };
        match inbound.recv_timeout(Duration::from_millis(poll_ms.min(5_000))) {
            Ok(raw_frame) => {
                handle_client_frame(&raw_frame, sample_rate, channels, &mut pending_playback, &ui);
                break; // Client data arrived: stop recording and play it back.
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => { /* re-check tail above */ },
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                warn!("recording aborted, client channel closed");
                break;
            }
        }
    }

    // Dropping the stream stops the audio callback so we own the buffer exclusively.
    drop(stream);
    let samples = match Arc::try_unwrap(recorded) {
        Ok(m) => m.into_inner().expect("record buffer poisoned"),
        Err(_) => return Err("record buffer still referenced".to_string()),
    };

    // Drain any client frames still queued in the channel (e.g. a burst of echoes that
    // arrived while we were deciding to stop on silence). They are all meant for playback.
    while let Ok(raw) = inbound.try_recv() {
        handle_client_frame(&raw, sample_rate, channels, &mut pending_playback, &ui);
    }

    let peak = peak_amplitude(&samples);
    let fill = signal_fill_ratio(&samples, channels, sample_rate, threshold);
    // A clip is "silent" when there is no detectable sound at all (peak below threshold) or
    // when only a small fraction of the interval carried real signal — short keyboard clicks
    // and other brief transients fall out here and never reach the recognizer.
    let silent = peak < threshold || fill < MIN_SIGNAL_FILL_RATIO;

    if samples.is_empty() {
        warn!("no audio captured this interval, skipping send");
    } else {
        info!(
            "interval stats: peak={:.4} fill_ratio={} (threshold {:.4}, min fill {})",
            peak,
            (fill * 100.0) as u32,
            threshold,
            (MIN_SIGNAL_FILL_RATIO * 100.0) as u32
        );
    }

    Ok(IntervalResult { samples, pending_playback, silent })
}

/// Decide whether a captured interval should be emitted to the client as a transport frame.
/// Any in-flight capture must be dropped if Mute is active at send time, so a recording that
/// started before the button was pressed never slips through.
fn should_emit(result: &IntervalResult, client_gone: bool, muted_at_send: bool) -> bool {
    !result.samples.is_empty() && !client_gone && !result.silent && !muted_at_send
}

/// Parse a client frame: push its text into the matching UI field (note vs dialogue)
/// based on the first-audio-byte kind, and queue any audio body for playback.
fn handle_client_frame(
    bytes: &[u8], sample_rate: u32, channels: u16, out: &mut Vec<PendingAudio>, ui: &super::ui::UiState,
) {
    let msg = match Message::decode(bytes) {
        Ok(m) => m,
        Err(e) => {
            warn!("invalid client message: {}", e);
            return;
        }
    };

    // Route text by the frame kind: dialogue -> the new field, note/other -> the existing log.
    let kind = msg.kind();
    if !msg.text.is_empty() {
        info!("Text to read (kind={:?}):\n{}", kind, msg.text);
        match kind {
            Some(MessageKind::Dialogue) => ui.append_dialogue(&msg.text),
            _ => ui.append_note(&msg.text),
        }
    }

    // The audio body is everything after the optional one-byte type marker.
    let body = msg.audio_body();
    if body.is_empty() {
        return;
    }

    // Inbound audio arrives at TARGET_SAMPLE_RATE mono (server side downsamples before send),
    // so resample it back to the output device rate and replicate across channels.
    let mono = f32_vec_from_le_bytes(body.as_ref());
    let seconds = if TARGET_SAMPLE_RATE > 0 { mono.len() as f64 / TARGET_SAMPLE_RATE as f64 } else { 0.0 };
    info!("service sent audio ~{seconds:.2}s ({} bytes, id={})", body.len(), msg.id);
    out.push(PendingAudio { samples: upsample_to_device(&mono, sample_rate, channels) });
}

/// Convert incoming mono 16 kHz speech into interleaved device-rate multichannel samples.
fn upsample_to_device(samples: &[f32], to_rate: u32, channels: u16) -> Vec<f32> {
    let mono = resample(samples, TARGET_SAMPLE_RATE, to_rate);
    let ch = channels.max(1) as usize;
    (mono.iter()).flat_map(|s| std::iter::repeat(*s).take(ch)).collect()
}

/// One message to the playback worker: a batch of buffers plus an optional completion signal.
struct PlaybackBatch {
    buffers: Vec<PendingAudio>,
    done_tx: Option<std::sync::mpsc::Sender<()>>,
}

/// Spawn a dedicated OS thread that owns rodio's output device and plays batches in order.
fn start_playback_worker(channels: u16, sample_rate: u32) -> tokio::sync::mpsc::UnboundedSender<PlaybackBatch> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<PlaybackBatch>();

    let builder = std::thread::Builder::new().name("playback".into());
    let _handle = builder.spawn(move || {
        // Keep MixerDeviceSink alive for the lifetime of this thread; dropping it stops playback.
        let device_sink = match DeviceSinkBuilder::open_default_sink() {
            Ok(v) => v,
            Err(e) => {
                error!("failed to open audio output: {}", e);
                return;
            }
        };

        let mixer = device_sink.mixer();
        playback_loop(&mut rx, mixer, channels, sample_rate);
    }).expect("playback worker thread failed to start");

    tx
}

fn playback_loop(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<PlaybackBatch>,
    mixer: &Mixer,
    channels: u16,
    sample_rate: u32,
) {
    while let Some(batch) = rx.blocking_recv() {
        if !batch.buffers.is_empty() {
            for p in batch.buffers {
                play_one(mixer, &p.samples, channels, sample_rate);
            }
        }

        if let Some(done) = batch.done_tx {
            let _ = done.send(());
        }
    }
}

fn play_one(
    mixer: &Mixer,
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
) {
    let buf = SamplesBuffer::new(
        NonZero::new(channels).expect("channels > 0"),
        NonZero::new(sample_rate).expect("sample_rate > 0"),
        samples.to_vec(),
    );

    // Player is the new per-source API in rodio >= 0.21; it owns a slot
    // on the mixer and exposes sleep_until_end for blocking playback.
    let player = rodio::Player::connect_new(mixer);
    player.append(buf);
    player.sleep_until_end();
}

/// Play queued client audio and wait for the worker to finish.
fn play_pending(worker_tx: &tokio::sync::mpsc::UnboundedSender<PlaybackBatch>, list: Vec<PendingAudio>) {
    if list.is_empty() {
        return;
    }
    let total: usize = list.iter().map(|p| p.samples.len()).sum();
    info!("playing back queued client audio ({} samples)", total);

    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    if worker_tx.send(PlaybackBatch { buffers: list, done_tx: Some(done_tx) }).is_err() {
        warn!("playback worker gone; queued audio dropped");
        return;
    }
    // Safe on a multi-threaded runtime.
    tokio::task::block_in_place(move || { let _ = done_rx.recv(); });
    info!("playback finished");
}

/// One-shot auth handshake: wait up to 0.25s for a valid greeting, then verify it.
async fn authenticate(
    framed: &mut Framed<TcpStream, LengthDelimitedCodec>,
    expected_id: Uuid,
    peer: SocketAddr,
    password: &str,
) -> bool {
    let timeout = Duration::from_millis(250);
    match tokio::time::timeout(timeout, framed.next()).await {
        Ok(Some(Ok(raw))) => {
            let ok = match Message::decode(&raw) {
                Err(e) => {
                warn!("auth frame undecodable from {}: {}", peer, e);
                false
            }
                // Client may use any id in the frame; we only care that its text matches.
                Ok(m) => m.verify_auth(&expected_id, password),
            };
            if ok {
                info!("client {} authenticated", peer);
            } else {
                warn!("bad auth token from {}", peer);
            }
            ok
        }
        Ok(Some(Err(e))) => {
            warn!("auth read error from {}: {}", peer, e);
            false
        }
        Ok(None) => {
            warn!("client {} closed before greeting", peer);
            false
        }
        Err(_) => {
            warn!("no auth greeting from {} within {:?}", peer, timeout);
            false
        }
    }
}

/// Serve one client connection until it disconnects or errors.
async fn run_session(
    mut framed: Framed<TcpStream, LengthDelimitedCodec>,
    peer: SocketAddr,
    settings: Settings,
    ui: super::ui::UiState,
) {
    info!("new client connected: {}", peer);

    // One-shot auth handshake before any recording happens. The server presents a fresh
    // session id; the client must echo back sha256(id + password). If no valid greeting
    // arrives within the timeout we drop the connection outright.
    let session_id = Uuid::new_v4();
    if framed.send(Message::greeting(session_id).encode()).await.is_err() {
        info!("send greeting to {} failed, ending session", peer);
        return;
    }

    if !authenticate(&mut framed, session_id, peer, &settings.password).await {
        info!("auth failed for {}, ending session", peer);
        return;
    }

    // New authenticated client: deliver the lesson text (if configured) right away.
    if !settings.lesson_data.is_empty() {
        let lesson = Message { id: Uuid::nil(), text: settings.lesson_data.clone(), audio: Bytes::new() };
        match framed.send(lesson.encode()).await {
            Ok(_) => info!("sent lesson to {} ({} chars)", peer, settings.lesson_data.len()),
            Err(e) => { warn!("failed to send lesson to {}: {}", peer, e); return; }
        }
    }

    let host = cpal::default_host();
    let Some(input_device) = host.default_input_device() else {
        warn!("no default input device, session ends without recording");
        return;
    };

    // Input config is stable for the lifetime of a device handle.
    let (channels, sample_rate): (u16, u32) = match input_device.default_input_config() {
        Ok(c) => (c.channels(), c.sample_rate()),
        Err(e) => { error!("input config error for {}: {}", peer, e); return; }
    };

    // Playback worker owns a MixerDeviceSink (rodio >= 0.21 API) on its own OS thread.
    let playback_tx = start_playback_worker(channels, sample_rate);

    // The session is live from here on; reflect that in the UI so the toggle becomes active.
    ui.set_connected(true);

    loop {
        // Per-interval channels: async side pushes client frames into the recorder thread.
        let (frame_tx, frame_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        let mut pending_playback: Vec<PendingAudio> = Vec::new();
        let mut client_gone = false;

        // Muted cycle: no mic capture; process one client frame (if any) then a short service wait.
        if ui.is_muted() {
            debug!("muted: skipping mic capture this cycle");
            match tokio::time::timeout(Duration::from_millis((MUTE_INTERVAL_SECS * 1000.0) as u64), framed.next()).await {
                Ok(Some(Ok(raw))) => handle_client_frame(&raw, sample_rate, channels, &mut pending_playback, &ui),
                Ok(Some(Err(e))) => { warn!("frame read error from {}: {}", peer, e); client_gone = true; },
                Ok(None) => client_gone = true,
                Err(_) => { /* timeout: re-check mute state */ },
            }
        } else {
            let device_for_thread = input_device.clone();
            let delay = settings.delay;
            let threshold = settings.level.threshold();
            // Clone before the closure so `ui` stays usable on the next loop iteration.
            let ui_for_record = ui.clone();

            let mut rec_handle = tokio::task::spawn_blocking(move || {
                record_on_thread(&device_for_thread, delay, threshold, frame_rx, ui_for_record)
            });
            // Forward client frames while the recorder runs. The recorder decides when to stop:
            // it ends on trailing silence (after the minimum interval), hits a safety cap, or as
            // soon as one client frame reaches it. We race against the recorder and only await its
            // result once below.
            let mut finished: Option<Result<Result<IntervalResult, String>, tokio::task::JoinError>> = None;

            tokio::select! {
                res = &mut rec_handle => finished = Some(res),
                frame = framed.next() => match frame {
                    Some(Ok(b)) => {
                        if frame_tx.send(b.to_vec()).is_err() {
                            client_gone = true;
                        }
                    },
                    Some(Err(e)) => { warn!("frame read error from {}: {}", peer, e); client_gone = true; },
                    None => client_gone = true,
                },
            }

            drop(frame_tx);
            let joined = match finished {
                Some(res) => res,
                None => rec_handle.await,
            };
            let result = match joined {
                Ok(Ok(r))  => r,
                Ok(Err(e)) => { error!("recorder thread failed: {}", e); break; },
                Err(e) => { error!("recorder task panicked/joined with error: {}", e); break; },
            };

            // Last gate: if Mute was pressed after the recorder started, do not emit a
            // transport frame for this capture even though we have samples. Playback of any
            // queued client audio below still happens as usual.
            let muted_at_send = ui.is_muted();

            if should_emit(&result, client_gone, muted_at_send) {
                // Whisper input only needs about 16 kHz mono, so downmix and resample first.
                let mono = downmix_mono(&result.samples, channels);
                let clip = resample(&mono, sample_rate, TARGET_SAMPLE_RATE);
                let n_in = result.samples.len();
                let msg_bytes: Vec<u8> = clip.iter().flat_map(|s| s.to_le_bytes()).collect();
                // Prefix the audio with an 'a' marker so receivers know this is a plain audio frame.
                let id = Uuid::new_v4();
                let message = Message {
                    id,
                    text: String::new(),
                    audio: Message::audio_with_kind(MessageKind::Audio, &msg_bytes)
                };
                info!(
                    "recording sent to {} (id={}, {} -> {} samples @ {} Hz, ~{:.2}s)",
                    peer, id, n_in, clip.len(), TARGET_SAMPLE_RATE,
                    clip.len() as f64 / TARGET_SAMPLE_RATE as f64
                );
                if let Err(e) = framed.send(message.encode()).await {
                    warn!("send error to {}: {}", peer, e);
                    break;
                }
            } else if !result.samples.is_empty() && muted_at_send {
                debug!(
                    "mute active at send moment: skipping ~{:.2}s of captured audio",
                    result.samples.len() as f64 / sample_rate as f64 / channels as f64
                );
            } else if !result.samples.is_empty() && result.silent {
                debug!(
                    "recording has little useful signal (~{:.2}s), skipping send",
                    result.samples.len() as f64 / sample_rate as f64 / channels as f64
                );
            }

            pending_playback = result.pending_playback;
        }

        play_pending(&playback_tx, pending_playback);

        if client_gone {
            info!("client gone, stopping session for {}", peer);
            break;
        }
    }

    // Drop the worker sender so its channel closes and the thread exits, then mark idle in UI.
    drop(playback_tx);
    ui.set_connected(false);
    info!("client disconnected: {}", peer);
}

/// Reject a new connection when one is already active: write a short marker then close.
async fn reject_busy(stream: TcpStream, peer: SocketAddr) {
    let mut f = Framed::new(stream, LengthDelimitedCodec::new());
    warn!("busy, rejecting new connection from {}", peer);
    if let Err(e) = f.send(Bytes::from_static(b"busy")).await {
        debug!("failed to write busy marker to {}: {}", peer, e);
    }
}

/// Run the single-client TCP server: bind once and keep accepting, one session at a time.
pub async fn run_server(settings: Settings, ui: super::ui::UiState) {
    // Warn up front if no mic is available.
    if cpal::default_host().default_input_device().is_none() {
        warn!("no default input device found; recording will be skipped per session");
    }

    let listener = match TcpListener::bind(settings.listen_addr()).await {
        Ok(l) => l,
        Err(e) => { error!("failed to bind {}: {}", settings.listen_addr(), e); return; },
    };
    match listener.local_addr() {
        Ok(addr) => info!("listening on {} (record every {}s, one client at a time)", addr, settings.delay),
        Err(e) => warn!("could not read local address: {}", e),
    };

    // Single-client server: only one active session at any moment.
    let mut active: Option<JoinHandle<()>> = None;

    loop {
        // Reap a finished session so the slot opens up for the next client.
        if let Some(h) = &active {
            if h.is_finished() {
                active.take();
            }
        }

        let (stream, peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => { error!("accept failed: {}", e); break; },
        };

        match &active {
            // A client is already connected: reject and drop the newcomer.
            Some(_) => {
                if let Err(e) = tokio::spawn(reject_busy(stream, peer)).await {
                    warn!("busy-reject task failed: {}", e);
                }
            },
            None => active = Some(tokio::spawn(run_session(
                Framed::new(stream, LengthDelimitedCodec::new()),
                peer,
                settings.clone(),
                ui.clone(),
            ))),
        }
    }

    drop(active);
}

#[cfg(test)]
mod tests {
    use super::{downmix_mono, resample, signal_fill_ratio, upsample_to_device, IntervalResult, should_emit, TARGET_SAMPLE_RATE};

    #[test]
    fn resample_keeps_length_ratio_and_values() {
        // 44.1 kHz mono -> 16 kHz: a 2 s recording keeps ~2 s of audio.
        let src = vec![0.5f32; 88_200];
        let out = resample(&src, 44_100, TARGET_SAMPLE_RATE);
        assert_eq!(out.len(), 32_000 + 1);
        // Constant input stays constant after interpolation.
        for v in &out {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn resample_identity_when_rates_match() {
        let src = vec![0.1, 0.3, 0.5];
        let out = resample(&src, TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE);
        assert_eq!(out, src);
    }

    #[test]
    fn downmix_stereo_to_mono_averages_channels() {
        // Interleaved L/R: (1.0, 0.2) then (-0.4, 0.8) -> mono means per frame.
        let stereo = vec![1.0f32, -0.4, 0.2, 0.8];
        let mono = downmix_mono(&stereo, 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.3).abs() < 1e-6);   // (1.0 + -0.4) / 2
        assert!((mono[1] - 0.5).abs() < 1e-6);   // (0.2 + 0.8) / 2
    }

    #[test]
    fn upsample_to_device_restores_rate_and_channels() {
        // 1 s of mono 16 kHz speech -> ~48 samples per channel at a 48 kHz stereo device.
        let src = vec![0.5f32; 16_000];
        let out = upsample_to_device(&src, 48_000, 2);

        assert!(out.len() >= 96_000 && out.len() <= 96_102); // ~1 s of stereo @ 48k (+1 resample slack)
        assert_eq!(out.len() % 2, 0);
        for i in (0..out.len()).step_by(2) {
            assert!((out[i] - 0.5).abs() < 1e-6);
            assert_eq!(out[i], out[i + 1]); // both channels carry the same mono signal
        }
    }
    #[test]
    fn downmix_single_channel_is_noop() {
        let mono = vec![0.1, 0.2];
        assert_eq!(downmix_mono(&mono, 1), mono);
    }

    #[test]
    fn fill_ratio_all_silence_is_zero() {
        // 1 s @ 48 kHz with everything below threshold -> 0% useful signal.
        let samples = vec![0.001f32; 48_000];
        assert_eq!(signal_fill_ratio(&samples, 1, 48_000, 0.02), 0.0);
    }

    #[test]
    fn fill_ratio_full_signal_is_one() {
        // Every 10 ms window carries loud samples -> 100%.
        let samples = vec![0.5f32; 48_000];
        assert_eq!(signal_fill_ratio(&samples, 1, 48_000, 0.02), 1.0);
    }

    #[test]
    fn fill_ratio_short_transient_is_low() {
        // 50 ms of loud audio at the start of a 1 s clip: ~5 of 100 windows are active,
        // which must sit below MIN_SIGNAL_FILL_RATIO so brief clicks do not pass.
        let mut samples = vec![0.0f32; 48_000];
        samples[..2400].iter_mut().for_each(|s| *s = 0.6);

        assert!(signal_fill_ratio(&samples, 1, 48_000, 0.02) < super::MIN_SIGNAL_FILL_RATIO);
    }

    #[test]
    fn fill_ratio_stereo_counts_per_frame() {
        // One channel silent, one loud: each frame still registers signal because we take a max.
        let mut samples = vec![0.0f32; 48_000 * 2];
        for i in (1..samples.len()).step_by(2) {
            samples[i] = 0.5;
        }

        assert_eq!(signal_fill_ratio(&samples, 2, 48_000, 0.02), 1.0);
    }

    #[test]
    fn fill_ratio_short_buffer_below_window() {
        // Buffer shorter than one 10 ms frame is judged by its peak alone.
        assert_eq!(signal_fill_ratio(&[0.3f32; 8], 1, 48_000, 0.02), 1.0);
        assert_eq!(signal_fill_ratio(&[0.001f32; 8], 1, 48_000, 0.02), 0.0);
    }

    fn sample_result(silent: bool) -> IntervalResult {
        IntervalResult { samples: vec![0.5f32; 16_000], pending_playback: Vec::new(), silent }
    }

    #[test]
    fn emit_when_audio_present_and_not_muted() {
        assert!(should_emit(&sample_result(false), false, false));
    }

    #[test]
    fn regression_do_not_emit_inflight_capture_if_mute_pressed_at_send() {
        // Recording started before Mute was pressed; by the time we build/send the frame Mute is on.
        assert!(!should_emit(&sample_result(false), false, true));
    }

    #[test]
    fn do_not_emit_silent_or_empty_or_client_gone() {
        let silent = sample_result(true);
        let empty = IntervalResult { samples: Vec::new(), pending_playback: Vec::new(), silent: false };
        assert!(!should_emit(&silent, false, false));
        assert!(!should_emit(&empty, false, false));
        assert!(!should_emit(&sample_result(false), true, false));
    }

}
