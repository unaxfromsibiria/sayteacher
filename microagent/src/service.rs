use log::{debug, error, info, warn};
use std::num::NonZero;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rodio::{buffer::SamplesBuffer, mixer::Mixer, stream::DeviceSinkBuilder};

use crate::datatraffic::{Client, Config};

use super::settings::Settings;
use super::ui::UiState;

// Silence-detection tuning (peak amplitude, not RMS — avoids the sqrt).
const TAIL_SILENCE_MS: u64 = 1100; // trailing quiet required to end a recording
const MAX_RECORD_FACTOR: f64 = 8.0; // safety cap so an endless stream stops the recorder
const TAIL_POLL_MS: u64 = 160; // how often to re-check the tail for silence

// Minimum fraction of audio frames that must carry signal above threshold before we
// consider the interval useful. Keyboard clicks and other short transients fail this check.
const MIN_SIGNAL_FILL_RATIO: f32 = 0.10;

/// Speech recognition only needs ~16 kHz mono (whisper), so downsample before sending.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

// Poll timeout for each datatraffic client per service cycle.
const RECV_POLL_MS: u32 = 200;

// Retry cadence for re-sending the greeting/lesson until sayteacher confirms receipt.
const GREETING_INTERVAL_SECS: u64 = 10;

// If no heartbeat received within this window, mark sayteacher as disconnected.
const HB_STALENESS_SECS: u64 = 25;

// Muted cycle: short sleep instead of recording.
const MUTE_INTERVAL_MS: u64 = 500;

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
    /// True when the captured fragment is not worth sending (silent or mostly silence).
    silent: bool,
}

/// Convert incoming mono 16 kHz speech into interleaved device-rate multichannel samples.
fn upsample_to_device(samples: &[f32], to_rate: u32, channels: u16) -> Vec<f32> {
    let mono = resample(samples, TARGET_SAMPLE_RATE, to_rate);
    let ch = channels.max(1) as usize;
    (mono.iter()).flat_map(|s| std::iter::repeat(*s).take(ch)).collect()
}

/// Decide whether a captured interval should be emitted to sayteacher.
fn should_emit(result: &IntervalResult, muted_at_send: bool) -> bool {
    !result.samples.is_empty() && !result.silent && !muted_at_send
}

/// Inline blocking recording: captures mic for at least `delay_secs`, extends while audio is
/// still present, and stops once the tail goes quiet or a safety cap is hit. Mute ends early.
fn record_interval(
    input_device: &cpal::Device,
    delay_secs: f64,
    threshold: f32,
    ui: &UiState,
) -> Result<IntervalResult, String> {
    let config = input_device.default_input_config().map_err(
        |e| format!("input config error: {}", e)
    )?;
    let channels = config.channels();
    let sample_rate = config.sample_rate();

    // Track when sound was last heard so the tail can be tested for silence.
    let started = Instant::now();
    let last_sound: Arc<Mutex<Instant>> = Arc::new(Mutex::new(started));

    info!(
        "recording {}s @ {} Hz, {} ch (level threshold {:.4})",
        delay_secs, sample_rate, channels, threshold
    );

    // The audio callback keeps running until the stream is dropped below.
    let recorded: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(
        (delay_secs * sample_rate as f64 * channels as f64) as usize,
    )));
    let buf_cb = Arc::clone(&recorded);
    let last_sound_cb = Arc::clone(&last_sound);

    let stream = input_device.build_input_stream(
        config.config(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if !data.is_empty() && peak_amplitude(data) >= threshold {
                *last_sound_cb.lock().expect("last_sound poisoned") = Instant::now();
            }
            if let Ok(mut g) = buf_cb.lock() {
                g.extend_from_slice(data);
            }
        },
        |err| warn!("input stream error: {}", err),
        None,
    ).map_err(|e| format!("build_input_stream failed: {}", e))?;
    stream.play().map_err(|e| format!("stream play failed: {}", e))?;

    // Recording must not run forever if the audio keeps going.
    let delay_ms = (delay_secs * 1000.0) as u64;
    let max_record_ms = ((delay_secs * MAX_RECORD_FACTOR).max(delay_secs) * 1000.0) as u64;

    loop {
        let now = Instant::now();
        let elapsed_ms = now.duration_since(started).as_millis() as u64;

        // Stop recording as soon as the user presses Mute.
        if ui.is_muted() {
            info!("mute active, ending in-flight recording early (~{}s)", elapsed_ms / 1000);
            break;
        }

        // Before the minimum interval we keep recording unconditionally; after it we
        // poll more often so silence can end the capture, capped at max_record_secs.
        if elapsed_ms >= delay_ms {
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

        // Sleep briefly so we can re-check the tail and mute state.
        let sleep_ms = if elapsed_ms >= delay_ms { TAIL_POLL_MS } else { 50 };
        std::thread::sleep(Duration::from_millis(sleep_ms));
    }

    // Dropping the stream stops the audio callback so we own the buffer exclusively.
    drop(stream);
    let samples = match Arc::try_unwrap(recorded) {
        Ok(m) => m.into_inner().expect("record buffer poisoned"),
        Err(_) => return Err("record buffer still referenced".to_string()),
    };

    let peak = peak_amplitude(&samples);
    let fill = signal_fill_ratio(&samples, channels, sample_rate, threshold);
    // A clip is "silent" when there is no detectable sound at all (peak below threshold) or
    // when only a small fraction of the interval carried real signal.
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

    Ok(IntervalResult { samples, silent })
}

/// Poll both datatraffic clients for inbound frames. Routes dialogue frames to the UI and
/// queues audio for playback; updates connection state from heartbeats and lesson acks.
fn handle_inbound(
    dialogue_client: &mut Client,
    ready_client: &mut Client,
    sample_rate: u32,
    channels: u16,
    ui: &UiState,
    last_hb: &mut Option<Instant>,
    lesson_confirmed: &mut bool,
) -> Vec<PendingAudio> {
    let mut pending_playback = Vec::new();

    // Poll dialogue client for sayteacher responses (audio + text).
    if let Some(file) = dialogue_client.recv(RECV_POLL_MS) {
        match file.name.as_str() {
            "d" => {
                // Dialogue: text in note, audio in data.
                if !file.note.is_empty() {
                    info!("Dialogue from sayteacher:\n{}", file.note);
                    ui.append_dialogue(&file.note);
                }
                if !file.data.is_empty() {
                    let mono = f32_vec_from_le_bytes(&file.data);
                    let seconds = mono.len() as f64 / TARGET_SAMPLE_RATE as f64;
                    info!("received dialogue audio ~{seconds:.2}s ({} bytes)", file.data.len());
                    pending_playback.push(PendingAudio { samples: upsample_to_device(&mono, sample_rate, channels) });
                }
            },
            "n" => {
                // Note: text only.
                if !file.note.is_empty() {
                    info!("Note from sayteacher:\n{}", file.note);
                    ui.append_note(&file.note);
                }
            },
            "q" => {
                // sayteacher finished the lesson (LLM quit marker): stop recording like a mute press
                // so no stale segment is sent, and mark it in UI. "Try again" resumes with a fresh capture.
                info!("lesson ended by sayteacher");
                ui.set_lesson_ended(true);
                ui.set_muted(true);
                ui.append_dialogue("— lesson ended —");
            },
            other => {
                debug!("unexpected dialogue frame name: '{}'", other);
            },
        }
    }

    // Poll ready client for heartbeats and the lesson-set ack.
    if let Some(file) = ready_client.recv(RECV_POLL_MS) {
        match file.name.as_str() {
            "hb" => {
                // A fresh heartbeat means sayteacher is alive. It no longer clears a finished
                // lesson: after quit sayteacher keeps heartbeating while waiting for a re-sent
                // lesson, so "Try again" must stay visible until the user acts on it.
                *last_hb = Some(Instant::now());
                ui.set_connected(true);
                debug!("heartbeat received from sayteacher");
            },
            "lesson-set" => {
                // sayteacher stored the lesson; no need to keep re-sending it.
                if !*lesson_confirmed {
                    info!("sayteacher confirmed the lesson is set, stopping greeting re-sends");
                }
                *lesson_confirmed = true;
            },
            other => {
                debug!("unexpected ready-state frame name: '{}'", other);
            },
        }
    }

    pending_playback
}

/// Play queued audio inline using the rodio mixer. Blocks until all buffers finish.
fn play_pending(mixer: &Mixer, list: &mut Vec<PendingAudio>, channels: u16, sample_rate: u32) {
    if list.is_empty() {
        return;
    }
    let total: usize = list.iter().map(|p| p.samples.len()).sum();
    info!("playing back queued audio ({} samples)", total);

    for p in list.drain(..) {
        let buf = SamplesBuffer::new(
            NonZero::new(channels).expect("channels > 0"),
            NonZero::new(sample_rate).expect("sample_rate > 0"),
            p.samples,
        );
        let player = rodio::Player::connect_new(mixer);
        player.append(buf);
        player.sleep_until_end();
    }

    info!("playback finished");
}

/// Build a datatraffic Config for the given project and client name suffix.
fn make_config(settings: &Settings, project: &str, suffix: &str) -> Config {
    Config {
        redis_url: settings.redis_url.clone(),
        key_hex: settings.crypto_key.clone(),
        client_name: format!("{}-{}", settings.client_name, suffix),
        password: settings.redis_password.clone(),
        watch_projects: vec![project.to_string()],
        send_to: project.to_string(),
    }
}

/// Run the fully synchronous service loop on a dedicated OS thread.
pub fn run_server(settings: Settings, ui: UiState) {
    // Warn up front if no mic is available.
    let host = cpal::default_host();
    let Some(input_device) = host.default_input_device() else {
        warn!("no default input device found; recording will be skipped");
        return;
    };

    let (channels, sample_rate): (u16, u32) = match input_device.default_input_config() {
        Ok(c) => (c.channels(), c.sample_rate()),
        Err(e) => { error!("input config error: {}", e); return; }
    };

    // Open audio output once for the lifetime of this thread.
    let device_sink = match DeviceSinkBuilder::open_default_sink() {
        Ok(v) => v,
        Err(e) => { error!("failed to open audio output: {}", e); return; }
    };
    let mixer = device_sink.mixer();

    // Create datatraffic clients for both projects.
    let dialogue_cfg = make_config(&settings, "lesson-dialogue", "dlg");
    let ready_cfg = make_config(&settings, "ready-state", "rdy");

    info!(
        "connecting to datatraffic (redis_url='{}', client={})",
        if settings.redis_url.is_empty() { "(env REDIS_URL)" } else { &settings.redis_url },
        settings.client_name
    );

    let mut dialogue_client = match Client::new(&dialogue_cfg) {
        Ok(c) => c,
        Err(e) => { error!("failed to create dialogue client: {}", e); return; }
    };
    let mut ready_client = match Client::new(&ready_cfg) {
        Ok(c) => c,
        Err(e) => { error!("failed to create ready-state client: {}", e); return; }
    };

    info!("datatraffic clients connected (lesson-dialogue + ready-state)");

    // Send the initial greeting/lesson once; sayteacher acks it with a "lesson-set" frame.
    let mut last_greeting = Instant::now();
    let mut lesson_confirmed = false;
    if !settings.lesson_data.is_empty() {
        match ready_client.send("greeting", &settings.lesson_data, &[]) {
            Ok(_) => info!("sent initial greeting ({} chars)", settings.lesson_data.len()),
            Err(e) => warn!("failed to send initial greeting: {}", e),
        }
    }

    let mut last_hb: Option<Instant> = None;

    loop {
        // 1. Poll both clients for inbound frames.
        let mut pending_playback = handle_inbound(
            &mut dialogue_client, &mut ready_client,
            sample_rate, channels, &ui, &mut last_hb, &mut lesson_confirmed,
        );

        // Restart requested by the user ("Try again"): re-send the lesson immediately and reset
        // confirmation so sayteacher (waiting after quit) receives it and acks with "lesson-set".
        if ui.is_restart_requested() {
            info!("restart requested by user, re-sending lesson");
            if !settings.lesson_data.is_empty() {
                match ready_client.send("greeting", &settings.lesson_data, &[]) {
                    Ok(_) => debug!("re-sent greeting after restart ({} chars)", settings.lesson_data.len()),
                    Err(e) => warn!("failed to re-send greeting after restart: {}", e),
                }
            }
            lesson_confirmed = false;
            last_greeting = Instant::now();
            ui.set_restart_requested(false);
        }

        // Check heartbeat staleness: if no hb for >25s, mark disconnected and drop the
        // lesson confirmation so a restarted sayteacher receives the lesson again.
        if let Some(lh) = last_hb {
            if lh.elapsed() >= Duration::from_secs(HB_STALENESS_SECS) {
                ui.set_connected(false);
                lesson_confirmed = false;
                debug!("sayteacher heartbeat stale (>{}s), marking disconnected", HB_STALENESS_SECS);
            }
        }

        // 2. Re-send the greeting/lesson until sayteacher confirms it has the lesson.
        if !settings.lesson_data.is_empty() && !lesson_confirmed
            && last_greeting.elapsed() >= Duration::from_secs(GREETING_INTERVAL_SECS) {
            match ready_client.send("greeting", &settings.lesson_data, &[]) {
                Ok(_) => debug!("re-sent greeting ({} chars)", settings.lesson_data.len()),
                Err(e) => warn!("failed to re-send greeting: {}", e),
            }
            last_greeting = Instant::now();
        }

        // 3. Play back any audio received from sayteacher during this cycle (before recording).
        let had_playback = !pending_playback.is_empty();
        play_pending(mixer, &mut pending_playback, channels, sample_rate);

        // If we just played audio, skip recording and start a fresh capture next iteration.
        if had_playback {
            debug!("playback in progress: skipping mic capture this cycle");
            continue;
        }

        // 4. Record from mic (unless muted).
        if !ui.is_muted() {
            match record_interval(&input_device, settings.delay, settings.level.threshold(), &ui) {
                Ok(result) => {
                    let muted_at_send = ui.is_muted();
                    if should_emit(&result, muted_at_send) {
                        // Downmix and resample to 16 kHz mono for whisper.
                        let mono = downmix_mono(&result.samples, channels);
                        let clip = resample(&mono, sample_rate, TARGET_SAMPLE_RATE);
                        let n_in = result.samples.len();
                        let msg_bytes: Vec<u8> = clip.iter().flat_map(|s| s.to_le_bytes()).collect();

                        info!(
                            "sending recording to sayteacher ({} -> {} samples @ {} Hz, ~{:.2}s)",
                            n_in, clip.len(), TARGET_SAMPLE_RATE,
                            clip.len() as f64 / TARGET_SAMPLE_RATE as f64
                        );
                        if let Err(e) = dialogue_client.send("a", "", &msg_bytes) {
                            warn!("failed to send recording: {}", e);
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
                },
                Err(e) => error!("recording failed: {}", e),
            }
        } else {
            // Muted: short sleep instead of recording.
            debug!("muted: skipping mic capture this cycle");
            std::thread::sleep(Duration::from_millis(MUTE_INTERVAL_MS));
        }
    }
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
        IntervalResult { samples: vec![0.5f32; 16_000], silent }
    }

    #[test]
    fn emit_when_audio_present_and_not_muted() {
        assert!(should_emit(&sample_result(false), false));
    }

    #[test]
    fn regression_do_not_emit_inflight_capture_if_mute_pressed_at_send() {
        // Recording started before Mute was pressed; by the time we build/send the frame Mute is on.
        assert!(!should_emit(&sample_result(false), true));
    }

    #[test]
    fn do_not_emit_silent_or_empty() {
        let silent = sample_result(true);
        let empty = IntervalResult { samples: Vec::new(), silent: false };
        assert!(!should_emit(&silent, false));
        assert!(!should_emit(&empty, false));
    }
}
