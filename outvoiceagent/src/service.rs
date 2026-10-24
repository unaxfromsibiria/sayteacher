use log::{debug, error, info, warn};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use super::datatraffic::{Client, DataFile};
use super::settings::{Settings, AUDIO_PROJECT, TEXT_PROJECT};
use super::ui::UiState;

/// Recognition downstream needs ~16 kHz mono (whisper), so resample before sending.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

// Pause-splitting tuning for the smart end-of-segment cut (see split_band / find_best_pause).
const PAUSE_WINDOW_MS: u32 = 60; // smoothing window size, ~50-75 ms per spec
const TAIL_SEARCH_FRACTION: f64 = 0.15; // search only the trailing fraction of a capture for the cut
const TAIL_ACTIVE_MS: u64 = 300; // how far back to look when extending while signal is active
const MIN_SIGNAL_FILL_RATIO: f32 = 0.02; // min share of samples that carry real audio
const WELL_FILLED_RATIO: f32 = 0.10; // fill ratio above which a segment counts as well filled
const TEMP_FILL_RATIO: f32 = 0.005; // temporary low fill threshold after well-filled segments

/// How long a single inbound poll may block while we wait out the capture interval or extend it.
/// Doubles as the sleep tick, so translation groups are routed to the UI within this window.
const RECV_POLL_MS: u32 = 100;

/// Fraction of mono samples whose |value| reaches `threshold` — cheap "is this real audio" test.
fn mono_fill_ratio(mono: &[f32], threshold: f32) -> f32 {
    if mono.is_empty() {
        return 0.0;
    }
    let active = mono.iter().filter(|s| s.abs() >= threshold).count();
    (active as f32 / mono.len() as f32).min(1.0)
}

/// Peak absolute amplitude of a sample buffer — cheap silence test (no sqrt).
fn peak_amplitude(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0_f32, |m, s| m.max(s.abs()))
}

/// Sample range where the end-of-segment cut may fall. Searching only the trailing
/// `TAIL_SEARCH_FRACTION` of what was captured keeps every sent segment near its full capture
/// length and leaves a small remainder to carry into the next one, so no short segments appear.
fn split_band(mono_len: usize, sample_rate: u32, delay: f64) -> Option<(usize, usize)> {
    let lo = ((delay * (1.0 - TAIL_SEARCH_FRACTION) * sample_rate as f64).floor() as i64 + 1).max(1);
    let hi = (mono_len.saturating_sub(1)) as i64;
    if hi < lo {
        None
    } else {
        Some((lo as usize, hi as usize))
    }
}

/// Find the best end-of-segment cut point inside `split_band` by scanning a smoothing window
/// (~PAUSE_WINDOW_MS) across that band and picking the index with the lowest smoothed peak-amplitude
/// — i.e. the quietest stretch of the longest duration available there. Returns None when no valid
/// cut exists (e.g. the buffer is too short to hold even one coarse step).
fn find_best_pause(mono: &[f32], sample_rate: u32, delay: f64) -> Option<usize> {
    let n = mono.len();
    if n < 2 || sample_rate == 0 {
        return None;
    }

    // Smoothing window in samples.
    let window = ((sample_rate as u64 * PAUSE_WINDOW_MS as u64) / 1000).max(1);
    // Coarse stride keeps the scan cheap while staying well below the band width.
    let step = (window / 4).max(1);

    // Build coarse per-position peak levels; each level is the peak over a ~PAUSE_WINDOW_MS window
    // starting at start, so an isolated dip reads lower than sustained signal.
    let mut levels: Vec<f32> = Vec::new();
    for start in (0..n).step_by(step as usize) {
        let end = ((start as u64 + window).min(n as u64)) as usize;
        levels.push(peak_amplitude(&mono[start..end]));
    }

    // Smooth by folding in the immediate neighbours so an isolated dip is not preferred over a
    // long quiet region; this naturally favours "lowest level of longest duration".
    let m = levels.len();
    let smoothed: Vec<f32> = (0..m)
        .map(|i| {
            let left = if i > 0 { levels[i - 1] } else { 0.0 };
            let right = if i + 1 < m { levels[i + 1] } else { 0.0 };
            levels[i] + left + right
        })
        .collect();

    // Map the split band (samples) onto coarse indices.
    let (band_lo_sample, band_hi_sample) = split_band(n, sample_rate, delay)?;
    let lo = (band_lo_sample as u64 / step).min(m as u64 - 1);
    let hi = (band_hi_sample as u64 / step).min(m as u64 - 1);

    // Argmin of the smoothed levels inside the band; ties resolve to the earlier index.
    let mut best_i = lo as usize;
    let mut best_v = smoothed[best_i];
    for i in (lo + 1..=hi).map(|i| i as usize) {
        if smoothed[i] < best_v {
            best_i = i;
            best_v = smoothed[i];
        }
    }

    // Convert the coarse index back to a mono sample offset and snap it into the declared band so
    // callers can rely on band_lo <= cut <= band_hi exactly (no misleading fallback logging).
    let raw = best_i * step as usize;
    Some(raw.max(band_lo_sample).min(band_hi_sample)).filter(|&cut| cut >= 1 && cut < n)
}

/// First sample index where a ~PAUSE_WINDOW_MS window reaches `threshold`, or None when the whole
/// buffer stays below it. Coarse (step-sized), like find_best_pause; returns the window start so a
/// little pre-signal is kept, never signal itself.
fn signal_onset(mono: &[f32], sample_rate: u32, threshold: f32) -> Option<usize> {
    let n = mono.len();
    if n == 0 || sample_rate == 0 {
        return None;
    }
    let window = ((sample_rate as u64 * PAUSE_WINDOW_MS as u64) / 1000).max(1);
    let step = (window / 4).max(1);
    for start in (0..n).step_by(step as usize) {
        let end = ((start as u64 + window).min(n as u64)) as usize;
        if peak_amplitude(&mono[start..end]) >= threshold {
            return Some(start);
        }
    }
    None
}

/// Drop leading silence from `carry` once it has grown beyond one capture interval. A long pause
/// must not accumulate a quiet backlog that delays the next useful segment: anything before the
/// first onset would fail the fill gate anyway (its peak is below threshold). An all-silent carry
/// is cleared entirely so the caller falls into the ping path.
fn trim_leading_silence(carry: &mut Vec<f32>, delay_samples: usize, threshold: f32) {
    if carry.len() <= delay_samples {
        return;
    }
    match signal_onset(carry, TARGET_SAMPLE_RATE, threshold) {
        Some(onset) if onset > 0 => {
            debug!(
                "dropping {:.2}s of leading silence from carry",
                onset as f64 / TARGET_SAMPLE_RATE as f64
            );
            carry.drain(..onset);
        }
        None => {
            debug!(
                "carry is all silence ({:.2}s), clearing it",
                carry.len() as f64 / TARGET_SAMPLE_RATE as f64
            );
            carry.clear();
        }
        _ => {}
    }
}

/// True when at least `lookback` worth of trailing samples carries signal above `threshold`,
/// used to extend capture while the source is still active (mirrors microagent's tail check).
fn tail_active(buffer: &Arc<Mutex<Vec<f32>>>, channels: u16, sample_rate: u32, threshold: f32) -> bool {
    let ch = (channels.max(1)) as usize;
    let lookback = ((TAIL_ACTIVE_MS as f64 * sample_rate as f64 * ch as f64) as usize).max(1);
    if let Ok(g) = buffer.lock() {
        if g.is_empty() {
            return false;
        }
        let start = g.len().saturating_sub(lookback);
        peak_amplitude(&g[start..]) >= threshold
    } else {
        false
    }
}

/// Current number of captured (interleaved) samples, 0 if the buffer is poisoned.
fn buffer_len(buffer: &Arc<Mutex<Vec<f32>>>) -> usize {
    buffer.lock().map(|g| g.len()).unwrap_or(0)
}

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

/// Select an audio capture device by name or auto-select.
/// On Linux prefers input devices with "monitor" in the name; otherwise falls back to default output.
fn select_device(host: &cpal::Host, wanted_name: &str) -> Option<cpal::Device> {
    if !wanted_name.is_empty() && wanted_name != "auto" {
        // Exact match on device name (case-insensitive), only for devices that support the direction.
        for dev in host.input_devices().ok().into_iter().flatten() {
            if let Ok(desc) = dev.description() {
                if desc.supports_input() && desc.name().eq_ignore_ascii_case(wanted_name) {
                    return Some(dev);
                }
            }
        }
        for dev in host.output_devices().ok().into_iter().flatten() {
            if let Ok(desc) = dev.description() {
                if desc.supports_output() && desc.name().eq_ignore_ascii_case(wanted_name) {
                    return Some(dev);
                }
            }
        }
        warn!("device '{}' not found, falling back to auto", wanted_name);
        print_available_devices(host);
    }

    // Auto-select: prefer monitor input devices on Linux.
    if let Ok(inputs) = host.input_devices() {
        for dev in inputs {
            if let Ok(desc) = dev.description() {
                if desc.supports_input() && desc.name().to_lowercase().contains("monitor") {
                    return Some(dev);
                }
            }
        }
    }

    // Fallback: default output device (loopback on Windows).
    if let Some(dev) = host.default_output_device() {
        return Some(dev);
    }

    // Last resort: first input device.
    if let Ok(mut inputs) = host.input_devices() && let Some(dev) = inputs.next() {
        return Some(dev);
    }

    None
}

fn print_available_devices(host: &cpal::Host) {
    info!("available input devices (supports_input):");
    if let Ok(inputs) = host.input_devices() {
        for dev in inputs {
            if let Ok(desc) = dev.description() {
                if desc.supports_input() {
                    let name = desc.name();
                    let is_monitor = name.to_lowercase().contains("monitor");
                    let is_virtual = desc.device_type() == cpal::DeviceType::Virtual;
                    if is_monitor || is_virtual {
                        info!("  input: {} | name='{}'", name, name);
                    }
                }
            }
        }
    }
    info!("available output devices (supports_output):");
    if let Ok(outputs) = host.output_devices() {
        for dev in outputs {
            if let Ok(desc) = dev.description() {
                if desc.supports_output() {
                    let name = desc.name();
                    info!("  output: {} | name='{}'", name, name);
                }
            }
        }
    }
}

/// Push captured samples (as f32) into the shared buffer from the audio callback thread.
fn push(buffer: &Mutex<Vec<f32>>, data: &[f32]) {
    if data.is_empty() { return; }
    if let Ok(mut g) = buffer.lock() {
        g.extend_from_slice(data);
    }
}

/// Open a loopback (output-device input stream) capture on the given device.
fn open_capture(device: &cpal::Device, buffer: Arc<Mutex<Vec<f32>>>) -> Result<cpal::Stream, String> {
    let config = device
        .default_input_config()
        .or_else(|_| device.default_output_config())
        .map_err(|e| format!("config error: {}", e))?;

    match config.sample_format() {
        cpal::SampleFormat::F32 => device
            .build_input_stream(
                config.config(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| push(&buffer, data),
                |err| warn!("capture stream error: {}", err),
                None,
            )
            .map_err(|e| format!("build_input_stream failed: {}", e)),
        cpal::SampleFormat::I16 => device
            .build_input_stream(
                config.config(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let f32_data: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                    push(&buffer, &f32_data);
                },
                |err| warn!("capture stream error: {}", err),
                None,
            )
            .map_err(|e| format!("build_input_stream failed: {}", e)),
        other => Err(format!("unsupported output sample format: {:?}", other)),
    }
}

/// Atomically take everything accumulated in the buffer since the last call.
fn drain_buffer(buffer: &Arc<Mutex<Vec<f32>>>) -> Vec<f32> {
    if let Ok(mut g) = buffer.lock() {
        std::mem::take(&mut *g)
    } else {
        Vec::new()
    }
}

/// Route one inbound translation-group file to the UI overlay. The filename is a zero-padded group
/// counter (e.g. "007"); an empty note clears that group's stale text, otherwise it sets/updates it.
fn handle_inbound_group(file: &DataFile, ui: &UiState) {
    let gid = match file.name.trim().parse::<u32>() {
        Ok(g) => g,
        Err(_) => {
            debug!("translation file without a numeric name ({:?}), ignoring", file.name);
            return;
        }
    };

    if file.note.is_empty() {
        ui.set_group_text(gid, "");
        info!("cleared recognition group {}", gid);
    } else {
        let is_new = ui.set_group_text(gid, &file.note);
        info!(
            "{} recognition group {}: {} word(s), {} char(s): {}",
            if is_new { "new" } else { "updated" },
            gid,
            file.note.split_whitespace().count(),
            file.note.chars().count(),
            file.note
        );
    }
}

/// Publish one already-split segment to the audio project. `mono` is f32 mono @ TARGET_SAMPLE_RATE;
/// returns false on a send error. The filename is a short incrementing counter (id).
fn send_segment(client: &Client, seq: u64, mono: &[f32]) -> bool {
    if mono.is_empty() {
        return true;
    }

    let bytes: Vec<u8> = mono.iter().flat_map(|s| s.to_le_bytes()).collect();
    match client.send(&seq.to_string(), "", &bytes) {
        Ok(()) => {
            info!(
                "published {} bytes of captured audio to {} (id={}, ~{:.2}s @ {} Hz)",
                bytes.len(), AUDIO_PROJECT, seq, mono.len() as f64 / TARGET_SAMPLE_RATE as f64, TARGET_SAMPLE_RATE
            );
        }
        Err(e) => {
            warn!("send error: {}", e);
            return false;
        }
    }
    true
}

/// Dynamic fill gate: after well-filled segments a temporary low fill threshold is armed,
/// so the tail of speech (low fill) is still sent instead of being skipped. The temp
/// threshold disarms after it lets a segment through or after a fully silent segment.
struct FillGate {
    temp_armed: bool,
}

impl FillGate {
    fn new() -> Self {
        Self { temp_armed: false }
    }

    /// Returns true when the segment should be sent; updates the temp-threshold state.
    fn check(&mut self, peak: f32, fill: f32, threshold: f32) -> bool {
        if peak >= threshold && fill >= MIN_SIGNAL_FILL_RATIO {
            if fill >= WELL_FILLED_RATIO {
                self.temp_armed = true;
            }
            return true;
        }
        if self.temp_armed && peak >= threshold && fill >= TEMP_FILL_RATIO {
            debug!(
                "segment passes the temporary low-fill threshold (fill={:.1}%), sending",
                fill * 100.0
            );
            self.temp_armed = false;
            return true;
        }
        self.temp_armed = false;
        false
    }
}

/// Run the capture-and-publish loop: watch `trans-text` for translation groups (routed to the UI)
/// and publish each captured, split audio segment to `out-audio`. Runs until the process exits.
pub fn run_server(settings: Settings, ui: UiState) {
    let cfg = settings.server_config();
    let mut client = match Client::new(&cfg) {
        Ok(c) => c,
        Err(e) => {
            error!("failed to create datatraffic client ({}): {}", TEXT_PROJECT, e);
            return;
        }
    };
    info!(
        "outvoiceagent server: watching {} for translation groups, publishing captured audio to {} every {:.1}s",
        TEXT_PROJECT, AUDIO_PROJECT, settings.delay
    );

    // A fresh run must not show text left over from the previous transcreenpt session.
    ui.clear_groups();

    let host = cpal::default_host();
    if settings.device.is_empty() {
        print_available_devices(&host);
    }
    let device = select_device(&host, &settings.device);
    let Some(device) = device else {
        warn!("no capture device found; capturing nothing");
        return;
    };
    let device_name = device.description().map(|d| d.name().to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());
    let config = match device.default_input_config().or_else(|_| device.default_output_config()) {
        Ok(c) => c,
        Err(e) => {
            error!("config error: {}", e);
            return;
        },
    };
    let channels = config.channels();
    let sample_rate = config.sample_rate();

    // Shared capture buffer filled by the cpal callback thread.
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(
        (settings.delay * sample_rate as f64 * channels as f64) as usize,
    )));

    let capture = match open_capture(&device, Arc::clone(&buffer)) {
        Ok(c) => c,
        Err(e) => {
            error!("failed to start loopback capture: {}", e);
            return;
        },
    };
    if let Err(e) = capture.play() {
        error!("{}", e);
        return;
    }

    info!(
        "capturing output device '{}' @ {} Hz ({} ch), publishing every {:.1}s",
        device_name, sample_rate, channels, settings.delay
    );

    let delay = Duration::from_secs_f64(settings.delay);
    // Audio captured in the previous cycle but not yet sent — carried to the front of this one.
    // Carry is always mono @ TARGET_SAMPLE_RATE.
    let mut carry: Vec<f32> = Vec::new();
    // Dynamic fill gate so the low-fill tail of speech is not dropped.
    let mut fill_gate = FillGate::new();
    // Short id/counter for each published audio segment (datatraffic filename).
    let mut audio_seq: u64 = 0;

    loop {
        // Phase 1a: wait out the full `delay` of capture. Inbound translation groups are routed to
        // the UI as they arrive but do NOT end the wait — recv blocks up to RECV_POLL_MS and doubles
        // as the sleep tick, so a reply arriving early no longer truncates the segment.
        let deadline = Instant::now() + delay;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let poll_ms = (remaining.as_millis().min(RECV_POLL_MS as u128)) as u32;
            if let Some(file) = client.recv(poll_ms.max(1)) {
                handle_inbound_group(&file, &ui);
            }
        }

        // Phase 1b: minimum reached — extend while a recent tail still has signal above threshold,
        // polling every RECV_POLL_MS so we stop as soon as silence appears. Hard cap at delay * 1.4
        // mirrors microagent's MAX_RECORD_FACTOR and keeps endless speech from starving the split.
        let min_sec = settings.delay;
        let max_sec = settings.delay * 1.4;

        loop {
            let buffer_samples = buffer_len(&buffer);
            let buffer_sec = buffer_samples as f64 / (sample_rate as f64 * channels as f64);
            let carry_sec = carry.len() as f64 / TARGET_SAMPLE_RATE as f64;
            let total_sec = carry_sec + buffer_sec;
            let signal_active = tail_active(&buffer, channels, sample_rate, settings.level.threshold());
            if total_sec >= min_sec && signal_active && total_sec < max_sec {
                debug!(
                    "signal active at ~{:.2}s (cap {:.2}s), extending capture",
                    total_sec,
                    max_sec
                );
                // Poll for inbound translation groups; the recv block doubles as the extend sleep.
                if let Some(file) = client.recv(RECV_POLL_MS) {
                    handle_inbound_group(&file, &ui);
                }
            } else {
                break;
            }
        }

        // Phase 2: pull whatever the cpal thread has buffered and convert to mono @ TARGET_SAMPLE_RATE.
        let fresh_raw = drain_buffer(&buffer);
        if !fresh_raw.is_empty() {
            let fresh_mono = resample(
                &downmix_mono(&fresh_raw, channels),
                sample_rate,
                TARGET_SAMPLE_RATE,
            );
            carry.extend_from_slice(&fresh_mono);
        }

        // Keep the carry bounded: a long pause must not pile up quiet audio that the fill gate would
        // reject anyway and that delays the next useful segment (post-pause stall).
        let delay_samples = (settings.delay * TARGET_SAMPLE_RATE as f64) as usize;
        trim_leading_silence(&mut carry, delay_samples, settings.level.threshold());

        if carry.is_empty() {
            debug!("no audio captured this interval");
            continue;
        }

        let peak_raw = peak_amplitude(&carry);
        let fill_raw = mono_fill_ratio(&carry, settings.level.threshold());
        info!(
            "segment captured: duration={:.2}s peak={:.4} fill={:.1}% (mono @ {} Hz)",
            carry.len() as f64 / TARGET_SAMPLE_RATE as f64,
            peak_raw,
            fill_raw * 100.0,
            TARGET_SAMPLE_RATE
        );

        // Phase 3: carry is already mono @ TARGET_SAMPLE_RATE, no further conversion needed.
        let mut mono = std::mem::take(&mut carry);

        // Phase 4: cut in the trailing ~TAIL_SEARCH_FRACTION of this capture so each sent segment keeps
        // nearly all of what we captured and only a small remainder carries into the next one (no short
        // segments). If the buffer is too short to hold that band, send it whole with no carry.
        let Some((lo, hi)) = split_band(mono.len(), TARGET_SAMPLE_RATE, settings.delay) else {
            debug!(
                "segment shorter than the search band (~{:.2}s), sending whole thing",
                mono.len() as f64 / TARGET_SAMPLE_RATE as f64
            );
            let head = std::mem::take(&mut mono);
            let threshold = settings.level.threshold();
            let peak = peak_amplitude(&head);
            let fill = mono_fill_ratio(&head, threshold);
            debug!(
                "segment stats before send: duration={:.2}s peak={:.4} fill={:.1}% threshold={:.4}",
                head.len() as f64 / TARGET_SAMPLE_RATE as f64,
                peak,
                fill * 100.0,
                threshold
            );
            if fill_gate.check(peak, fill, threshold) {
                info!(
                    "segment useful (peak={:.4}, fill={:.1}%), publishing ~{:.2}s",
                    peak,
                    fill * 100.0,
                    head.len() as f64 / TARGET_SAMPLE_RATE as f64
                );
                audio_seq += 1;
                if !send_segment(&client, audio_seq, &head) {
                    break;
                }
            } else {
                debug!(
                    "segment ~{:.2}s not useful (peak={:.4}, fill={:.1}%), skipping send",
                    head.len() as f64 / TARGET_SAMPLE_RATE as f64,
                    peak,
                    fill * 100.0
                );
            }
            carry.clear();
            continue;
        };

        let cut = find_best_pause(&mono, TARGET_SAMPLE_RATE, settings.delay).unwrap_or(hi);
        if cut >= lo && cut <= hi {
            info!(
                "split at sample {} (~{:.2}s), band [{}..={}] (delay={:.1}s)",
                cut,
                cut as f64 / TARGET_SAMPLE_RATE as f64,
                lo,
                hi,
                settings.delay
            );
        } else {
            info!(
                "no quiet split in band [{}..={}]; falling back to sample {} (~{:.2}s)",
                lo, hi, cut, cut as f64 / TARGET_SAMPLE_RATE as f64
            );
        }

        let (head, tail) = mono.split_at(cut);
        carry = tail.to_vec();

        let threshold = settings.level.threshold();
        let peak = peak_amplitude(head);
        let fill = mono_fill_ratio(head, threshold);
        debug!(
            "segment stats before send: duration={:.2}s peak={:.4} fill={:.1}% threshold={:.4}",
            head.len() as f64 / TARGET_SAMPLE_RATE as f64,
            peak,
            fill * 100.0,
            threshold
        );

        if fill_gate.check(peak, fill, threshold) {
            info!(
                "segment useful (peak={:.4}, fill={:.1}%), publishing ~{:.2}s",
                peak,
                fill * 100.0,
                head.len() as f64 / TARGET_SAMPLE_RATE as f64
            );
            audio_seq += 1;
            if !send_segment(&client, audio_seq, head) {
                break;
            }
        } else {
            debug!(
                "segment ~{:.2}s not useful (peak={:.4}, fill={:.1}%), skipping send",
                head.len() as f64 / TARGET_SAMPLE_RATE as f64,
                peak,
                fill * 100.0
            );
        }

        if !carry.is_empty() {
            info!(
                "carrying {} samples (~{:.2}s) into next segment",
                carry.len(),
                carry.len() as f64 / TARGET_SAMPLE_RATE as f64
            );
        }
    }

    // Stop the capture stream once we exit (process teardown).
    drop(capture);
}

#[cfg(test)]
mod tests {
    use super::{
        downmix_mono, find_best_pause, handle_inbound_group, mono_fill_ratio, peak_amplitude,
        resample, signal_onset, split_band, trim_leading_silence, FillGate, TAIL_SEARCH_FRACTION,
        TARGET_SAMPLE_RATE,
    };
    use crate::datatraffic::DataFile;
    use crate::ui::UiState;

    #[test]
    fn handle_inbound_group_routes_translation_files_to_ui() {
        let ui = UiState::default();
        // A translation group file: name is the zero-padded counter, note is the text.
        let file = DataFile { name: "007".into(), note: "привет мир, как дела".into(), data: Vec::new() };
        handle_inbound_group(&file, &ui);
        assert_eq!(ui.combined_text(), "привет мир, как дела");

        // A shorter refresh for the same group is a stale/delayed result and must not clobber it.
        let file2 = DataFile { name: "007".into(), note: "привет мир".into(), data: Vec::new() };
        handle_inbound_group(&file2, &ui);
        assert_eq!(ui.combined_text(), "привет мир, как дела");

        // A file without a numeric name is not routed to any group.
        let file3 = DataFile { name: "plain".into(), note: "note text".into(), data: Vec::new() };
        handle_inbound_group(&file3, &ui);
        assert_eq!(ui.combined_text(), "привет мир, как дела");

        // An empty note clears the group's stale text.
        let file4 = DataFile { name: "007".into(), note: String::new(), data: Vec::new() };
        handle_inbound_group(&file4, &ui);
        assert_eq!(ui.combined_text(), "");
    }

    #[test]
    fn resample_keeps_length_ratio() {
        let src = vec![0.5f32; 48_000];
        let out = resample(&src, 48_000, TARGET_SAMPLE_RATE);
        assert!(out.len() >= 15_999 && out.len() <= 16_001);
    }

    #[test]
    fn resample_identity_when_rates_match() {
        let src = vec![0.1, 0.3, 0.5];
        assert_eq!(resample(&src, TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE), src);
    }

    #[test]
    fn downmix_stereo_averages_channels() {
        let stereo = vec![1.0f32, -0.4, 0.2, 0.8]; // two frames of L/R
        let mono = downmix_mono(&stereo, 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.3).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn downmix_single_channel_is_noop() {
        let mono = vec![0.1, 0.2];
        assert_eq!(downmix_mono(&mono, 1), mono);
    }

    #[test]
    fn peak_amplitude_returns_max_abs() {
        assert_eq!(peak_amplitude(&[0.1, -0.5, 0.3]), 0.5);
        assert_eq!(peak_amplitude(&[]), 0.0);
    }

    #[test]
    fn fill_ratio_all_zero_is_zero() {
        let silent = vec![0.0f32; 16_000];
        assert_eq!(mono_fill_ratio(&silent, 0.02), 0.0);
    }

    #[test]
    fn fill_ratio_all_loud_is_one() {
        let loud = vec![0.5f32; 16_000];
        assert_eq!(mono_fill_ratio(&loud, 0.02), 1.0);
    }

    #[test]
    fn fill_gate_detects_real_audio() {
        // 0.5 s of loud audio at 16 kHz is well above threshold and fill ratio.
        let mono = vec![0.4f32; 8_000];
        let mut gate = FillGate::new();
        assert!(gate.check(peak_amplitude(&mono), mono_fill_ratio(&mono, 0.02), 0.02));
    }

    #[test]
    fn fill_gate_rejects_silence() {
        // Everything below threshold → not useful.
        let mono = vec![0.001f32; 8_000];
        let mut gate = FillGate::new();
        assert!(!gate.check(peak_amplitude(&mono), mono_fill_ratio(&mono, 0.02), 0.02));
    }

    #[test]
    fn fill_gate_sends_low_fill_tail_after_well_filled() {
        let mut gate = FillGate::new();
        // A well-filled segment is sent and arms the temp threshold.
        assert!(gate.check(0.5, 0.30, 0.005));
        // The low-fill tail of speech (below the main 2% fill) is then sent.
        assert!(gate.check(0.5, 0.01, 0.005));
    }

    #[test]
    fn fill_gate_drops_low_fill_without_prior_speech() {
        let mut gate = FillGate::new();
        // No well-filled segment before: the low-fill tail is not sent.
        assert!(!gate.check(0.5, 0.01, 0.005));
    }

    #[test]
    fn fill_gate_temp_threshold_is_one_shot() {
        let mut gate = FillGate::new();
        assert!(gate.check(0.5, 0.30, 0.005));
        assert!(gate.check(0.5, 0.01, 0.005));
        // Disarmed after use: the next low-fill segment is dropped.
        assert!(!gate.check(0.5, 0.01, 0.005));
    }

    #[test]
    fn fill_gate_disarms_on_silence() {
        let mut gate = FillGate::new();
        assert!(gate.check(0.5, 0.30, 0.005));
        // A fully silent segment disarms the temp threshold.
        assert!(!gate.check(0.001, 0.0, 0.005));
        // A low-fill segment after silence is dropped.
        assert!(!gate.check(0.5, 0.01, 0.005));
    }

    #[test]
    fn fill_gate_rearms_on_next_well_filled_segment() {
        let mut gate = FillGate::new();
        assert!(gate.check(0.5, 0.30, 0.005));
        assert!(gate.check(0.5, 0.01, 0.005)); // tail sent, disarmed
        // Speech resumes: a well-filled segment re-arms the temp threshold.
        assert!(gate.check(0.5, 0.25, 0.005));
        assert!(gate.check(0.5, 0.01, 0.005)); // the next tail is sent again
    }

    #[test]
    fn split_band_targets_the_trailing_fraction() {
        // A full capture of ~delay should give a band confined to the last TAIL_SEARCH_FRACTION,
        // so any cut keeps at least (1 - fraction) of the segment and carry stays small.
        let delay = 7.0;
        let sr = TARGET_SAMPLE_RATE;
        let n = (delay * sr as f64) as usize; // exactly one capture interval
        let (lo, hi) = split_band(n, sr, delay).unwrap();

        assert!(hi == n - 1, "band end {} should reach the buffer end", hi);
        let min_sent_samples = lo; // head is [0..cut), cut >= lo -> head keeps at least lo samples
        assert!((min_sent_samples as f64) >= delay * (1.0 - TAIL_SEARCH_FRACTION) * sr as f64 - 2.0);
    }

    #[test]
    fn split_band_is_none_when_buffer_shorter_than_delay() {
        // Buffer shorter than the (trailing) band start -> no in-band cut, send whole thing.
        let delay = 7.0;
        let sr = TARGET_SAMPLE_RATE;
        assert!(split_band((1.0 * sr as f64) as usize, sr, delay).is_none());
    }

    #[test]
    fn find_best_pause_stays_in_trailing_fraction() {
        // 7 s of steady audio, delay = 7 s: the cut must land in the last TAIL_SEARCH_FRACTION so the
        // sent head is near a full interval (no short segments) and carry stays small.
        let sr = TARGET_SAMPLE_RATE;
        let delay = 7.0;
        let mono = vec![0.2f32; (delay * sr as f64) as usize];
        let cut = find_best_pause(&mono, sr, delay).unwrap();

        let band_lo = ((delay * (1.0 - TAIL_SEARCH_FRACTION) * sr as f64) as usize).max(1);
        assert!(cut >= band_lo, "cut {} before trailing-fraction start {}", cut, band_lo);
        // head keeps the bulk of the capture: at least 80% of delay.
        let sent_sec = cut as f64 / sr as f64;
        assert!(sent_sec >= delay * 0.8 - 0.15, "head {:.2}s too short for delay {}", sent_sec, delay);
    }

    #[test]
    fn find_best_pause_prefers_quiet_stretch() {
        // 6 s @ 16 kHz: loud (0.5) from t=0..3, silent afterwards with a 100 ms blip at t=4s.
        // delay = 5 s -> the trailing band sits in silence; argmin should be near its start.
        let sr = TARGET_SAMPLE_RATE;
        let dur = 6.0;
        let n = (dur * sr as f64) as usize;
        let mut mono = vec![0.5f32; n];
        for s in &mut mono[((3.0 * sr as f64) as usize)..] { *s = 0.0; }
        let blip_start = (4.0 * sr as f64) as usize;
        let blip_end = (4.1 * sr as f64) as usize;
        for s in &mut mono[blip_start..blip_end] { *s = 0.3; }

        let delay = 5.0;
        let cut = find_best_pause(&mono, sr, delay).unwrap();

        // Trailing band start is after the blip (>= 4.25 s), so the whole search region is quiet and a
        // valid in-band cut must exist past it.
        assert!(cut >= ((delay * (1.0 - TAIL_SEARCH_FRACTION) * sr as f64) as usize).max(1));
        assert!(cut < n, "cut {} past end", cut);
    }

    #[test]
    fn find_best_pause_returns_none_when_band_is_at_start() {
        // When the buffer is so short that lo and hi both clamp to index 0, no valid cut exists
        // (cut must be >=1), so the function reports None — callers can then use a fallback.
        let mono = vec![0.5f32; 30];   // ~2 ms @ 16 kHz
        assert!(find_best_pause(&mono, TARGET_SAMPLE_RATE, 1.0).is_none());
    }

    #[test]
    fn find_best_pause_returns_some_for_a_typical_buffer() {
        // A normal 5 s buffer at delay=4s produces a cut inside the band; this is the common case.
        let mono = vec![0.3f32; (5.0 * TARGET_SAMPLE_RATE as f64) as usize];
        assert!(find_best_pause(&mono, TARGET_SAMPLE_RATE, 4.0).is_some());
    }

    #[test]
    fn signal_onset_finds_first_signal() {
        let mut mono = vec![0.0f32; (5.0 * TARGET_SAMPLE_RATE as f64) as usize]; // 5 s silence
        mono.extend(std::iter::repeat(0.5f32).take((1.0 * TARGET_SAMPLE_RATE as f64) as usize));
        let onset = signal_onset(&mono, TARGET_SAMPLE_RATE, 0.005).unwrap();
        // Coarse scan: within one smoothing window of the true start (80000), never past it.
        assert!(onset <= (5.0 * TARGET_SAMPLE_RATE as f64) as usize);
        assert!(onset >= (5.0 * TARGET_SAMPLE_RATE as f64) as usize - 960);

        // All-silent buffer reports no onset; a leading signal reports index 0.
        let silent = vec![0.0f32; (5.0 * TARGET_SAMPLE_RATE as f64) as usize];
        assert!(signal_onset(&silent, TARGET_SAMPLE_RATE, 0.005).is_none());
        let loud_first = vec![0.5f32; (1.0 * TARGET_SAMPLE_RATE as f64) as usize];
        assert_eq!(signal_onset(&loud_first, TARGET_SAMPLE_RATE, 0.005), Some(0));
    }

    #[test]
    fn trim_leading_silence_clears_all_silent_carry() {
        let mut mono = vec![0.0f32; (10.0 * TARGET_SAMPLE_RATE as f64) as usize]; // 10 s > delay
        trim_leading_silence(&mut mono, (4.0 * TARGET_SAMPLE_RATE as f64) as usize, 0.005);
        assert!(mono.is_empty());
    }

    #[test]
    fn trim_leading_silence_drops_quiet_prefix() {
        let mut mono = vec![0.0f32; (5.0 * TARGET_SAMPLE_RATE as f64) as usize]; // 5 s silence
        mono.extend(std::iter::repeat(0.5f32).take((4.0 * TARGET_SAMPLE_RATE as f64) as usize));
        let delay_samples = (4.0 * TARGET_SAMPLE_RATE as f64) as usize;
        trim_leading_silence(&mut mono, delay_samples, 0.005);
        // The quiet prefix is gone up to one smoothing window of tolerance; the loud tail survives.
        let dropped = (9.0 * TARGET_SAMPLE_RATE as f64) as usize - mono.len();
        assert!(dropped >= (5.0 * TARGET_SAMPLE_RATE as f64) as usize - 960, "only {} dropped", dropped);
        assert_eq!(mono.last(), Some(&0.5f32));
    }

    #[test]
    fn trim_leading_silence_ignores_short_carry() {
        // Below one capture interval the existing fill-gate path handles quiet audio; leave it alone.
        let mut mono = vec![0.0f32; (3.0 * TARGET_SAMPLE_RATE as f64) as usize];
        trim_leading_silence(&mut mono, (4.0 * TARGET_SAMPLE_RATE as f64) as usize, 0.005);
        assert_eq!(mono.len(), (3.0 * TARGET_SAMPLE_RATE as f64) as usize);
    }

    #[test]
    fn trim_leading_silence_keeps_signal_start() {
        let mut mono = vec![0.5f32; (10.0 * TARGET_SAMPLE_RATE as f64) as usize];
        trim_leading_silence(&mut mono, (4.0 * TARGET_SAMPLE_RATE as f64) as usize, 0.005);
        assert_eq!(mono.len(), (10.0 * TARGET_SAMPLE_RATE as f64) as usize);
    }
}
