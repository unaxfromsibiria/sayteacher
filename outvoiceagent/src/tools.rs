use log::{debug, info, warn};
use std::path::PathBuf;

use super::datatraffic::Client;
use super::settings::Settings;
use super::service::TARGET_SAMPLE_RATE;

/// How long (in seconds) to wait for the next file before treating the link as idle and exiting.
const IDLE_TIMEOUT_SECONDS: u64 = 180;
/// Poll interval in milliseconds; doubles as the sleep tick between recv calls.
const RECV_POLL_MS: u32 = 100;

/// Test client: watches `out-audio` for captured audio segments and prints each one until
/// nothing arrives within the idle timeout. Runs synchronously on the calling thread.
pub fn run_client(settings: Settings) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = settings.client_config();
    info!(
        "connecting to datatraffic ({}), watching {}",
        cfg.redis_url,
        cfg.watch_projects.join(",")
    );

    let mut client = Client::new(&cfg)
        .map_err(|e| format!("failed to create datatraffic client: {}", e))?;
    info!("connected");

    // Consecutive timeouts (recv returns None) until we declare the link idle.
    let max_idle_polls = IDLE_TIMEOUT_SECONDS * 1000 / RECV_POLL_MS as u64;
    let mut idle_count: u64 = 0;

    loop {
        match client.recv(RECV_POLL_MS) {
            Some(file) => {
                idle_count = 0;
                log_inbound(&file)?;
            }
            None => {
                idle_count += 1;
                if idle_count >= max_idle_polls {
                    info!(
                        "no file from server in {:.0}s, disconnecting",
                        IDLE_TIMEOUT_SECONDS as f64
                    );
                    break;
                }
            }
        }
    }

    client.close();
    info!("disconnected");
    Ok(())
}

/// Log a received audio segment: its name (id) and approximate duration.
fn log_inbound(file: &super::datatraffic::DataFile) -> Result<(), Box<dyn std::error::Error>> {
    let body_len = file.data.len();
    if body_len == 0 {
        debug!("received empty audio segment (name={})", file.name);
        return Ok(());
    }

    // f32 LE mono @ TARGET_SAMPLE_RATE: seconds ≈ bytes / 4 / rate.
    let secs = body_len as f64 / 4.0 / TARGET_SAMPLE_RATE as f64;
    info!(
        "received audio (name={}): {} bytes (~{:.2}s)",
        file.name, body_len, secs
    );

    Ok(())
}

/// Check-mode client: watches `out-audio` and saves every received segment as a PCM16 mono WAV
/// file named out_<id>.wav in `dir`. No playback. Runs synchronously on the calling thread.
pub fn run_check(settings: Settings, dir: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if std::fs::create_dir_all(&dir).is_err() {
        warn!("could not create directory {}, saving to the current directory", dir.display());
    }

    let cfg = settings.client_config();
    info!(
        "check mode: watching {} ({}), saving segments into {} as out_<id>.wav",
        cfg.watch_projects.join(","),
        cfg.redis_url,
        dir.display()
    );

    let mut client = Client::new(&cfg)
        .map_err(|e| format!("failed to create datatraffic client: {}", e))?;
    info!("connected");

    let max_idle_polls = IDLE_TIMEOUT_SECONDS * 1000 / RECV_POLL_MS as u64;
    let mut idle_count: u64 = 0;
    let mut saved = 0usize;

    loop {
        match client.recv(RECV_POLL_MS) {
            Some(file) => {
                idle_count = 0;
                save_output_frame(&dir, &file);
                saved += 1;
            }
            None => {
                idle_count += 1;
                if idle_count >= max_idle_polls {
                    info!(
                        "no file from server in {:.0}s, disconnecting",
                        IDLE_TIMEOUT_SECONDS as f64
                    );
                    break;
                }
            }
        }
    }

    client.close();
    info!("disconnected; {} segment(s) saved to {}", saved, dir.display());
    Ok(())
}

/// Decode f32 LE mono samples into an in-memory PCM16 WAV file (44-byte header + payload).
fn wav_from_samples(samples_f32: &[f32], sample_rate: u32) -> Vec<u8> {
    let mut pcm16: Vec<u8> = Vec::with_capacity(samples_f32.len() * 2);
    for &s in samples_f32 {
        let v = s.clamp(-1.0, 1.0);
        let i = (v * i16::MAX as f32).round() as i16;
        pcm16.extend_from_slice(&i.to_le_bytes());
    }

    // RIFF header + fmt chunk + data chunk: 44 bytes before the payload.
    let mut buf = Vec::with_capacity(44 + pcm16.len());
    let size_32 = 36u32 + pcm16.len() as u32;
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&size_32.to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt "); // RIFF marker, then fmt chunk header
    buf.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size (PCM)
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format code
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate as u32 * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(pcm16.len() as u32).to_le_bytes());
    buf.extend_from_slice(&pcm16);

    buf
}

/// When `file` carries audio data, write it as out_<name>.wav (PCM16 mono @ TARGET_SAMPLE_RATE).
fn save_output_frame(dir: &PathBuf, file: &super::datatraffic::DataFile) {
    if file.data.is_empty() {
        return;
    }

    // f32 LE samples (outvoiceagent sends mono @ TARGET_SAMPLE_RATE); truncate any tail bytes.
    let body = &file.data;
    let n = body.len() / 4;
    let samples: Vec<f32> = body[..n * 4].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let buf = wav_from_samples(&samples, TARGET_SAMPLE_RATE);
    let path = dir.join(format!("out_{}.wav", file.name));
    match std::fs::write(&path, &buf) {
        Ok(()) => info!(
            "saved {} -> {} ({} bytes, ~{:.2}s)",
            file.name, path.display(), buf.len() - 44, n as f64 / TARGET_SAMPLE_RATE as f64
        ),
        Err(e) => warn!("failed to write {}: {}", path.display(), e),
    }
}

#[cfg(test)]
mod tests {
    use super::{save_output_frame, wav_from_samples, TARGET_SAMPLE_RATE};
    use crate::datatraffic::DataFile;

    #[test]
    fn wav_header_layout_is_riff_pcm16_mono() {
        let buf = wav_from_samples(&[0.5f32, -0.5], 48_000);
        assert_eq!(&buf[..4], b"RIFF");
        assert_eq!(&buf[8..12], b"WAVE");
        assert_eq!(&buf[12..16], b"fmt ");
        // fmt: PCM(1), 1 channel, 48000 Hz, byte rate 96000, block align 2, 16 bit.
        let fmt = &buf[16..36];
        assert_eq!(&fmt[..4], &(16u32.to_le_bytes())); // chunk size
        assert_eq!(&fmt[4..6], &[1u8, 0]);             // format code PCM
        assert_eq!(&fmt[6..8], &[1u8, 0]);             // channels
        assert_eq!(u32::from_le_bytes(fmt[8..12].try_into().unwrap()), 48_000);
        assert_eq!(u32::from_le_bytes(fmt[12..16].try_into().unwrap()), 96_000);
        assert_eq!(&fmt[16..18], &[2u8, 0]);           // block align
        assert_eq!(&fmt[18..20], &[16u8, 0]);          // bits per sample
        assert_eq!(&buf[36..40], b"data");
        let data_len = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        assert_eq!(data_len as usize + 44, buf.len());
    }

    #[test]
    fn wav_pcm16_values_match_input() {
        let buf = wav_from_samples(&[1.0f32, -1.0, 0.0], 16_000);
        assert_eq!(i16::from_le_bytes(buf[44..46].try_into().unwrap()), i16::MAX);
        assert_eq!(i16::from_le_bytes(buf[46..48].try_into().unwrap()), -32767);
        assert_eq!(i16::from_le_bytes(buf[48..50].try_into().unwrap()), 0);
    }

    #[test]
    fn save_output_frame_writes_audio_files() {
        let dir = std::env::temp_dir().join(format!("outvoice_check_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // A file with audio data -> one WAV file.
        let payload: Vec<u8> = [0.25f32, -0.25].iter().flat_map(|s| s.to_le_bytes()).collect();
        let audio_file = DataFile { name: "42".into(), note: String::new(), data: payload };
        save_output_frame(&dir, &audio_file);

        // A file without audio data -> nothing saved.
        let empty_file = DataFile { name: "43".into(), note: "text only".into(), data: Vec::new() };
        save_output_frame(&dir, &empty_file);

        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
        assert_eq!(files.len(), 1, "only the audio file must be saved");
        let name = files[0].file_name().to_string_lossy().into_owned();
        assert!(name.starts_with("out_") && name.ends_with(".wav"), "unexpected name: {}", name);

        // The stored file is a valid WAV whose sample rate matches TARGET_SAMPLE_RATE.
        let bytes = std::fs::read(&dir.join(name)).unwrap();
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), TARGET_SAMPLE_RATE);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
