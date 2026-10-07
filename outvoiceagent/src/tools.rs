use futures_util::StreamExt;
use log::{debug, info, warn};
use std::path::PathBuf;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_util::codec::{Framed, LengthDelimitedCodec};

use super::settings::Settings;
use super::service::TARGET_SAMPLE_RATE;
use microagent_transport::{Message, MessageKind};
/// How long to wait for the next frame before treating the link as idle and disconnecting.
const IDLE_TIMEOUT_SECONDS: f64 = 180.0;

/// Test client: connects to outvoiceagent and prints every transport frame it receives
/// until the server closes the connection or nothing arrives within the idle timeout.
pub async fn run_client(settings: Settings) -> Result<(), Box<dyn std::error::Error>> {
    let addr = settings.connect_addr_str();
    info!("connecting to server at {}", addr);

    let stream = TcpStream::connect(&addr).await.map_err(|e| format!(
        "failed to connect to {}: {}", addr, e
    ))?;
    info!("connected to {}", addr);

    let mut framed: Framed<TcpStream, LengthDelimitedCodec> = Framed::new(stream, LengthDelimitedCodec::new());
    let idle_timeout = Duration::from_secs_f64(IDLE_TIMEOUT_SECONDS);

    loop {
        // A single bounded await keeps the client from hanging on a silent server.
        match tokio::time::timeout(idle_timeout, framed.next()).await {
            Ok(Some(Ok(raw))) => log_inbound(&raw)?,
            Ok(Some(Err(e))) => {
                warn!("frame read error: {}", e);
                break;
            }
            Ok(None) => {
                info!("server closed the connection");
                break;
            }
            Err(_) => {
                info!("no frame from server in {:.0}s, disconnecting", IDLE_TIMEOUT_SECONDS);
                break;
            }
        }
    }

    info!("disconnected");
    Ok(())
}

/// Log a received transport frame: its text and/or captured audio payload.
fn log_inbound(raw: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let msg = Message::decode(raw).map_err(
        |e| format!("undecodable frame ({} bytes): {}", raw.len(), e)
    )?;

    if !msg.text.is_empty() {
        info!("received text (id={}, kind={:?}): {}", msg.id, msg.kind(), msg.text);
    } else if let Some(kind) = msg.kind() {
        // Captured audio arrives at ~16 kHz mono f32: seconds ≈ bytes / 4 / 16_000.
        let body_len = msg.audio_body().len();
        let secs = body_len as f64 / 4.0 / 16_000.0;
        info!("received audio (id={}, kind={:?}): {} bytes (~{:.2}s)", msg.id, kind, body_len, secs);
    } else {
        debug!("received empty frame (id={})", msg.id);
    }

    Ok(())
}

/// Check-mode client: connects to the server and saves every received output-frame
/// ('o' marker) as a PCM16 mono WAV file named out_<uuid>.wav in `dir`. No playback.
pub async fn run_check(settings: Settings, dir: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if std::fs::create_dir_all(&dir).is_err() {
        warn!("could not create directory {}, saving to the current directory", dir.display());
    }

    info!(
        "check mode: saving output frames from {} into {} as out_<uuid>.wav",
        settings.connect_addr_str(),
        dir.display()
    );

    let addr = settings.connect_addr_str();
    let stream = TcpStream::connect(&addr).await.map_err(|e| format!(
        "failed to connect to {}: {}", addr, e
    ))?;
    info!("connected to {}", addr);

    let mut framed: Framed<TcpStream, LengthDelimitedCodec> = Framed::new(stream, LengthDelimitedCodec::new());
    let idle_timeout = Duration::from_secs_f64(IDLE_TIMEOUT_SECONDS);
    let mut saved = 0usize;

    loop {
        match tokio::time::timeout(idle_timeout, framed.next()).await {
            Ok(Some(Ok(raw))) => {
                if let Some(msg) = Message::decode(&raw).ok() {
                    save_output_frame(&dir, &msg);
                    saved += 1;
                } else {
                    warn!("undecodable frame ({} bytes), skipping", raw.len());
                }
            },
            Ok(Some(Err(e))) => {
                warn!("frame read error: {}", e);
                break;
            }
            Ok(None) => {
                info!("server closed the connection");
                break;
            }
            Err(_) => {
                info!("no frame from server in {:.0}s, disconnecting", IDLE_TIMEOUT_SECONDS);
                break;
            }
        }
    }

    info!("disconnected; {} output frame(s) saved to {}", saved, dir.display());
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

/// When `msg` carries an 'o'-marked payload, write it as out_<uuid>.wav (PCM16 mono).
fn save_output_frame(dir: &PathBuf, msg: &Message) {
    if msg.kind() != Some(MessageKind::Output) || msg.audio_body().is_empty() {
        return;
    }

    // f32 LE samples (outvoiceagent sends mono @ TARGET_SAMPLE_RATE); truncate any tail bytes.
    let body = msg.audio_body();
    let n = body.len() / 4;
    let samples: Vec<f32> = body[..n * 4].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let buf = wav_from_samples(&samples, TARGET_SAMPLE_RATE);
    let id = msg.id;
    let path = dir.join(format!("out_{}.wav", id));
    match std::fs::write(&path, &buf) {
        Ok(()) => info!(
            "saved {} -> {} ({} bytes, ~{:.2}s)",
            msg.id, path.display(), buf.len() - 44, n as f64 / TARGET_SAMPLE_RATE as f64
        ),
        Err(e) => warn!("failed to write {}: {}", path.display(), e),
    }
}

#[cfg(test)]
mod tests {
    use super::{save_output_frame, wav_from_samples, TARGET_SAMPLE_RATE};
    use microagent_transport::{Message, MessageKind, Uuid};

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
    fn save_output_frame_writes_only_o_frames() {
        let dir = std::env::temp_dir().join(format!("outvoice_check_test_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let payload: Vec<u8> = [0.25f32, -0.25].iter().flat_map(|s| s.to_le_bytes()).collect();

        // 'o' frame -> one WAV file; other kinds -> nothing.
        let out_msg = Message {
            id: Uuid::new_v4(),
            text: String::new(),
            audio: Message::audio_with_kind(MessageKind::Output, &payload),
        };
        save_output_frame(&dir, &out_msg);
        let note_msg = Message {
            id: Uuid::new_v4(),
            text: "hi".into(),
            audio: Message::audio_with_kind(MessageKind::Note, &payload),
        };
        save_output_frame(&dir, &note_msg);

        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
        assert_eq!(files.len(), 1, "only the 'o' frame must be saved");
        let name = files[0].file_name().to_string_lossy().into_owned();
        assert!(name.starts_with("out_") && name.ends_with(".wav"), "unexpected name: {}", name);

        // The stored file is a valid WAV whose samples round-trip to 16-bit values.
        let bytes = std::fs::read(&dir.join(name)).unwrap();
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), TARGET_SAMPLE_RATE);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
