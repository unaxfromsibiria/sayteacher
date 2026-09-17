use futures_util::{SinkExt, StreamExt};
use log::{error, info, warn};
use tokio::net::TcpStream;
use tokio_util::codec::{Framed, LengthDelimitedCodec};

use super::settings::Settings;
use microagent_transport::{Bytes, Message};

/// Echo client: connects to the server, reads messages, logs their size and
/// echoes each one back with a short summary in the text field.
pub async fn run_client(settings: Settings) -> Result<(), Box<dyn std::error::Error>> {
    let addr = settings.connect_addr_str();
    info!("connecting to server at {}", addr);

    let stream = TcpStream::connect(&addr).await.map_err(|e| format!(
        "failed to connect to {}: {}", addr, e
    ))?;
    info!("connected to {}", addr);
    let mut framed: Framed<TcpStream, LengthDelimitedCodec> = Framed::new(stream, LengthDelimitedCodec::new());
    // Handshake: server sends a greeting carrying the session id; client replies with
    // sha256(id + password) in the text field. Without this round-trip the server drops us.
    let Some(greet_frame) = framed.next().await else {
        warn!("server closed before handshake");
        return Ok(());
    };
    let greet_raw = match greet_frame {
        Ok(b) => b.to_vec(),
        Err(e) => {
            warn!("greeting read error: {}", e);
            return Ok(());
        }
    };
    let greet = match Message::decode(&greet_raw) {
        Ok(m) => m,
        Err(e) => {
            warn!("bad greeting from server ({} bytes): {}", greet_raw.len(), e);
            return Ok(());
        }
    };
    let token = Message::auth_token_for(&greet.id, &settings.password);
    info!("sending auth handshake (id={} token_len={})", greet.id, token.len());
    let reply = Message { id: greet.id, text: token, audio: Bytes::new() };
    if framed.send(reply.encode()).await.is_err() {
        error!("auth send failed; closing");
        return Ok(());
    }

    while let Some(frame) = framed.next().await {
        let raw = match frame {
            Ok(b) => b.to_vec(),
            Err(e) => {
                warn!("frame read error: {}", e);
                break;
            }
        };
        let msg = match Message::decode(&raw) {
            Ok(m) => m,
            // The server sends a short "busy" marker when rejecting extra clients.
            Err(e) => {
                warn!("undecodable frame ({} bytes): {}", raw.len(), e);
                continue;
            }
        };
        // Lesson: empty audio + non-empty text sent right after auth. Log it and do not echo.
        if msg.audio.is_empty() && !msg.text.is_empty() {
            info!("Lesson: {}", msg.text);
            continue;
        }
        let audio_len = msg.audio.len();
        info!(
            "received message id={} text='{}' audio_bytes={}",
            msg.id, msg.text, audio_len
        );
        // Echo back the same record (id + audio) with a summary in the text field.
        let reply_text = format!("record {} length: {}", msg.id, audio_len);
        info!("echoing back text='{}'", reply_text);
        let reply = Message { id: msg.id, text: reply_text, audio: msg.audio };

        if framed.send(reply.encode()).await.is_err() {
            error!("send to server failed; closing");
            break;
        }
    }

    info!("disconnected from server");
    Ok(())
}
