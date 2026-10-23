use log::{error, info};

use crate::datatraffic::{Client, Config};

use super::settings::Settings;

/// Echo client: connects to the lesson-dialogue project via datatraffic, reads frames,
/// logs their size and echoes each audio frame back with a summary in the note field.
pub fn run_client(settings: Settings) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config {
        redis_url: settings.redis_url.clone(),
        key_hex: settings.crypto_key.clone(),
        client_name: format!("{}-echo", settings.client_name),
        password: settings.redis_password.clone(),
        watch_projects: vec!["lesson-dialogue".into()],
        send_to: "lesson-dialogue".into(),
    };

    info!(
        "connecting to datatraffic (redis_url='{}', client={}-echo)",
        if cfg.redis_url.is_empty() { "(env REDIS_URL)" } else { &cfg.redis_url },
        settings.client_name
    );

    let mut client = Client::new(&cfg)?;
    info!("connected to lesson-dialogue project");

    loop {
        match client.recv(1000) {
            Some(file) => {
                if file.name == "greeting" {
                    info!("Lesson: {}", file.note);
                    continue;
                }

                let audio_len = file.data.len();
                info!(
                    "received name='{}' note='{}' data_bytes={}",
                    file.name, file.note, audio_len
                );

                // Echo back the same audio with a summary in the note field.
                if !file.data.is_empty() {
                    let reply_note = format!("echo: {} bytes", audio_len);
                    info!("echoing back note='{}'", reply_note);
                    if client.send(&file.name, &reply_note, &file.data).is_err() {
                        error!("send failed; closing");
                        break;
                    }
                } else if !file.note.is_empty() {
                    // Text-only frame: echo the note back.
                    let reply = format!("echo: {}", file.note);
                    info!("echoing text note='{}'", reply);
                    if client.send(&file.name, &reply, &[]).is_err() {
                        error!("send failed; closing");
                        break;
                    }
                }
            },
            None => {
                // Timeout: no frame within 1s. Keep polling (the echo client runs until killed).
            },
        }
    }

    info!("disconnected from lesson-dialogue");
    Ok(())
}
