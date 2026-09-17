mod service;
mod settings;
mod tools;
mod ui;

use log::info;
use settings::Settings;
use ui::{MicroagentApp, UiState, window_title_from_lesson};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let settings = Settings::from_args();

    // Window title: first non-empty lesson line with '#' characters stripped from the edges.
    let window_title = window_title_from_lesson(&settings.lesson_data);
    info!(
        "window title: '{}' ({} chars lesson data)", window_title, settings.lesson_data.len()
    );

    if settings.client {
        let auth = !settings.password.is_empty();
        info!(
            "mode=client connect={} delay={}s level={} auth={}",
            settings.connect_addr_str(),
            settings.delay,
            settings.level,
            auth
        );
        tools::run_client(settings).await?;
    } else {
        // One clone for the server thread, one for the egui window. Both share the same Arc-backed state.
        let ui_server = UiState::default();
        let ui_app = ui_server.clone();
        // Server runs on its own thread + runtime so the egui event loop owns the main thread.
        let builder = std::thread::Builder::new().name("server".into());
        let _handle = builder.spawn(move || {
            match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
                Ok(rt) => rt.block_on(service::run_server(settings, ui_server)),
                Err(e) => log::error!("failed to build server runtime: {}", e),
            }
        }).expect("failed to spawn server thread");

        // Wider default so two text columns (notes + dialogue) fit comfortably side by side.
        let vp = egui::ViewportBuilder::default().with_title(
            window_title.clone()
        ).with_inner_size([1080.0, 520.0]).with_min_inner_size([720.0, 360.0]);
        let options = eframe::NativeOptions {
            viewport: vp,
            ..Default::default()
        };
        if let Err(e) = eframe::run_native(
            &window_title,
            options,
            Box::new(|_cc| Ok(Box::new(MicroagentApp::new(ui_app)))),
        ) {
            log::warn!("egui window closed with error: {}", e);
        }

        // The server thread is still alive here; exiting ends the process and tears it down.
        std::process::exit(0);
    }

    Ok(())
}
