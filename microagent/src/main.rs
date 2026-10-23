mod datatraffic;
mod service;
mod settings;
mod tools;
mod ui;

use log::info;
use settings::Settings;
use ui::{MicroagentApp, UiState, window_title_from_lesson};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let settings = Settings::from_args();

    // Window title: first non-empty lesson line with '#' characters stripped from the edges.
    let window_title = window_title_from_lesson(&settings.lesson_data);
    info!(
        "window title: '{}' ({} chars lesson data)", window_title, settings.lesson_data.len()
    );

    if settings.client {
        info!(
            "mode=client delay={}s level={}",
            settings.delay, settings.level
        );
        tools::run_client(settings)?;
    } else {
        // One clone for the server thread, one for the egui window. Both share the same Arc-backed state.
        let ui_server = UiState::default();
        let ui_app = ui_server.clone();

        // Server runs on its own OS thread (fully synchronous, no tokio).
        let builder = std::thread::Builder::new().name("server".into());
        let _handle = builder.spawn(move || {
            service::run_server(settings, ui_server);
        }).expect("failed to spawn server thread");

        // Wider default so two text columns (notes + dialogue) fit comfortably side by side.
        let mut vp = egui::ViewportBuilder::default().with_title(
            window_title.clone()
        ).with_min_inner_size([720.0, 360.0]);
        if let Some((pos, size)) = ui::resolve_window_geometry() {
            vp = vp.with_position([pos.x, pos.y]).with_inner_size([size.x, size.y]);
        } else {
            vp = vp.with_inner_size([1080.0, 520.0]);
        }
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
