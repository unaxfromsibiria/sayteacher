mod service;
mod settings;
mod tools;
mod ui;

use log::info;
use settings::Settings;
use std::sync::{Arc, atomic::AtomicBool};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let settings = Settings::from_args();

    if settings.check_mode() {
        info!(
            "mode=check dir={} connect={}",
            settings.check_dir,
            settings.connect_addr_str(),
        );
        tools::run_check(settings.clone(), std::path::PathBuf::from(&settings.check_dir)).await?;
    } else if settings.client {
        info!(
            "mode=client connect={} delay={:.1}s",
            settings.connect_addr_str(),
            settings.delay,
        );
        tools::run_client(settings).await?;
    } else {
        info!("mode=server listen={} delay={:.1}s", settings.listen_addr(), settings.delay);

        // One clone for the server thread, one for the egui overlay window; both share Arc-backed state.
        let ui_server = ui::UiState::default();
        let ui_app = ui_server.clone();
        // Shared shutdown flag: set by the server on SIGINT/SIGTERM so the overlay can close cleanly.
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_for_server = shutdown.clone();

        // Server runs on its own thread + runtime so the egui event loop owns the main thread.
        std::thread::Builder::new().name("server".into()).spawn(move || {
            match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
                Ok(rt) => rt.block_on(service::run_server(settings, ui_server, shutdown_for_server)),
                Err(e) => log::error!("failed to build server runtime: {}", e),
            }
        }).expect("failed to spawn server thread");

        let mut vp = egui::ViewportBuilder::default().with_title("outvoiceagent")
            .with_transparent(true).with_decorations(false).with_always_on_top();
        if let Some((pos, size)) = ui::resolve_window_geometry() {
            vp = vp.with_position([pos.x, pos.y]).with_inner_size([size.x, size.y]);
        } else {
            vp = vp.with_inner_size([1536.0, 216.0]).with_position([96.0, 720.0]);
        }
        let options = eframe::NativeOptions {
            viewport: vp,
            ..Default::default()
        };

        if let Err(e) = eframe::run_native(
            "outvoiceagent",
            options,
            Box::new(|_cc| Ok(Box::new(ui::OverlayApp::new(ui_app, shutdown)))),
        ) {
            log::warn!("overlay window closed with error: {}", e);
        }

        // The server thread is still alive here; exiting ends the process and tears it down.
        std::process::exit(0);
    }

    Ok(())
}
