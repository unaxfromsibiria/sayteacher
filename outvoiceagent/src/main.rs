mod datatraffic;
mod service;
mod settings;
mod tools;
mod ui;

use log::info;
use settings::Settings;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let settings = Settings::from_args();

    if settings.check_mode() {
        info!(
            "mode=check dir={} redis={}",
            settings.check_dir,
            settings.redis_url,
        );
        tools::run_check(settings.clone(), std::path::PathBuf::from(&settings.check_dir))?;
    } else if settings.client {
        info!(
            "mode=client redis={} delay={:.1}s",
            settings.redis_url,
            settings.delay,
        );
        tools::run_client(settings)?;
    } else {
        info!("mode=server redis={} delay={:.1}s", settings.redis_url, settings.delay);

        // One clone for the server thread, one for the egui overlay window.
        let ui_server = ui::UiState::default();
        let ui_app = ui_server.clone();

        // Server runs on its own plain thread (synchronous datatraffic loop).
        std::thread::Builder::new().name("server".into()).spawn(move || {
            service::run_server(settings, ui_server);
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
            Box::new(|_cc| Ok(Box::new(ui::OverlayApp::new(ui_app)))),
        ) {
            log::warn!("overlay window closed with error: {}", e);
        }

        // The server thread is still alive here; exiting ends the process and tears it down.
        std::process::exit(0);
    }

    Ok(())
}
