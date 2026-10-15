use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Shared state between the session loop (server thread) and the egui window (main thread).
/// Cheaply cloneable — both sides hold an Arc to the same data.
#[derive(Clone)]
pub struct UiState {
    muted: Arc<std::sync::atomic::AtomicBool>,
    connected: Arc<std::sync::atomic::AtomicBool>,
    notes: Arc<Mutex<Vec<String>>>,
    dialogue: Arc<Mutex<Vec<String>>>
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            muted: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            connected: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            notes: Arc::default(),
            dialogue: Arc::default(),
        }
    }
}

/// Window title: first lesson line that, after trimming `#` characters from both edges and
/// whitespace, is non-empty. Falls back to "microagent" when nothing qualifies.
pub fn window_title_from_lesson(lesson_data: &str) -> String {
    for line in lesson_data.lines() {
        let cleaned = line.trim().trim_matches('#').trim();
        if !cleaned.is_empty() {
            return cleaned.to_string();
        }
    }

    "microagent".to_string()
}

const MAX_LINES: usize = 500;

/// Resolve window position/size centered on the primary monitor before the first frame.
pub fn resolve_window_geometry() -> Option<(egui::Pos2, egui::Vec2)> {
    let monitors = xcap::Monitor::all().ok()?;
    let monitor = monitors.iter().find(|m| m.is_primary().unwrap_or(false)).or_else(|| monitors.first())?;

    let mon_w = monitor.width().ok()? as f32;
    let mon_h = monitor.height().ok()? as f32;
    let mon_x = monitor.x().ok()? as f32;
    let mon_y = monitor.y().ok()? as f32;

    let w = 1080.0f32.min(mon_w - 40.0);
    let h = 520.0f32.min(mon_h - 40.0);
    let x = mon_x + (mon_w - w) / 2.0;
    let y = mon_y + (mon_h - h) / 2.0;

    Some((egui::Pos2::new(x, y), egui::Vec2::new(w, h)))
}

/// Separator inserted in the notes field between consecutive messages.
pub const NOTE_SEPARATOR: &str = "----------";

fn trim_and_cap(guard: &mut Vec<String>) {
    let drop_n = guard.len().saturating_sub(MAX_LINES);
    if drop_n > 0 {
        guard.drain(0..drop_n);
    }

    // Capping may chop a message in half and leave its separator as the first entry.
    if guard.first().map(String::as_str) == Some(NOTE_SEPARATOR) {
        guard.remove(0);
    }
}

/// Append one whole note message, preceded by a separator when the field is not empty.
/// Both steps happen under one lock so concurrent appends never interleave.
fn append_note_message(lines: &Mutex<Vec<String>>, text: &str) {
    let mut guard = lines.lock().expect("ui notes poisoned");

    if !guard.is_empty() {
        guard.push(NOTE_SEPARATOR.to_string());
    }

    for line in text.split('\n') {
        if !line.trim().is_empty() {
            guard.push(line.to_string());
        }
    }

    trim_and_cap(&mut guard);
}

fn append_dialogue_lines(lines: &Mutex<Vec<String>>, text: &str) {
    let mut guard = lines.lock().expect("ui dialogue poisoned");

    for line in text.split('\n') {
        if !line.trim().is_empty() {
            guard.push(line.to_string());
        }
    }

    trim_and_cap(&mut guard);
}

impl UiState {
    /// Mute is active when the button is pressed: recording pauses until it is released.
    pub fn set_muted(&self, muted: bool) {
        self.muted.store(muted, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn is_muted(&self) -> bool {
        self.muted.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// A client session is active: recording controls become usable.
    pub fn set_connected(&self, connected: bool) {
        self.connected.store(connected, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn is_connected(&self) -> bool {
        self.connected.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Append one note frame (kind 'n' or untyped text), separated from the previous message,
    /// keeping the field bounded.
    pub fn append_note(&self, text: &str) {
        append_note_message(&self.notes, text);
    }

    /// Append one dialogue fragment (kind 'd') to the dialogue field, keeping it bounded.
    pub fn append_dialogue(&self, text: &str) {
        append_dialogue_lines(&self.dialogue, text);
    }

    /// Snapshot of the notes log for display.
    pub fn snapshot_notes(&self) -> Vec<String> {
        self.notes.lock().expect("ui notes poisoned").clone()
    }

    /// Snapshot of the dialogue field for display.
    pub fn snapshot_dialogue(&self) -> Vec<String> {
        self.dialogue.lock().expect("ui dialogue poisoned").clone()
    }
}

/// egui window: a Mute/Record toggle button plus two side-by-side text fields (notes, dialogue).
pub struct MicroagentApp {
    ui: UiState,
}

impl MicroagentApp {
    pub fn new(ui: UiState) -> Self {
        Self { ui }
    }
}

/// One titled, vertically scrollable text column inside a `ui.columns` child.
/// `id_salt` must be unique per field: without it egui reuses one persistent ID for both
/// ScrollAreas and paints the red "Second use of ... ID" warning on the second one.
fn render_field(ui: &mut egui::Ui, title: &str, lines: &[String], placeholder: &str, id_salt: &str) {
    ui.heading(title);

    egui::ScrollArea::vertical()
        .id_salt(id_salt)
        .show(ui, |inner| {
            if lines.is_empty() {
                inner.weak(placeholder);
            }
            for line in lines {
                // Wrap long monospace lines so they stay inside the column width.
                inner.add(egui::Label::new(egui::RichText::new(line).monospace()).wrap());
            }
        });
}

impl eframe::App for MicroagentApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let muted = self.ui.is_muted();
        let connected = self.ui.is_connected();
        let button_label = if muted { "Record" } else { "Mute" };

        egui::CentralPanel::default().show(ui, |ui| {
            // Close button in the top-right corner.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                let close = egui::Button::new(
                    egui::RichText::new("✕").size(18.0).color(egui::Color32::from_rgb(90, 15, 15)),
                ).fill(egui::Color32::from_rgb(255, 140, 140));
                if ui.add(close).clicked() {
                    ui.ctx().send_viewport_cmd(egui::viewport::ViewportCommand::Close);
                }
            });

            // Connection status: the toggle is only useful while a client session is active.
            if connected {
                ui.colored_label(egui::Color32::from_rgb(80, 160, 80), "Teacher connection: active");
            } else {
                ui.colored_label(egui::Color32::from_rgb(170, 90, 90), "Teacher connection: inactive — waiting for a client");
            }

            // Toggle is disabled without a live session so recording cannot be toggled while idle.
            let fill = if muted {
                egui::Color32::from_rgb(80, 160, 80)
            } else {
                egui::Color32::from_rgb(170, 90, 90)
            };
            let button = egui::Button::new(
                egui::RichText::new(button_label).color(egui::Color32::WHITE),
            ).fill(fill).min_size([120.0, 40.0].into());
            if ui.add_enabled(connected, button).clicked() {
                self.ui.set_muted(!muted);
            }

            ui.separator();

            // Two equal columns side by side: notes on the left, dialogue on the right.
            // `columns` gives each child a bounded rect, so scroll areas fit the window width.
            ui.columns(2, |cols| {
                render_field(
                    &mut cols[0], "Notes", &self.ui.snapshot_notes(),
                    "Received notes will appear here…", "notes_scroll_area",
                );
                render_field(
                    &mut cols[1], "Dialogue", &self.ui.snapshot_dialogue(),
                    "The lesson dialogue will appear here…", "dialogue_scroll_area",
                );
            });
        });

        // Redraw periodically so mute state and new log lines stay fresh.
        ui.ctx().request_repaint_after(Duration::from_millis(250));
    }
}

#[cfg(test)]
mod tests {
    use super::{UiState, window_title_from_lesson, MAX_LINES, NOTE_SEPARATOR};

    #[test]
    fn title_strips_hashes_and_whitespace() {
        assert_eq!(window_title_from_lesson("### Lesson: City Transport\n"), "Lesson: City Transport");
        assert_eq!(window_title_from_lesson("#\n  # First Line\n"), "First Line");
        assert_eq!(window_title_from_lesson(""), "microagent");
    }

    #[test]
    fn mute_flag_roundtrips() {
        let ui = UiState::default();
        assert!(!ui.is_muted());
        ui.set_muted(true);
        assert!(ui.is_muted());
        ui.set_muted(false);
        assert!(!ui.is_muted());
    }

    #[test]
    fn connected_flag_roundtrips_and_shared_by_clone() {
        let a = UiState::default();
        let b = a.clone();
        assert!(!a.is_connected());
        b.set_connected(true);
        assert!(a.is_connected());
        b.set_connected(false);
        assert!(!a.is_connected());
    }

    #[test]
    fn note_and_dialogue_fields_are_separate() {
        let ui = UiState::default();
        ui.append_note("hello\nworld");
        ui.append_dialogue("you: hi\n\nteacher: hello");

        assert_eq!(ui.snapshot_notes(), vec!["hello", "world"]);
        assert_eq!(ui.snapshot_dialogue(), vec!["you: hi", "teacher: hello"]);
    }

    #[test]
    fn consecutive_notes_are_separated() {
        let ui = UiState::default();
        ui.append_note("first");
        ui.append_note("second");

        assert_eq!(ui.snapshot_notes(), vec!["first", NOTE_SEPARATOR, "second"]);
        // The very first message is not preceded by a separator.
        assert_ne!(ui.snapshot_notes().first().unwrap(), NOTE_SEPARATOR);
    }

    #[test]
    fn notes_drop_oldest_beyond_cap() {
        let ui = UiState::default();
        for i in 0..300u32 {
            ui.append_note(&format!("line {}", i));
        }

        // The field never grows past the cap.
        let notes = ui.snapshot_notes();
        assert!(notes.len() <= MAX_LINES);

        // The oldest messages were dropped; the newest one stays at the tail, preceded by its separator.
        assert_eq!(*notes.last().unwrap(), "line 299");
    }

    #[test]
    fn dialogue_drops_oldest_beyond_cap() {
        let ui = UiState::default();
        for i in 0..600u32 {
            ui.append_dialogue(&format!("you: {}", i));
        }
        assert_eq!(ui.snapshot_dialogue().len(), 500);
    }

    #[test]
    fn clone_shares_the_same_mute_flag_and_fields() {
        let a = UiState::default();
        let b = a.clone();
        b.set_muted(true);
        assert!(a.is_muted());
        b.append_note("note");
        b.append_dialogue("teacher: reply");
        assert_eq!(a.snapshot_notes(), vec!["note"]);
        assert_eq!(a.snapshot_dialogue(), vec!["teacher: reply"]);
    }
}
