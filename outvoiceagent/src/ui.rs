use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};

/// Number of recognized text groups (by group_id) kept in memory.
pub const MAX_GROUPS: usize = 50;
/// Steady auto-scroll speed for the subtitle effect (px/s).
const SCROLL_SPEED: f32 = 28.75; // ~15% faster than the original 25 px/s
/// Faster manual scroll speed while an arrow key is held (px/s).
const MANUAL_SCROLL_SPEED: f32 = 120.0;
/// Larger, light mint-green text for readability over any background.
const TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(145, 237, 209);
/// Central panel margins. The top grip panel already sits above this panel,
/// so the top margin only needs a small gap below it (not a full line of room).
const MARGIN: egui::Margin = egui::Margin { left: 8, right: 8, top: 8, bottom: 8 };

/// One recognized group: stable id from the transport counter + its latest full text.
#[derive(Clone, Debug)]
pub struct GroupText {
    pub id: u32,
    pub text: String,
}

/// Shared state between the server session thread and the egui overlay window.
#[derive(Clone, Default)]
pub struct UiState {
    groups: Arc<Mutex<Vec<GroupText>>>,
    /// egui context captured by the overlay so the server thread can wake the UI
    /// immediately when a new recognition arrives (no fixed repaint timer needed).
    repaint: Arc<Mutex<Option<egui::Context>>>,
}

impl UiState {
    /// Store the overlay's egui context so other threads can request an immediate repaint.
    pub fn set_repaint_ctx(&self, ctx: egui::Context) {
        *self.repaint.lock().expect("ui repaint ctx poisoned") = Some(ctx);
    }

    /// Wake the overlay right away (no-op until the context has been captured).
    fn request_repaint(&self) {
        if let Some(ctx) = self.repaint.lock().expect("ui repaint ctx poisoned").as_ref() {
            ctx.request_repaint();
        }
    }

    /// Store or refresh a group's full recognized text. An empty incoming text is an explicit
    /// "clear" signal; otherwise a known group is updated only when the new text is longer, so a
    /// delayed shorter/stale result cannot clobber a fuller one. Returns true on a newly
    /// inserted group, false otherwise.
    pub fn set_group_text(&self, id: u32, text: &str) -> bool {
        let mut guard = self.groups.lock().expect("ui groups poisoned");
        let (is_new, changed) = match guard.iter_mut().find(|g| g.id == id) {
            Some(g) => {
                // Empty clears; a non-empty update wins only when strictly longer (char count, not bytes).
                let changed = if text.is_empty() {
                    !g.text.is_empty()
                } else {
                    text.chars().count() > g.text.chars().count()
                };
                if changed {
                    g.text = text.to_string();
                }
                (false, changed)
            },
            None => {
                guard.push(GroupText { id, text: text.to_string() });
                let drop_n = guard.len().saturating_sub(MAX_GROUPS);
                if drop_n > 0 {
                    guard.drain(0..drop_n);
                }
                (true, true)
            },
        };
        drop(guard);
        // Refresh the overlay as soon as the text changes so it tracks recognition closely.
        if changed {
            self.request_repaint();
        }
        is_new
    }

    /// Drop all stored groups and wake the overlay so a fresh session starts with an empty display.
    pub fn clear_groups(&self) {
        let mut guard = self.groups.lock().expect("ui groups poisoned");
        if !guard.is_empty() {
            guard.clear();
            drop(guard);
            self.request_repaint();
        }
    }

    /// Combined display string: non-empty groups joined by spaces in insertion order (test helper).
    #[cfg(test)]
    pub fn combined_text(&self) -> String {
        let guard = self.groups.lock().expect("ui groups poisoned");
        let texts: Vec<&str> = guard.iter().map(|g| g.text.as_str()).filter(|t| !t.is_empty()).collect();
        texts.join(" ")
    }

    /// Copy of all non-empty groups in insertion order (for rendering only).
    pub fn groups_snapshot(&self) -> Vec<GroupText> {
        let guard = self.groups.lock().expect("ui groups poisoned");
        guard.iter().filter(|g| !g.text.is_empty()).cloned().collect()
    }

    /// Number of groups currently in the queue (test helper).
    #[cfg(test)]
    pub fn group_count(&self) -> usize {
        self.groups.lock().expect("ui groups poisoned").len()
    }
}

/// Transparent, always-on-top overlay pinned to the bottom of the screen.
pub struct OverlayApp {
    ui: UiState,
    /// True once size/position were applied from the real monitor size.
    placed: bool,
    /// Set by the server on SIGINT/SIGTERM; when true the window closes for a clean exit.
    shutdown: Arc<AtomicBool>,
    /// Current subtitle scroll position (px); advanced each frame so text drifts up like credits.
    scroll_offset: f32,
    /// Timestamp of the previous frame, used to compute dt for smooth scrolling.
    last_update: Instant,
    /// Measured max scroll range from the previous frame (content height minus viewport height,
    /// clamped >= 0). `scroll_offset` is bounded by this so it never scrolls past the content into
    /// blank space — egui only clamps its own state at end-of-frame, but we re-force `.scroll_offset()`
    /// every frame and would otherwise overwrite that clamp.
    max_scroll: f32,
}

impl OverlayApp {
    pub fn new(ui: UiState, shutdown: Arc<AtomicBool>) -> Self {
        Self { ui, placed: false, shutdown, scroll_offset: 0.0, last_update: Instant::now(), max_scroll: 0.0 }
    }
}

impl eframe::App for OverlayApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Graceful shutdown requested (SIGINT/SIGTERM): close the window so the process exits cleanly.
        if self.shutdown.load(Ordering::Relaxed) {
            ui.ctx().send_viewport_cmd(egui::viewport::ViewportCommand::Close);
            return;
        }

        let ctx = ui.ctx().clone();
        // Share the context so the server thread can repaint the overlay the moment
        // a new recognition arrives instead of waiting for the periodic timer.
        self.ui.set_repaint_ctx(ctx.clone());

        // Apply the real size/position exactly once; afterwards the user's drags win.
        if !self.placed {
            if let Some(monitor) = ctx.input(|i| i.viewport().monitor_size) {
                if (monitor.x > 0.0) && (monitor.y > 0.0) {
                    self.placed = true;
                    let w = (monitor.x * 0.8).max(320.0);
                    // Tall enough for the grip panel plus 3 wrapped lines of 30 px text.
                    let h = (monitor.y * 0.16).min(monitor.y - 56.0).max(40.0);
                    ctx.send_viewport_cmd(egui::viewport::ViewportCommand::InnerSize(egui::Vec2::new(w, h)));
                    // OuterPosition is in monitor space; window height == inner height (no decorations).
                    let pos = egui::Pos2::new(monitor.x * 0.1, monitor.y - h - 40.0);
                    ctx.send_viewport_cmd(egui::viewport::ViewportCommand::OuterPosition(pos));
                }
            }
        }

        // Subtitle-style scrolling: advance the offset by dt each frame so new text appears at the
        // bottom and old text drifts up, like movie credits. Arrow keys override for manual control.
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        self.last_update = now;

        if ctx.input(|i| i.key_down(egui::Key::ArrowUp)) {
            // Rewind: scroll back up to re-read missed text.
            self.scroll_offset -= MANUAL_SCROLL_SPEED * dt;
            if self.scroll_offset < 0.0 { self.scroll_offset = 0.0; }
        } else if ctx.input(|i| i.key_down(egui::Key::ArrowDown)) {
            // Fast-forward: scroll ahead faster than the auto speed.
            self.scroll_offset += MANUAL_SCROLL_SPEED * dt;
        } else {
            // No keys held: steady auto-scroll so subtitles keep flowing.
            self.scroll_offset += SCROLL_SPEED * dt;
        }
        // Bound to the measured content range so we never scroll past the last line into blank space.
        self.scroll_offset = self.scroll_offset.clamp(0.0, self.max_scroll);

        let groups = self.ui.groups_snapshot();
        let scroll = self.scroll_offset;

        // Top-right controls for the undecorated overlay: drag handle + close button.
        // Native drag: the window manager moves the window (X11 _NET_WM_MOVERESIZE /
        // Wayland xdg-toplevel move), so it stays smooth and needs no position tracking.
        egui::Panel::top("overlay_grip").frame(
            egui::Frame::new().fill(egui::Color32::TRANSPARENT).inner_margin(4)
        ).show(ui, |ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                // Close button (far right): closes the overlay window and exits the process.
                let close = egui::Button::new(
                    egui::RichText::new("✕").size(20.0).color(egui::Color32::from_rgb(90, 15, 15)),
                ).fill(egui::Color32::from_rgb(255, 140, 140));
                if ui.add(close).clicked() {
                    ui.ctx().send_viewport_cmd(egui::viewport::ViewportCommand::Close);
                }
                // Drag handle (left of the close button): hold it to move the window.
                let grip = egui::Button::new(
                    egui::RichText::new("⠿").size(20.0).color(egui::Color32::from_rgb(15, 60, 47)),
                ).fill(TEXT_COLOR);
                let response = ui.add(grip);
                // True from the press frame on (a plain Button never reports `dragged`).
                if response.is_pointer_button_down_on() {
                    ui.ctx().send_viewport_cmd(egui::viewport::ViewportCommand::StartDrag);
                }
            });
        });

        egui::CentralPanel::default().frame(
            egui::Frame::new().fill(egui::Color32::TRANSPARENT).inner_margin(MARGIN)
        ).show(ui, |ui| {
            // Subtitle-style scrolling: each group is one line; the offset moves the "camera" down.
            let output = egui::ScrollArea::vertical()
                .id_salt("subtitle_scroll")
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                .scroll_offset(egui::Vec2::new(0.0, scroll))
                .show(ui, |ui| {
                    for g in &groups {
                        let label = egui::Label::new(
                            egui::RichText::new(g.text.clone()).size(30.0).color(TEXT_COLOR)
                                .background_color(egui::Color32::from_black_alpha(140)),
                        ).wrap();
                        ui.add(label);
                    }
                });
            // Remember the real scroll range so next frame's offset is clamped to it (no blank space).
            self.max_scroll = (output.content_size.y - output.inner_rect.height()).max(0.0);
        });

        // Refresh a few times a second so newly recognized groups appear promptly.
        ui.ctx().request_repaint_after(Duration::from_millis(250));
    }
}

#[cfg(test)]
mod tests {
    use super::{UiState, MAX_GROUPS};

    #[test]
    fn same_group_updates_in_place() {
        let ui = UiState::default();
        assert!(ui.set_group_text(1, "first")); // new group
        assert!(!ui.set_group_text(1, "updated")); // refresh, not a new one
        assert_eq!(ui.combined_text(), "updated");
    }

    #[test]
    fn clear_groups_resets_state() {
        let ui = UiState::default();
        ui.set_group_text(1, "one");
        ui.set_group_text(2, "two");
        assert_eq!(ui.group_count(), 2);
        // A fresh session wipes the previous one's text.
        ui.clear_groups();
        assert_eq!(ui.group_count(), 0);
        assert_eq!(ui.combined_text(), "");
    }

    #[test]
    fn groups_are_kept_in_order_and_combined() {
        let ui = UiState::default();
        ui.set_group_text(2, "two");
        ui.set_group_text(1, "one");
        assert_eq!(ui.combined_text(), "two one");
        // The oldest (earlier group) is refreshed in place.
        ui.set_group_text(1, "one v2");
        assert_eq!(ui.combined_text(), "two one v2");
    }

    #[test]
    fn empty_groups_are_skipped_in_display() {
        let ui = UiState::default();
        ui.set_group_text(1, "one");
        ui.set_group_text(2, ""); // cleared group waiting for recognition
        ui.set_group_text(3, "three");
        assert_eq!(ui.combined_text(), "one three");
    }

    #[test]
    fn queue_caps_at_max_groups() {
        let ui = UiState::default();
        for i in 0..(MAX_GROUPS as u32 + 10) {
            ui.set_group_text(i, &format!("g{i}"));
        }
        assert_eq!(ui.group_count(), MAX_GROUPS);
    }

    #[test]
    fn groups_snapshot_returns_non_empty_in_order() {
        let ui = UiState::default();
        assert!(ui.groups_snapshot().is_empty()); // empty initially
        ui.set_group_text(1, "one");
        ui.set_group_text(2, ""); // cleared group waiting for recognition
        ui.set_group_text(3, "three");
        let snap = ui.groups_snapshot();
        assert_eq!(snap.len(), 2); // the empty group is skipped
        assert_eq!(snap[0].id, 1);
        assert_eq!(snap[0].text, "one");
        assert_eq!(snap[1].id, 3);
        assert_eq!(snap[1].text, "three");
    }

    #[test]
    fn refresh_keeps_queue_order_by_insertion() {
        let ui = UiState::default();
        ui.set_group_text(1, "one");
        ui.set_group_text(2, "two");
        // Refresh the first group again — it must stay before the second.
        ui.set_group_text(1, "one v2");
        assert_eq!(ui.combined_text(), "one v2 two");
    }

    #[test]
    fn longer_update_wins_over_shorter() {
        // A delayed shorter/stale result must not clobber a fuller one.
        let ui = UiState::default();
        assert!(ui.set_group_text(1, "short"));
        assert!(!ui.set_group_text(1, "longer text")); // longer -> updates in place
        assert_eq!(ui.combined_text(), "longer text");

        let ui2 = UiState::default();
        assert!(ui2.set_group_text(1, "abcdef"));
        assert!(!ui2.set_group_text(1, "ab")); // shorter -> ignored
        assert_eq!(ui2.combined_text(), "abcdef");
    }
}
