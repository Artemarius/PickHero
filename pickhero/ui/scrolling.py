"""Scrolling note display for the playing screen.

Renders 6 string lanes with notes scrolling right-to-left, synchronized
to a playback clock. Optionally captures audio and shows hit/miss feedback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pygame

from pickhero.audio.midi_playback import BackingTrack, MidiPlayer
from pickhero.config import Config
from pickhero.matcher import NoteMatcher
from pickhero.tabs.timeline import NoteEvent, Timeline
from pickhero.ui.colors import (
    BG_COLOR,
    HIT_ZONE_COLOR,
    HUD_ACCENT_COLOR,
    HUD_TEXT_COLOR,
    LANE_BG_EVEN,
    LANE_BG_ODD,
    LANE_LINE_COLOR,
    LOOP_MARKER_COLOR,
    LOOP_MARKER_DISABLED_COLOR,
    LOOP_REGION_COLOR,
    LOOP_REGION_DISABLED_COLOR,
    NOTE_BORDER_COLOR,
    NOTE_TEXT_COLOR,
    STRING_COLORS,
    dimmed,
)
from pickhero.ui.feedback import FeedbackRenderer

# Layout constants
LANE_TOP_MARGIN = 80
LANE_BOTTOM_MARGIN = 40
MIN_NOTE_WIDTH_PX = 20
NOTE_HEIGHT_FRACTION = 0.7
NOTE_CORNER_RADIUS = 4

# Left margin for notes that already passed the hit zone (ms)
LEFT_MARGIN_MS = 2000
# Right margin for notes not yet visible (ms)
RIGHT_MARGIN_MS = 500


def _get_font(name: str, size: int) -> pygame.font.Font:
    """Try to load a system font with fallbacks."""
    for family in (name, "Courier New", "monospace"):
        font = pygame.font.SysFont(family, size)
        if font:
            return font
    return pygame.font.Font(None, size)


def format_time(ms: float) -> str:
    """Format milliseconds as M:SS."""
    total_seconds = max(0, int(ms / 1000))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


@dataclass
class _Layout:
    """Computed layout dimensions for current surface size."""

    screen_w: int
    screen_h: int
    lane_height: float
    note_h: float
    hit_zone_x: float
    usable_width: float
    pixels_per_ms: float
    visible_window_ms: float


class PlayingScreen:
    """Scrolling tab display with playback clock and optional audio matching."""

    def __init__(self, timeline: Timeline, visible_beats: int = 4,
                 hit_zone_fraction: float = 0.20, config: Config | None = None,
                 backing_track: BackingTrack | None = None):
        self._timeline = timeline
        self._visible_beats = visible_beats
        self._hit_zone_fraction = hit_zone_fraction
        self._config = config or Config()

        self._tempo_factor = max(0.5, min(1.0, self._config.tempo_factor))

        self._playback_ms: float = 0.0
        self._playing = False
        self._last_tick: float | None = None

        tempo = max(1, self._timeline.metadata.tempo)
        self._ms_per_beat = 60_000 / tempo
        self._visible_window_ms = self._visible_beats * self._ms_per_beat

        # Audio matching
        self._audio_capture = None  # AudioCapture, created on demand
        self._matcher: NoteMatcher | None = None
        self._feedback = FeedbackRenderer()
        self._audio_enabled = False
        self._noise_gate_db: float = self._config.audio.noise_gate_db

        # Loop state
        self._loop_start_ms: float | None = None
        self._loop_end_ms: float | None = None
        self._loop_enabled: bool = False

        # MIDI backing track
        self._midi_player: MidiPlayer | None = None
        self._backing_muted = not self._config.backing_track_enabled
        if backing_track is not None and len(backing_track) > 0:
            self._init_midi_player(backing_track)

    def toggle_play(self) -> None:
        """Toggle play/pause. Restarts if past the end."""
        if self._playback_ms >= self._timeline.duration_ms and not self._playing:
            self._playback_ms = 0.0
            if self._matcher:
                self._matcher.reset()
            self._feedback.reset()
        self._playing = not self._playing
        if self._playing:
            self._last_tick = time.perf_counter()
            if self._audio_enabled:
                self._start_audio()
            if self._midi_player is not None:
                self._midi_player.seek(self._playback_ms)
        else:
            self._last_tick = None
            self._stop_audio()
            if self._midi_player is not None:
                self._midi_player.pause()

    def seek(self, ms: float) -> None:
        """Seek to an absolute position in ms, clamped to [0, duration]."""
        self._playback_ms = max(0.0, min(ms, self._timeline.duration_ms))
        if self._matcher:
            self._matcher.reset()
        self._feedback.reset()
        if self._midi_player is not None:
            self._midi_player.seek(self._playback_ms)
        # Restart audio with new offset if active
        if self._audio_enabled and self._playing:
            self._stop_audio()
            self._start_audio()

    def is_playing(self) -> bool:
        return self._playing

    def set_tempo_factor(self, factor: float) -> None:
        """Set tempo scaling factor, clamped to [0.5, 1.0] and rounded to nearest 0.05."""
        factor = max(0.5, min(1.0, factor))
        factor = round(factor * 20) / 20  # round to nearest 0.05
        self._tempo_factor = factor
        self._config.tempo_factor = factor
        if self._matcher:
            self._matcher.reset()
        self._feedback.reset()

    def set_noise_gate_db(self, db: float) -> None:
        """Set noise gate threshold, clamped to [-80, -20] and rounded to int."""
        db = max(-80, min(-20, round(db)))
        self._noise_gate_db = db
        self._config.audio.noise_gate_db = db
        if self._audio_capture is not None:
            self._audio_capture.set_noise_gate_db(db)
        self._config.save()

    def update(self) -> None:
        """Advance playback clock by real elapsed time."""
        if not self._playing:
            return

        now = time.perf_counter()
        if self._last_tick is not None:
            elapsed_ms = (now - self._last_tick) * 1000.0 * self._tempo_factor
            self._playback_ms += elapsed_ms
        self._last_tick = now

        # Process audio matching
        if self._audio_enabled and self._audio_capture is not None and self._matcher is not None:
            detected = self._audio_capture.get_notes()
            for d in detected:
                d.timestamp_ms *= self._tempo_factor
            results = self._matcher.process_detected_notes(detected, self._playback_ms)
            self._feedback.add_results(results, self._playback_ms)
            self._feedback.cleanup(self._playback_ms)

        # Advance MIDI backing track
        if self._midi_player is not None:
            self._midi_player.update(self._playback_ms)

        # Loop check — jump back to start marker when reaching end marker
        if (self._loop_enabled and self._loop_end_ms is not None
                and self._loop_start_ms is not None
                and self._playback_ms >= self._loop_end_ms):
            if self._midi_player is not None:
                self._midi_player.pause()
            self._playback_ms = self._loop_start_ms
            self._last_tick = time.perf_counter()
            if self._matcher:
                self._matcher.reset()
            self._feedback.reset()
            if self._midi_player is not None:
                self._midi_player.seek(self._loop_start_ms)
            if self._audio_enabled and self._playing:
                self._stop_audio()
                self._start_audio()
            return

        if self._playback_ms >= self._timeline.duration_ms:
            self._playback_ms = self._timeline.duration_ms
            self._playing = False
            self._last_tick = None
            if self._midi_player is not None:
                self._midi_player.pause()
            self._stop_audio()

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Handle input. Returns 'menu' to go back, else None."""
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_SPACE:
            self.toggle_play()
        elif event.key == pygame.K_ESCAPE:
            self.stop_audio()
            return "menu"
        elif event.key == pygame.K_LEFT:
            self.seek(self._playback_ms - self._ms_per_beat)
        elif event.key == pygame.K_RIGHT:
            self.seek(self._playback_ms + self._ms_per_beat)
        elif event.key == pygame.K_HOME:
            self.seek(0)
        elif event.key == pygame.K_a:
            self._toggle_audio()
        elif event.key == pygame.K_PAGEDOWN:
            self.set_tempo_factor(self._tempo_factor - 0.05)
        elif event.key == pygame.K_PAGEUP:
            self.set_tempo_factor(self._tempo_factor + 0.05)
        elif event.key == pygame.K_i:
            self._set_loop_start(self._playback_ms)
        elif event.key == pygame.K_o:
            self._set_loop_end(self._playback_ms)
        elif event.key == pygame.K_p:
            self._toggle_loop()
        elif event.key == pygame.K_b:
            self._toggle_backing()
        elif event.key == pygame.K_LEFTBRACKET:
            self.set_noise_gate_db(self._noise_gate_db - 5)
        elif event.key == pygame.K_RIGHTBRACKET:
            self.set_noise_gate_db(self._noise_gate_db + 5)

        return None

    def render(self, surface: pygame.Surface) -> None:
        """Draw the full playing screen."""
        layout = self._layout(surface)

        surface.fill(BG_COLOR)
        self._draw_lanes(surface, layout)
        self._draw_loop_region(surface, layout)
        self._draw_hit_zone(surface, layout)
        self._draw_notes(surface, layout)
        self._draw_hud(surface, layout)

    # -- Pure math helpers (testable without display) --

    def _layout(self, surface: pygame.Surface) -> _Layout:
        """Compute layout from current surface dimensions."""
        w, h = surface.get_size()
        lane_area = h - LANE_TOP_MARGIN - LANE_BOTTOM_MARGIN
        lane_height = lane_area / 6
        note_h = lane_height * NOTE_HEIGHT_FRACTION
        hit_zone_x = w * self._hit_zone_fraction
        usable_width = w - hit_zone_x
        pixels_per_ms = usable_width / self._visible_window_ms if self._visible_window_ms > 0 else 1.0
        return _Layout(
            screen_w=w,
            screen_h=h,
            lane_height=lane_height,
            note_h=note_h,
            hit_zone_x=hit_zone_x,
            usable_width=usable_width,
            pixels_per_ms=pixels_per_ms,
            visible_window_ms=self._visible_window_ms,
        )

    @staticmethod
    def note_x(note_timestamp_ms: float, playback_ms: float,
               hit_zone_x: float, pixels_per_ms: float) -> float:
        """Calculate the x position of a note."""
        return hit_zone_x + (note_timestamp_ms - playback_ms) * pixels_per_ms

    @staticmethod
    def note_width(duration_ms: float, pixels_per_ms: float) -> float:
        """Calculate note rectangle width, enforcing minimum."""
        return max(duration_ms * pixels_per_ms, MIN_NOTE_WIDTH_PX)

    # -- Drawing --

    def _draw_lanes(self, surface: pygame.Surface, layout: _Layout) -> None:
        for i in range(6):
            y = LANE_TOP_MARGIN + i * layout.lane_height
            bg = LANE_BG_EVEN if i % 2 == 0 else LANE_BG_ODD
            pygame.draw.rect(
                surface, bg,
                (0, y, layout.screen_w, layout.lane_height),
            )
            # Divider line at bottom of lane
            line_y = int(y + layout.lane_height)
            pygame.draw.line(
                surface, LANE_LINE_COLOR,
                (0, line_y), (layout.screen_w, line_y),
            )

    def _draw_hit_zone(self, surface: pygame.Surface, layout: _Layout) -> None:
        x = int(layout.hit_zone_x)
        top = int(LANE_TOP_MARGIN)
        bottom = int(LANE_TOP_MARGIN + 6 * layout.lane_height)
        pygame.draw.line(surface, HIT_ZONE_COLOR, (x, top), (x, bottom), 2)

    def _draw_notes(self, surface: pygame.Surface, layout: _Layout) -> None:
        # Visible time range with margins for long notes
        view_start = self._playback_ms - LEFT_MARGIN_MS
        view_end = self._playback_ms + self._visible_window_ms + RIGHT_MARGIN_MS

        notes = self._timeline.get_notes_in_range(view_start, view_end)

        fret_font = _get_font("consolas", 14)

        for note in notes:
            x = self.note_x(
                note.timestamp_ms, self._playback_ms,
                layout.hit_zone_x, layout.pixels_per_ms,
            )
            w = self.note_width(note.duration_ms, layout.pixels_per_ms)

            # Skip notes fully off-screen
            if x + w < 0 or x > layout.screen_w:
                continue

            # Y position: string 1-6
            lane_y = LANE_TOP_MARGIN + (note.string - 1) * layout.lane_height
            y = lane_y + layout.lane_height / 2 - layout.note_h / 2

            # Color: feedback color if matched, dimmed if past the hit zone
            base_color = STRING_COLORS.get(note.string, (180, 180, 180))
            past_hit_zone = note.timestamp_ms < self._playback_ms
            if self._audio_enabled:
                color = self._feedback.get_note_color(
                    note, base_color, self._playback_ms, past_hit_zone,
                )
            else:
                color = dimmed(base_color) if past_hit_zone else base_color

            rect = pygame.Rect(int(x), int(y), int(w), int(layout.note_h))
            pygame.draw.rect(surface, color, rect, border_radius=NOTE_CORNER_RADIUS)
            pygame.draw.rect(surface, NOTE_BORDER_COLOR, rect, width=2, border_radius=NOTE_CORNER_RADIUS)

            # Fret number — only draw if note is wide enough to fit it
            fret_text = fret_font.render(str(note.fret), True, NOTE_TEXT_COLOR)
            if fret_text.get_width() + 4 <= rect.width:
                tx = rect.x + rect.width // 2 - fret_text.get_width() // 2
                ty = rect.y + rect.height // 2 - fret_text.get_height() // 2
                surface.blit(fret_text, (tx, ty))

    def _draw_hud(self, surface: pygame.Surface, layout: _Layout) -> None:
        title_font = _get_font("arial", 20)
        time_font = _get_font("consolas", 20)
        hint_font = _get_font("arial", 14)

        meta = self._timeline.metadata
        w = layout.screen_w

        # Top-left: title + artist
        title = meta.title or "Untitled"
        if meta.artist:
            title = f"{meta.artist} — {title}"
        title_surf = title_font.render(title, True, HUD_TEXT_COLOR)
        surface.blit(title_surf, (12, 12))

        # Top-center: BPM with tempo percentage (and streak below it)
        pct = int(self._tempo_factor * 100)
        bpm_text = f"{meta.tempo} BPM ({pct}%)"
        bpm_surf = title_font.render(bpm_text, True, HUD_ACCENT_COLOR)
        surface.blit(bpm_surf, (w // 2 - bpm_surf.get_width() // 2, 12))

        # Loop status below BPM
        loop_y = 36
        loop_info = self._loop_hud_text()
        if loop_info:
            loop_color = HUD_ACCENT_COLOR if self._loop_enabled else HUD_TEXT_COLOR
            loop_surf = hint_font.render(loop_info, True, loop_color)
            surface.blit(loop_surf, (w // 2 - loop_surf.get_width() // 2, loop_y))
            loop_y += 18

        if self._audio_enabled:
            self._feedback.draw_streak(surface, title_font, w // 2, loop_y)

        # Top-right: time
        current = format_time(self._playback_ms)
        total = format_time(self._timeline.duration_ms)
        time_text = f"{current} / {total}"
        time_surf = time_font.render(time_text, True, HUD_TEXT_COLOR)
        surface.blit(time_surf, (w - time_surf.get_width() - 12, 12))

        # Top-right second line: accuracy stats
        stats_bottom_y = 36
        if self._audio_enabled and self._matcher is not None:
            stats = self._matcher.get_statistics()
            if stats["total"] > 0:
                self._feedback.draw_stats(surface, stats, hint_font, w - 12, 36)
                stats_bottom_y = 54

        # Top-right: noise gate (below stats, only when audio enabled)
        if self._audio_enabled:
            gate_text = f"Gate: {int(self._noise_gate_db)} dB"
            gate_surf = hint_font.render(gate_text, True, HUD_ACCENT_COLOR)
            surface.blit(gate_surf, (w - gate_surf.get_width() - 12, stats_bottom_y))

        # Bottom-center: play state + controls
        state = "Playing" if self._playing else "Paused"
        audio_state = "ON" if self._audio_enabled else "off"
        loop_state = "ON" if self._loop_enabled else "off"
        backing_state = ""
        if self._midi_player is not None:
            backing_state = f"|  B: backing {'off' if self._backing_muted else 'ON'}  "
        hint = (
            f"{state}  |  SPACE: play/pause  |  LEFT/RIGHT: seek  "
            f"|  HOME: restart  |  PgDn/PgUp: tempo  |  [/]: gate  "
            f"|  A: audio {audio_state}  "
            f"{backing_state}"
            f"|  I/O: loop {loop_state}  |  P: toggle  |  ESC: menu"
        )
        hint_surf = hint_font.render(hint, True, HUD_TEXT_COLOR)
        y = layout.screen_h - LANE_BOTTOM_MARGIN + 8
        surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, y))

        # Top-left second line: track name
        if meta.track_name:
            track_surf = hint_font.render(
                f"Track: {meta.track_name}", True, HUD_TEXT_COLOR
            )
            surface.blit(track_surf, (12, 38))

    # -- Loop control --

    def _set_loop_start(self, ms: float) -> None:
        """Set loop start marker. Auto-swap if after end, auto-enable when both set."""
        self._loop_start_ms = ms
        if self._loop_end_ms is not None and self._loop_start_ms > self._loop_end_ms:
            self._loop_start_ms, self._loop_end_ms = self._loop_end_ms, self._loop_start_ms
        self._enforce_min_loop()
        if self._loop_start_ms is not None and self._loop_end_ms is not None:
            self._loop_enabled = True

    def _set_loop_end(self, ms: float) -> None:
        """Set loop end marker. Auto-swap if before start, auto-enable when both set."""
        self._loop_end_ms = ms
        if self._loop_start_ms is not None and self._loop_end_ms < self._loop_start_ms:
            self._loop_start_ms, self._loop_end_ms = self._loop_end_ms, self._loop_start_ms
        self._enforce_min_loop()
        if self._loop_start_ms is not None and self._loop_end_ms is not None:
            self._loop_enabled = True

    def _enforce_min_loop(self) -> None:
        """Ensure loop region is at least one beat long."""
        if self._loop_start_ms is not None and self._loop_end_ms is not None:
            if self._loop_end_ms - self._loop_start_ms < self._ms_per_beat:
                self._loop_end_ms = self._loop_start_ms + self._ms_per_beat

    def _toggle_loop(self) -> None:
        """Toggle loop off (keep markers), then clear markers on second press."""
        if self._loop_enabled:
            self._loop_enabled = False
        elif self._loop_start_ms is not None or self._loop_end_ms is not None:
            self._loop_start_ms = None
            self._loop_end_ms = None
            self._loop_enabled = False
        # If everything is already None/False, do nothing

    def _loop_hud_text(self) -> str | None:
        """Return loop status text for HUD, or None if no markers."""
        if self._loop_start_ms is not None and self._loop_end_ms is not None:
            s = format_time(self._loop_start_ms)
            e = format_time(self._loop_end_ms)
            if self._loop_enabled:
                return f"LOOP {s} - {e}"
            return f"loop {s} - {e} (off)"
        if self._loop_start_ms is not None:
            return f"loop start: {format_time(self._loop_start_ms)}"
        if self._loop_end_ms is not None:
            return f"loop end: {format_time(self._loop_end_ms)}"
        return None

    def _draw_loop_region(self, surface: pygame.Surface, layout: _Layout) -> None:
        """Draw loop markers and shaded region between them."""
        if self._loop_start_ms is None and self._loop_end_ms is None:
            return

        lane_top = int(LANE_TOP_MARGIN)
        lane_bottom = int(LANE_TOP_MARGIN + 6 * layout.lane_height)
        lane_h = lane_bottom - lane_top

        marker_color = LOOP_MARKER_COLOR if self._loop_enabled else LOOP_MARKER_DISABLED_COLOR
        region_color = LOOP_REGION_COLOR if self._loop_enabled else LOOP_REGION_DISABLED_COLOR

        # Draw shaded region between both markers
        if self._loop_start_ms is not None and self._loop_end_ms is not None:
            x_start = int(self.note_x(self._loop_start_ms, self._playback_ms,
                                      layout.hit_zone_x, layout.pixels_per_ms))
            x_end = int(self.note_x(self._loop_end_ms, self._playback_ms,
                                    layout.hit_zone_x, layout.pixels_per_ms))
            # Clamp to screen
            x_start = max(0, min(x_start, layout.screen_w))
            x_end = max(0, min(x_end, layout.screen_w))
            if x_end > x_start:
                overlay = pygame.Surface((x_end - x_start, lane_h), pygame.SRCALPHA)
                overlay.fill(region_color)
                surface.blit(overlay, (x_start, lane_top))

        # Draw start marker
        if self._loop_start_ms is not None:
            x = int(self.note_x(self._loop_start_ms, self._playback_ms,
                                layout.hit_zone_x, layout.pixels_per_ms))
            if 0 <= x <= layout.screen_w:
                pygame.draw.line(surface, marker_color, (x, lane_top), (x, lane_bottom), 2)
                # Right-pointing triangle at top
                pygame.draw.polygon(surface, marker_color, [
                    (x, lane_top), (x + 10, lane_top + 7), (x, lane_top + 14),
                ])

        # Draw end marker
        if self._loop_end_ms is not None:
            x = int(self.note_x(self._loop_end_ms, self._playback_ms,
                                layout.hit_zone_x, layout.pixels_per_ms))
            if 0 <= x <= layout.screen_w:
                pygame.draw.line(surface, marker_color, (x, lane_top), (x, lane_bottom), 2)
                # Left-pointing triangle at top
                pygame.draw.polygon(surface, marker_color, [
                    (x, lane_top), (x - 10, lane_top + 7), (x, lane_top + 14),
                ])

    # -- Audio control --

    def _toggle_audio(self) -> None:
        """Toggle audio capture on/off."""
        self._audio_enabled = not self._audio_enabled
        if self._audio_enabled and self._playing:
            self._start_audio()
        else:
            self._stop_audio()

    def _start_audio(self) -> None:
        """Start audio capture and create matcher."""
        try:
            from pickhero.audio.input import AudioCapture
            if self._audio_capture is None:
                self._audio_capture = AudioCapture(self._config)
            self._audio_capture.start()
            self._matcher = NoteMatcher(
                self._timeline,
                timing_window_ms=self._config.timing_window_ms,
                audio_offset_ms=self._playback_ms + self._config.audio_latency_offset_ms,
                chord_threshold_ms=self._config.chord_threshold_ms,
            )
            self._feedback.reset()
        except Exception as e:
            print(f"Audio start failed: {e}")
            self._audio_enabled = False

    def _stop_audio(self) -> None:
        """Stop audio capture."""
        if self._audio_capture is not None:
            self._audio_capture.stop()

    def stop_audio(self) -> None:
        """Public method to stop audio (called on state transitions)."""
        self._stop_audio()
        self._audio_enabled = False
        if self._midi_player is not None:
            self._midi_player.close()
            self._midi_player = None

    # -- MIDI backing track --

    def _init_midi_player(self, backing_track: BackingTrack) -> None:
        """Create and open MidiPlayer. Silently continues if MIDI unavailable."""
        try:
            player = MidiPlayer(backing_track)
            if player.open():
                player.set_muted(self._backing_muted)
                self._midi_player = player
            else:
                player.close()
        except Exception as e:
            print(f"MIDI player init failed: {e}")

    def _toggle_backing(self) -> None:
        """Toggle backing track mute on/off."""
        if self._midi_player is None:
            return
        self._backing_muted = not self._backing_muted
        self._midi_player.set_muted(self._backing_muted)
