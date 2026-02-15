"""Scrolling note display for the playing screen.

Renders 6 string lanes with notes scrolling right-to-left, synchronized
to a playback clock. No audio matching — visual display only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pygame

from pickhero.tabs.timeline import NoteEvent, Timeline
from pickhero.ui.colors import (
    BG_COLOR,
    HIT_ZONE_COLOR,
    HUD_ACCENT_COLOR,
    HUD_TEXT_COLOR,
    LANE_BG_EVEN,
    LANE_BG_ODD,
    LANE_LINE_COLOR,
    NOTE_TEXT_COLOR,
    STRING_COLORS,
    dimmed,
)

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
    """Scrolling tab display with playback clock."""

    def __init__(self, timeline: Timeline, visible_beats: int = 4,
                 hit_zone_fraction: float = 0.20):
        self._timeline = timeline
        self._visible_beats = visible_beats
        self._hit_zone_fraction = hit_zone_fraction

        self._playback_ms: float = 0.0
        self._playing = False
        self._last_tick: float | None = None

        tempo = max(1, self._timeline.metadata.tempo)
        self._ms_per_beat = 60_000 / tempo
        self._visible_window_ms = self._visible_beats * self._ms_per_beat

    def toggle_play(self) -> None:
        """Toggle play/pause. Restarts if past the end."""
        if self._playback_ms >= self._timeline.duration_ms and not self._playing:
            self._playback_ms = 0.0
        self._playing = not self._playing
        if self._playing:
            self._last_tick = time.perf_counter()
        else:
            self._last_tick = None

    def seek(self, ms: float) -> None:
        """Seek to an absolute position in ms, clamped to [0, duration]."""
        self._playback_ms = max(0.0, min(ms, self._timeline.duration_ms))

    def is_playing(self) -> bool:
        return self._playing

    def update(self) -> None:
        """Advance playback clock by real elapsed time."""
        if not self._playing:
            return

        now = time.perf_counter()
        if self._last_tick is not None:
            elapsed_ms = (now - self._last_tick) * 1000.0
            self._playback_ms += elapsed_ms
        self._last_tick = now

        if self._playback_ms >= self._timeline.duration_ms:
            self._playback_ms = self._timeline.duration_ms
            self._playing = False
            self._last_tick = None

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Handle input. Returns 'menu' to go back, else None."""
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_SPACE:
            self.toggle_play()
        elif event.key == pygame.K_ESCAPE:
            return "menu"
        elif event.key == pygame.K_LEFT:
            self.seek(self._playback_ms - self._ms_per_beat)
        elif event.key == pygame.K_RIGHT:
            self.seek(self._playback_ms + self._ms_per_beat)
        elif event.key == pygame.K_HOME:
            self.seek(0)

        return None

    def render(self, surface: pygame.Surface) -> None:
        """Draw the full playing screen."""
        layout = self._layout(surface)

        surface.fill(BG_COLOR)
        self._draw_lanes(surface, layout)
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

        fret_font = _get_font("consolas", max(12, int(layout.note_h * 0.6)))

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

            # Color: dimmed if past the hit zone
            base_color = STRING_COLORS.get(note.string, (180, 180, 180))
            past_hit_zone = note.timestamp_ms < self._playback_ms
            color = dimmed(base_color) if past_hit_zone else base_color

            rect = pygame.Rect(int(x), int(y), int(w), int(layout.note_h))
            pygame.draw.rect(surface, color, rect, border_radius=NOTE_CORNER_RADIUS)

            # Fret number
            fret_text = fret_font.render(str(note.fret), True, NOTE_TEXT_COLOR)
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

        # Top-center: BPM
        bpm_text = f"{meta.tempo} BPM"
        bpm_surf = title_font.render(bpm_text, True, HUD_ACCENT_COLOR)
        surface.blit(bpm_surf, (w // 2 - bpm_surf.get_width() // 2, 12))

        # Top-right: time
        current = format_time(self._playback_ms)
        total = format_time(self._timeline.duration_ms)
        time_text = f"{current} / {total}"
        time_surf = time_font.render(time_text, True, HUD_TEXT_COLOR)
        surface.blit(time_surf, (w - time_surf.get_width() - 12, 12))

        # Bottom-center: play state + controls
        state = "Playing" if self._playing else "Paused"
        hint = f"{state}  |  SPACE: play/pause  |  LEFT/RIGHT: seek  |  HOME: restart  |  ESC: menu"
        hint_surf = hint_font.render(hint, True, HUD_TEXT_COLOR)
        y = layout.screen_h - LANE_BOTTOM_MARGIN + 8
        surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, y))

        # Top-left second line: track name
        if meta.track_name:
            track_surf = hint_font.render(
                f"Track: {meta.track_name}", True, HUD_TEXT_COLOR
            )
            surface.blit(track_surf, (12, 38))
