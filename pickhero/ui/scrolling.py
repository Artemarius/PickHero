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
from pickhero.progress import ProgressTracker
from pickhero.tabs.timeline import NoteEvent, Timeline
from pickhero.audio.note_utils import freq_to_cents_deviation, midi_to_name
from pickhero.ui.colors import (
    STRING_COLORS,
    cycle_theme,
    dimmed,
    get_theme,
)
from pickhero.ui.feedback import FeedbackRenderer, TimingOverlay
from pickhero.timing import TimingVerdict


class _FontCache:
    """Cache rendered font surfaces to avoid repeated font.render() calls."""
    def __init__(self):
        self._cache: dict[tuple, pygame.Surface] = {}

    def render(self, font: pygame.font.Font, text: str, antialias: bool, color: tuple) -> pygame.Surface:
        key = (id(font), text, antialias, color)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        surf = font.render(text, antialias, color)
        surf = surf.convert_alpha()
        self._cache[key] = surf
        return surf


_font_cache = _FontCache()

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

# Difficulty filter: fret limit cycle values
FRET_LIMITS = [24, 12, 7, 5, 3]


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
                 backing_track: BackingTrack | None = None,
                 progress_tracker: ProgressTracker | None = None,
                 song_key: str = ""):
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
        self._visible_window_ms = 8000.0  # Fixed 8-second window

        # Count-in state
        count_in_beats = max(0, self._config.count_in_beats)
        self._count_in_ms = count_in_beats * self._ms_per_beat
        self._last_count_in_beat: int = -1

        # Audio matching
        self._audio_capture = None  # AudioCapture, created on demand
        self._matcher: NoteMatcher | None = None
        self._feedback = FeedbackRenderer()
        self._audio_enabled = True
        self._noise_gate_db: float = self._config.audio.noise_gate_db

        # Loop state
        self._loop_start_ms: float | None = None
        self._loop_end_ms: float | None = None
        self._loop_enabled: bool = False

        # Progress tracking
        self._progress_tracker = progress_tracker
        self._song_key = song_key
        self._song_completed = False
        self._is_new_best = False
        self._recommendations: list[str] = []

        # MIDI backing track
        self._midi_player: MidiPlayer | None = None
        self._backing_muted = not self._config.backing_track_enabled
        if backing_track is not None and len(backing_track) > 0:
            self._init_midi_player(backing_track)

        # Difficulty filter
        self._max_fret: int = self._config.max_fret
        self._active_strings: list[bool] = list(self._config.active_strings)

        # Signal level meter
        self._signal_db: float = -120.0
        self._signal_db_smooth: float = -120.0

        # Tuner display
        self._tuner_freq: float = 0.0
        self._tuner_confidence: float = 0.0
        self._tuner_freq_smooth: float = 0.0
        self._tuner_displayed_note: int = -1
        self._tuner_note_stable_frames: int = 0

        # Chord partial credit mode
        self._chord_partial_credit: bool = self._config.chord_partial_credit

        # Help overlay
        self._show_help: bool = False

        # Wait mode
        self._wait_mode: bool = self._config.wait_mode
        self._wait_mode_frozen: bool = False

        # Timing Judge mode
        self._timing_judge: bool = self._config.timing_judge_mode
        self._pitch_strict: bool = self._config.pitch_strict_mode
        self._last_obs_count: int = 0
        self._timing_overlay: TimingOverlay | None = None
        self._timing_summary = None
        self._timing_worst_measures: list[tuple[int, float, float]] = []

    def _note_passes_filter(self, note: NoteEvent) -> bool:
        """Check if a note passes the difficulty filter."""
        if note.fret > self._max_fret:
            return False
        if not self._active_strings[note.string - 1]:
            return False
        return True

    def _is_filter_active(self) -> bool:
        """Check if any difficulty filter is active."""
        return self._max_fret < 24 or not all(self._active_strings)

    def toggle_play(self) -> None:
        """Toggle play/pause. Restarts with count-in if at beginning or past end."""
        if self._playback_ms >= self._timeline.duration_ms and not self._playing:
            # Restart from beginning with count-in
            self._playback_ms = -self._count_in_ms if self._count_in_ms > 0 else 0.0
            self._last_count_in_beat = -1
            self._song_completed = False
            self._is_new_best = False
            self._weakest_sections = []
            self._recommendations = []
            self._timing_summary = None
            self._timing_worst_measures = []
            self._last_obs_count = 0
            if self._matcher:
                self._matcher.reset()
            self._feedback.reset()
            if self._timing_overlay:
                self._timing_overlay.reset()
        elif self._playback_ms == 0.0 and not self._playing and self._count_in_ms > 0:
            # Starting from the very beginning — add count-in
            self._playback_ms = -self._count_in_ms
            self._last_count_in_beat = -1
            self._song_completed = False
            self._is_new_best = False
            self._weakest_sections = []
            self._recommendations = []
            self._timing_summary = None
            self._timing_worst_measures = []
            self._last_obs_count = 0
        elif self._playback_ms == 0.0 and not self._playing and self._count_in_ms > 0:
            # Starting from the very beginning — add count-in
            self._playback_ms = -self._count_in_ms
            self._last_count_in_beat = -1
            self._song_completed = False
            self._is_new_best = False
            self._weakest_sections = []
            self._recommendations = []
        self._playing = not self._playing
        if self._playing:
            self._last_tick = time.perf_counter()
            # Only start audio capture when past count-in
            if self._audio_enabled and self._playback_ms >= 0:
                self._start_audio()
            if self._midi_player is not None:
                if self._playback_ms >= 0:
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
        # Update signal level meter and tuner even when paused (so user can verify signal)
        if self._audio_capture is not None:
            raw_db = self._audio_capture.get_signal_db()
            self._signal_db = raw_db
            self._signal_db_smooth = self._signal_db_smooth * 0.7 + raw_db * 0.3
            freq, conf = self._audio_capture.get_tuner_data()
            self._tuner_freq = freq
            self._tuner_confidence = conf
            if freq > 0 and conf > 0.5:
                # Frequency jump guard: ignore wild jumps (> 50% change)
                if (self._tuner_freq_smooth > 0
                        and abs(freq - self._tuner_freq_smooth) / self._tuner_freq_smooth > 0.5):
                    # Wild jump — use very low alpha to dampen
                    alpha = 0.02
                else:
                    # Adaptive EMA: high confidence → faster, low → slower
                    alpha = 0.10 if conf > 0.8 else 0.03
                if self._tuner_freq_smooth > 0:
                    self._tuner_freq_smooth = self._tuner_freq_smooth * (1 - alpha) + freq * alpha
                else:
                    self._tuner_freq_smooth = freq
                # Note hysteresis: only change displayed note after 8 stable frames (~130ms)
                from pickhero.audio.note_utils import freq_to_midi
                candidate_note = freq_to_midi(self._tuner_freq_smooth)
                if candidate_note != self._tuner_displayed_note:
                    self._tuner_note_stable_frames += 1
                    if self._tuner_note_stable_frames >= 8:
                        self._tuner_displayed_note = candidate_note
                        self._tuner_note_stable_frames = 0
                else:
                    self._tuner_note_stable_frames = 0
            else:
                # Slow decay instead of instant reset
                self._tuner_freq_smooth *= 0.92
                if self._tuner_freq_smooth < 20.0:
                    self._tuner_freq_smooth = 0.0
                    self._tuner_displayed_note = -1
                    self._tuner_note_stable_frames = 0

        if not self._playing:
            return

        now = time.perf_counter()
        prev_ms = self._playback_ms
        if self._last_tick is not None:
            elapsed_ms = (now - self._last_tick) * 1000.0 * self._tempo_factor
            self._playback_ms += elapsed_ms
        self._last_tick = now

        # Wait mode: freeze if there are pending notes the player hasn't hit yet
        if (self._wait_mode and self._audio_enabled
                and self._playback_ms >= 0 and self._matcher is not None):
            if self._matcher.has_pending_notes_at(self._playback_ms):
                self._playback_ms = prev_ms
                self._last_tick = now
                self._wait_mode_frozen = True
                if self._midi_player is not None and not self._backing_muted:
                    self._midi_player.pause()
            elif self._wait_mode_frozen:
                self._wait_mode_frozen = False
                if self._midi_player is not None and not self._backing_muted:
                    self._midi_player.seek(self._playback_ms)

        # Count-in: play metronome clicks and start audio/midi when crossing 0
        if prev_ms < 0:
            # Play count-in clicks at beat boundaries
            if self._count_in_ms > 0 and self._midi_player is not None:
                beat_index = int((self._count_in_ms + self._playback_ms) / self._ms_per_beat)
                if beat_index > self._last_count_in_beat:
                    self._midi_player.play_click(100)
                    self._last_count_in_beat = beat_index

            # Crossed from negative to non-negative — song starts
            if self._playback_ms >= 0:
                if self._audio_enabled:
                    self._start_audio()
                if self._midi_player is not None:
                    self._midi_player.seek(0)

        # Process audio matching (only during actual song, not count-in)
        if (self._playback_ms >= 0
                and self._audio_enabled
                and self._audio_capture is not None
                and self._matcher is not None):
            detected = self._audio_capture.get_notes()
            for d in detected:
                d.timestamp_ms *= self._tempo_factor
            # While frozen in wait mode, pin detected timestamps to the frozen
            # playback position so matching hits the notes at the hit zone,
            # not future notes that drift ahead as real time passes.
            if self._wait_mode_frozen and detected:
                pinned_ts = self._playback_ms - self._matcher.audio_offset_ms
                for d in detected:
                    d.timestamp_ms = pinned_ts
            results = self._matcher.process_detected_notes(detected, self._playback_ms)
            # FFT-based chord verification for multi-note passages
            if (self._audio_capture is not None
                    and self._audio_capture.chord_detector is not None):
                results.extend(self._matcher.verify_chord_at(
                    self._playback_ms,
                    self._audio_capture.chord_detector,
                ))
            self._feedback.add_results(results, self._playback_ms)
            self._feedback.cleanup(self._playback_ms)

            # Feed timing observations to the overlay (delta since last frame)
            if self._timing_judge and self._timing_overlay is not None:
                observations = self._matcher.get_timing_observations()
                new_obs = observations[self._last_obs_count:]
                for obs in new_obs:
                    if obs.verdict in (TimingVerdict.EARLY, TimingVerdict.LATE, TimingVerdict.ON_TIME):
                        # Find the matching NoteEvent to key the overlay
                        for note in self._timeline.get_active_notes_at_time(
                            obs.expected_ms, self._config.timing_window_ms
                        ):
                            if (note.midi_note == obs.expected_midi
                                    and note.timestamp_ms == obs.expected_ms):
                                self._timing_overlay.add(note, obs.timing_error_ms, self._playback_ms)
                                break
                self._last_obs_count = len(observations)
                self._timing_overlay.cleanup(self._playback_ms)

        # Advance MIDI backing track (only during actual song)
        if self._playback_ms >= 0 and self._midi_player is not None:
            self._midi_player.update(self._playback_ms)

        # Loop check — jump back to start marker when reaching end marker
        # (no count-in on loop)
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

            if not self._song_completed:
                if (self._audio_enabled
                        and self._matcher is not None
                        and self._progress_tracker is not None
                        and self._song_key):
                    # Audio-scored completion
                    stats = self._matcher.get_statistics()
                    if stats["total"] > 0:
                        weakest = self._matcher.get_weakest_sections()
                        self._is_new_best, self._recommendations = (
                            self._progress_tracker.record_detailed_result(
                                self._song_key, stats,
                                weakest, self._tempo_factor,
                            )
                        )
                        self._weakest_sections = weakest
                        self._song_completed = True
                        # Compute timing summary if Timing Judge is active
                        if self._timing_judge:
                            self._compute_timing_summary()
                elif not self._audio_enabled:
                    # Auto-scroll (passive) completion
                    self._weakest_sections = []
                    self._song_completed = True

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
        elif event.key == pygame.K_x:
            self.set_noise_gate_db(self._noise_gate_db - 5)
        elif event.key == pygame.K_c:
            self.set_noise_gate_db(self._noise_gate_db + 5)
        elif event.key == pygame.K_t:
            self._cycle_theme()
        elif event.key == pygame.K_f:
            self._cycle_fret_limit()
        elif event.key == pygame.K_F1:
            self._toggle_string(1)
        elif event.key == pygame.K_F2:
            self._toggle_string(2)
        elif event.key == pygame.K_F3:
            self._toggle_string(3)
        elif event.key == pygame.K_F4:
            self._toggle_string(4)
        elif event.key == pygame.K_F5:
            self._toggle_string(5)
        elif event.key == pygame.K_F6:
            self._toggle_string(6)
        elif event.key == pygame.K_v:
            self._toggle_chord_mode()
        elif event.key == pygame.K_l:
            self._loop_weakest_section()
        elif event.key == pygame.K_h:
            self._show_help = not self._show_help
        elif event.key == pygame.K_w:
            self._toggle_wait_mode()
        elif event.key == pygame.K_j:
            self._toggle_timing_judge()
        elif event.key == pygame.K_k:
            self._toggle_pitch_strict()
        elif event.key == pygame.K_TAB:
            return "next_track"

        return None

    def render(self, surface: pygame.Surface) -> None:
        """Draw the full playing screen."""
        t = get_theme()
        layout = self._layout(surface)

        surface.fill(t.bg)
        self._draw_lanes(surface, layout)
        self._draw_loop_region(surface, layout)
        self._draw_hit_zone(surface, layout)
        self._draw_notes(surface, layout)
        self._draw_hud(surface, layout)

        if self._show_help:
            self._draw_help_overlay(surface, layout)

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
        t = get_theme()
        for i in range(6):
            y = LANE_TOP_MARGIN + i * layout.lane_height
            bg = t.lane_bg_even if i % 2 == 0 else t.lane_bg_odd
            pygame.draw.rect(
                surface, bg,
                (0, y, layout.screen_w, layout.lane_height),
            )
            # Divider line at bottom of lane
            line_y = int(y + layout.lane_height)
            pygame.draw.line(
                surface, t.lane_line,
                (0, line_y), (layout.screen_w, line_y),
            )

    def _draw_hit_zone(self, surface: pygame.Surface, layout: _Layout) -> None:
        t = get_theme()
        x = int(layout.hit_zone_x)
        top = int(LANE_TOP_MARGIN)
        bottom = int(LANE_TOP_MARGIN + 6 * layout.lane_height)
        pygame.draw.line(surface, t.hit_zone, (x, top), (x, bottom), 2)

    def _draw_notes(self, surface: pygame.Surface, layout: _Layout) -> None:
        t = get_theme()
        # Visible time range with margins for long notes
        view_start = self._playback_ms - LEFT_MARGIN_MS
        view_end = self._playback_ms + self._visible_window_ms + RIGHT_MARGIN_MS

        notes = self._timeline.get_notes_in_range(view_start, view_end)

        fret_font_size = max(12, int(layout.note_h * 0.55))
        fret_font = _get_font("consolas", fret_font_size)

        for note in notes:
            # Difficulty filter: skip notes that fail
            if not self._note_passes_filter(note):
                continue

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
            pygame.draw.rect(surface, t.note_border, rect, width=2, border_radius=NOTE_CORNER_RADIUS)

            # Fret number with outline, left-aligned inside note
            fret_label = str(note.fret)
            fret_text = _font_cache.render(fret_font, fret_label, True, t.note_text)
            if fret_text.get_width() + 4 <= rect.width:
                tx = rect.x + 4
                ty = rect.y + rect.height // 2 - fret_text.get_height() // 2
                # Black outline (render at offsets)
                outline = _font_cache.render(fret_font, fret_label, True, (0, 0, 0))
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    surface.blit(outline, (tx + dx, ty + dy))
                surface.blit(fret_text, (tx, ty))

            # Timing Judge: draw early/late indicator above the note
            if self._timing_judge and self._timing_overlay is not None:
                error = self._timing_overlay.get_indicator(note, self._playback_ms)
                if error is not None:
                    indicator_y = int(y) - 12
                    cx = int(x) + int(w) // 2
                    if error < -25:
                        # Early: blue left-pointing arrow
                        col = t.timing_early
                        pygame.draw.polygon(surface, col, [
                            (cx - 8, indicator_y), (cx + 2, indicator_y - 5),
                            (cx + 2, indicator_y + 5),
                        ])
                    elif error > 25:
                        # Late: red right-pointing arrow
                        col = t.timing_late
                        pygame.draw.polygon(surface, col, [
                            (cx + 8, indicator_y), (cx - 2, indicator_y - 5),
                            (cx - 2, indicator_y + 5),
                        ])
                    else:
                        # On time: green dot
                        col = t.timing_on_time
                        pygame.draw.circle(surface, col, (cx, indicator_y), 3)
    def _draw_hud(self, surface: pygame.Surface, layout: _Layout) -> None:
        t = get_theme()
        title_font = _get_font("arial", 20)
        time_font = _get_font("consolas", 20)
        hint_font = _get_font("arial", 14)

        meta = self._timeline.metadata
        w = layout.screen_w
        h = layout.screen_h

        # Count-in overlay — large centered beat countdown
        if self._playback_ms < 0 and self._count_in_ms > 0:
            remaining_beats = int(-self._playback_ms / self._ms_per_beat) + 1
            remaining_beats = min(remaining_beats, self._config.count_in_beats)
            countdown_font = _get_font("arial", 120)
            countdown_surf = _font_cache.render(countdown_font, str(remaining_beats), True, t.hud_accent)
            surface.blit(
                countdown_surf,
                (w // 2 - countdown_surf.get_width() // 2,
                 h // 2 - countdown_surf.get_height() // 2),
            )

        # Song completion overlay
        if self._song_completed:
            self._draw_completion_overlay(surface, layout)

        # Top-left: title + artist
        title = meta.title or "Untitled"
        if meta.artist:
            title = f"{meta.artist} — {title}"
        title_surf = _font_cache.render(title_font, title, True, t.hud_text)
        surface.blit(title_surf, (12, 12))

        # Top-center: BPM with tempo percentage (and streak below it)
        pct = int(self._tempo_factor * 100)
        bpm_text = f"{meta.tempo} BPM ({pct}%)"
        bpm_surf = _font_cache.render(title_font, bpm_text, True, t.hud_accent)
        surface.blit(bpm_surf, (w // 2 - bpm_surf.get_width() // 2, 12))

        # Loop status below BPM
        loop_y = 36
        loop_info = self._loop_hud_text()
        if loop_info:
            loop_color = t.hud_accent if self._loop_enabled else t.hud_text
            loop_surf = _font_cache.render(hint_font, loop_info, True, loop_color)
            surface.blit(loop_surf, (w // 2 - loop_surf.get_width() // 2, loop_y))
            loop_y += 18

        if self._audio_enabled:
            self._feedback.draw_streak(surface, title_font, w // 2, loop_y)

        # Timing Judge live readout (below streak)
        if self._timing_judge and self._matcher is not None:
            tstats = self._matcher.get_timing_stats()
            tj_y = loop_y + 20
            tj_text = (
                f"TIMING JUDGE  |  "
                f"Early: {tstats.early_count}  Late: {tstats.late_count}  "
                f"\u03c3: {tstats.std_dev_ms:.0f}ms"
            )
            tj_surf = _font_cache.render(hint_font, tj_text, True, t.hud_accent)
            surface.blit(tj_surf, (w // 2 - tj_surf.get_width() // 2, tj_y))
            tj_y += 16
            if self._pitch_strict:
                ps_surf = _font_cache.render(hint_font, "STRICT PITCH", True, t.feedback_close)
                surface.blit(ps_surf, (w // 2 - ps_surf.get_width() // 2, tj_y))
        # Top-right: time
        current = format_time(self._playback_ms)
        total = format_time(self._timeline.duration_ms)
        time_text = f"{current} / {total}"
        time_surf = _font_cache.render(time_font, time_text, True, t.hud_text)
        surface.blit(time_surf, (w - time_surf.get_width() - 12, 12))

        # Top-right second line: accuracy stats
        stats_bottom_y = 36
        if self._audio_enabled and self._matcher is not None:
            stats = self._matcher.get_statistics()
            if stats["total"] > 0:
                self._feedback.draw_stats(surface, stats, hint_font, w - 12, 36)
                stats_bottom_y = 54

        # Top-right: noise gate + signal meter + tuner (below stats, when audio capture exists)
        if self._audio_enabled:
            gate_text = f"Gate: {int(self._noise_gate_db)} dB"
            gate_surf = _font_cache.render(hint_font, gate_text, True, t.hud_accent)
            surface.blit(gate_surf, (w - gate_surf.get_width() - 12, stats_bottom_y))
            if self._audio_capture is not None:
                self._draw_signal_meter(surface, hint_font, w, stats_bottom_y + 18)
                self._draw_tuner(surface, hint_font, w, stats_bottom_y + 36)
        elif self._audio_capture is not None:
            # Audio off but capture exists — still show meter and tuner
            self._draw_signal_meter(surface, hint_font, w, stats_bottom_y)
            self._draw_tuner(surface, hint_font, w, stats_bottom_y + 18)

        # Bottom-center: play state + controls
        if self._playback_ms < 0:
            state = "Count-in"
        elif self._playing:
            if self._wait_mode_frozen:
                state = "Waiting..."
            elif self._audio_enabled:
                state = "Playing"
            else:
                state = "Auto-scroll"
        else:
            state = "Paused"
        audio_state = "ON" if self._audio_enabled else "off"
        loop_state = "ON" if self._loop_enabled else "off"
        backing_state = ""
        if self._midi_player is not None:
            backing_state = f"|  B: backing {'off' if self._backing_muted else 'ON'}  "
        wait_state = ""
        if self._wait_mode:
            wait_state = f"|  W: wait {'WAIT' if self._wait_mode_frozen else 'ON'}  "
        elif self._audio_enabled:
            wait_state = "|  W: wait off  "
        timing_state = f"|  J: timing {'ON' if self._timing_judge else 'off'}  "
        if self._pitch_strict:
            timing_state += "|  K: strict ON  "
        hint = (
            f"{state}  |  SPACE: play/pause  |  LEFT/RIGHT: seek  "
            f"|  HOME: restart  |  PgDn/PgUp: tempo  |  X/C: gate"
            f"|  A: audio {audio_state}  "
            f"{backing_state}"
            f"{wait_state}"
            f"{timing_state}"
            f"|  I/O: loop {loop_state}  |  P: toggle  |  ESC: menu"
        )
        hint_surf = _font_cache.render(hint_font, hint, True, t.hud_text)
        y = layout.screen_h - LANE_BOTTOM_MARGIN + 8
        surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, y))

        # Top-left second line: track name + filter info
        info_y = 38
        if meta.track_name:
            track_surf = _font_cache.render(hint_font, f"Track: {meta.track_name}", True, t.hud_text)
            surface.blit(track_surf, (12, info_y))
            info_y += 16

        # Difficulty filter HUD
        filter_text = self._filter_hud_text()
        if filter_text:
            filter_surf = _font_cache.render(hint_font, filter_text, True, t.hud_accent)
            surface.blit(filter_surf, (12, info_y))
            info_y += 16

        # Chord mode HUD
        if self._chord_partial_credit != self._config._default_chord_partial_credit:
            chord_text = "Chords: strict" if self._chord_partial_credit else "Chords: easy"
            chord_surf = _font_cache.render(hint_font, chord_text, True, t.hud_accent)
            surface.blit(chord_surf, (12, info_y))

    def _draw_signal_meter(self, surface: pygame.Surface, font: pygame.font.Font,
                           screen_w: int, y: int) -> None:
        """Draw a compact horizontal signal level meter with dB label."""
        t = get_theme()
        db = self._signal_db_smooth

        bar_w = 100
        bar_h = 8
        db_min = -80.0
        db_max = -10.0

        # dB label
        db_display = max(db_min, min(db_max, db))
        label = f"Signal: {int(db_display)} dB"
        label_surf = _font_cache.render(font, label, True, t.hud_text)
        label_x = screen_w - label_surf.get_width() - 12
        surface.blit(label_surf, (label_x, y))

        # Bar position: to the left of the label
        bar_x = label_x - bar_w - 8
        bar_y = y + label_surf.get_height() // 2 - bar_h // 2

        # Bar background
        pygame.draw.rect(surface, t.signal_cold, (bar_x, bar_y, bar_w, bar_h))

        # Fill proportion
        fill_frac = max(0.0, min(1.0, (db - db_min) / (db_max - db_min)))
        fill_w = int(fill_frac * bar_w)

        if fill_w > 0:
            if db >= -30:
                color = t.signal_hot
            elif db >= self._noise_gate_db:
                color = t.signal_warm
            else:
                color = t.signal_cold
            pygame.draw.rect(surface, color, (bar_x, bar_y, fill_w, bar_h))

        # Bar border
        pygame.draw.rect(surface, t.hud_text, (bar_x, bar_y, bar_w, bar_h), 1)

        # Noise gate tick mark
        gate_frac = max(0.0, min(1.0, (self._noise_gate_db - db_min) / (db_max - db_min)))
        gate_x = bar_x + int(gate_frac * bar_w)
        pygame.draw.line(surface, t.hud_accent, (gate_x, bar_y - 2), (gate_x, bar_y + bar_h + 2), 1)

    def _draw_tuner(self, surface: pygame.Surface, font: pygame.font.Font,
                    screen_w: int, y: int) -> None:
        """Draw a compact tuner display with cents bar and note name."""
        t = get_theme()

        bar_w = 100
        bar_h = 8

        freq = self._tuner_freq_smooth

        if freq <= 0 or self._tuner_displayed_note < 0:
            # No pitch — show placeholder
            label = "Tuner: ---"
            label_surf = _font_cache.render(font, label, True, t.hud_text)
            surface.blit(label_surf, (screen_w - label_surf.get_width() - 12, y))
            return

        # Use hysteresis-stabilized note for the label, smoothed freq for cents
        midi_note, cents = freq_to_cents_deviation(freq)
        if midi_note < 0:
            return

        note_name = midi_to_name(self._tuner_displayed_note)
        # Recompute cents relative to the displayed note for consistency
        from pickhero.audio.note_utils import midi_to_freq as _mtf
        target_freq = _mtf(self._tuner_displayed_note)
        if target_freq > 0:
            import math
            cents = 1200 * math.log2(freq / target_freq)

        # Choose color based on cents deviation
        abs_cents = abs(cents)
        if abs_cents < 5:
            fill_color = t.tuner_in_tune
        elif abs_cents < 15:
            fill_color = t.tuner_close
        else:
            fill_color = t.tuner_off

        # Note name + cents label
        sign = "+" if cents >= 0 else ""
        label = f"{note_name} {sign}{int(cents)}\u00A2"
        label_surf = _font_cache.render(font, label, True, fill_color)
        label_x = screen_w - label_surf.get_width() - 12
        surface.blit(label_surf, (label_x, y))

        # Bar position: to the left of the label
        bar_x = label_x - bar_w - 8
        bar_y = y + label_surf.get_height() // 2 - bar_h // 2

        # Bar background
        pygame.draw.rect(surface, t.signal_cold, (bar_x, bar_y, bar_w, bar_h))

        # Fill indicator: center = in-tune, left = flat, right = sharp
        center_x = bar_x + bar_w // 2
        fill_offset = int((cents / 50.0) * (bar_w // 2))
        fill_offset = max(-bar_w // 2, min(bar_w // 2, fill_offset))

        if fill_offset >= 0:
            pygame.draw.rect(surface, fill_color,
                             (center_x, bar_y, fill_offset, bar_h))
        else:
            pygame.draw.rect(surface, fill_color,
                             (center_x + fill_offset, bar_y, -fill_offset, bar_h))

        # Bar border
        pygame.draw.rect(surface, t.hud_text, (bar_x, bar_y, bar_w, bar_h), 1)

        # Center tick mark (in-tune reference)
        pygame.draw.line(surface, t.hud_text,
                         (center_x, bar_y - 2), (center_x, bar_y + bar_h + 2), 1)

    def _draw_completion_overlay(self, surface: pygame.Surface, layout: _Layout) -> None:
        """Draw the song completion results overlay."""
        t = get_theme()
        w, h = layout.screen_w, layout.screen_h

        # Semi-transparent dark overlay
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surface.blit(overlay, (0, 0))

        header_font = _get_font("arial", 48)
        stat_font = _get_font("consolas", 28)
        hint_font = _get_font("arial", 18)

        center_y = h // 2 - 80

        # "Song Complete!" header
        header_surf = _font_cache.render(header_font, "Song Complete!", True, t.hud_accent)
        surface.blit(header_surf, (w // 2 - header_surf.get_width() // 2, center_y))

        if self._audio_enabled and self._matcher is not None:
            # Accuracy stats
            stats = self._matcher.get_statistics()
            accuracy_text = (
                f"Accuracy: {stats['accuracy_percent']:.1f}%  "
                f"({stats['hits']}/{stats['total']})"
            )
            acc_surf = _font_cache.render(stat_font, accuracy_text, True, t.hud_text)
            surface.blit(acc_surf, (w // 2 - acc_surf.get_width() // 2, center_y + 60))

            # "New Best!" indicator
            if self._is_new_best:
                best_surf = _font_cache.render(stat_font, "New Best!", True, (255, 220, 50))
                surface.blit(best_surf, (w // 2 - best_surf.get_width() // 2, center_y + 100))

            # Weakest sections
            weak = getattr(self, "_weakest_sections", [])
            if weak:
                section = weak[0]
                weak_text = (
                    f"Weakest: bars {section[0]+1}-{section[1]+1} "
                    f"({section[2]:.0f}%) -- press L to loop"
                )
                weak_surf = _font_cache.render(hint_font, weak_text, True, t.feedback_close)
                surface.blit(weak_surf, (w // 2 - weak_surf.get_width() // 2, center_y + 140))

            # Practice recommendations
            rec_y = center_y + 170
            for rec in self._recommendations:
                rec_surf = _font_cache.render(hint_font, rec, True, t.hud_accent)
                surface.blit(rec_surf, (w // 2 - rec_surf.get_width() // 2, rec_y))
                rec_y += 24

            # Controls hint
            hint_y = max(center_y + 180, rec_y + 10)
            hint_text = "SPACE to replay  |  L to loop weak section  |  ESC to menu"
            hint_surf = _font_cache.render(hint_font, hint_text, True, t.hud_text)
            surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, hint_y))
        else:
            # Auto-scroll completion — no stats
            hint_text = "SPACE to replay  |  ESC to menu"
            hint_surf = _font_cache.render(hint_font, hint_text, True, t.hud_text)
            surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, center_y + 70))

        # Timing Judge summary (augments the completion overlay)
        if self._timing_judge and self._timing_summary is not None:
            self._draw_timing_summary(surface, layout)

    def _compute_timing_summary(self) -> None:
        """Compute timing stats and worst measures for the summary screen."""
        if self._matcher is None:
            return
        stats = self._matcher.get_timing_stats()
        self._timing_summary = stats
        # Find worst 3 measures by abs(mean_error_ms) descending
        if stats.per_measure:
            sorted_measures = sorted(
                stats.per_measure.items(),
                key=lambda x: abs(x[1].mean_error_ms),
                reverse=True,
            )
            self._timing_worst_measures = [
                (idx, ms.mean_error_ms, ms.std_dev_ms)
                for idx, ms in sorted_measures[:3]
            ]

    def _draw_timing_summary(self, surface: pygame.Surface, layout: _Layout) -> None:
        """Draw the Timing Judge results panel within the completion overlay."""
        t = get_theme()
        w, h = layout.screen_w, layout.screen_h
        stats = self._timing_summary
        if stats is None:
            return

        stat_font = _get_font("consolas", 22)
        small_font = _get_font("arial", 16)
        hist_font = _get_font("consolas", 14)

        # Panel position: right side of the completion overlay
        panel_x = w - 460
        panel_y = h // 2 - 100

        # Title
        title_surf = _font_cache.render(stat_font, "Timing Judge", True, t.hud_accent)
        surface.blit(title_surf, (panel_x, panel_y))
        panel_y += 30

        if stats.count == 0:
            no_data = _font_cache.render(
                small_font,
                "No timing data -- enable audio (A) and play a take",
                True, t.hud_text,
            )
            surface.blit(no_data, (panel_x, panel_y))
            return

        # Mean error
        mean_sign = "+" if stats.mean_error_ms >= 0 else ""
        mean_text = f"\u03bc = {mean_sign}{stats.mean_error_ms:.1f} ms"
        mean_color = t.timing_late if stats.mean_error_ms > 5 else (
            t.timing_early if stats.mean_error_ms < -5 else t.timing_on_time
        )
        mean_surf = _font_cache.render(stat_font, mean_text, True, mean_color)
        surface.blit(mean_surf, (panel_x, panel_y))
        panel_y += 28

        # Std dev
        std_text = f"\u03c3 = {stats.std_dev_ms:.1f} ms"
        std_surf = _font_cache.render(stat_font, std_text, True, t.hud_text)
        surface.blit(std_surf, (panel_x, panel_y))
        panel_y += 28

        # Counts
        counts_text = (
            f"Early: {stats.early_count}  On-time: {stats.on_time_count}  "
            f"Late: {stats.late_count}"
        )
        counts_surf = _font_cache.render(small_font, counts_text, True, t.hud_text)
        surface.blit(counts_surf, (panel_x, panel_y))
        panel_y += 20

        missed_extra = f"Missed: {stats.missed_count}  Extra: {stats.extra_count}"
        me_surf = _font_cache.render(small_font, missed_extra, True, t.hud_text)
        surface.blit(me_surf, (panel_x, panel_y))
        panel_y += 28

        # Histogram
        hist_label = _font_cache.render(small_font, "Error Distribution", True, t.hud_accent)
        surface.blit(hist_label, (panel_x, panel_y))
        panel_y += 20

        hist_w = 400
        hist_h = 100
        hist_x = panel_x
        hist_y = panel_y

        # Background
        pygame.draw.rect(surface, t.signal_cold, (hist_x, hist_y, hist_w, hist_h))

        max_bin = max(stats.histogram_bins) if stats.histogram_bins else 0
        bar_w = hist_w // len(stats.histogram_bins)
        center_bin = len(stats.histogram_bins) // 2

        for i, count in enumerate(stats.histogram_bins):
            if count == 0 or max_bin == 0:
                continue
            bar_h = int((count / max_bin) * (hist_h - 4))
            bx = hist_x + i * bar_w
            by = hist_y + hist_h - bar_h
            if i < center_bin:
                color = t.timing_early
            elif i > center_bin:
                color = t.timing_late
            else:
                color = t.timing_on_time
            pygame.draw.rect(surface, color, (bx + 1, by, bar_w - 2, bar_h))

        # Center line (0ms reference)
        center_x = hist_x + center_bin * bar_w + bar_w // 2
        pygame.draw.line(surface, t.hud_text, (center_x, hist_y - 2), (center_x, hist_y + hist_h + 2), 1)

        # Axis labels
        left_label = _font_cache.render(hist_font, "-100ms", True, t.hud_text)
        right_label = _font_cache.render(hist_font, "+100ms", True, t.hud_text)
        zero_label = _font_cache.render(hist_font, "0", True, t.hud_text)
        surface.blit(left_label, (hist_x, hist_y + hist_h + 2))
        surface.blit(right_label, (hist_x + hist_w - right_label.get_width(), hist_y + hist_h + 2))
        surface.blit(zero_label, (center_x - zero_label.get_width() // 2, hist_y + hist_h + 2))
        panel_y += hist_h + 22

        # Worst 3 bars
        if self._timing_worst_measures:
            worst_label = _font_cache.render(small_font, "Worst Bars:", True, t.hud_accent)
            surface.blit(worst_label, (panel_x, panel_y))
            panel_y += 20
            for measure_idx, mean_err, std_dev in self._timing_worst_measures:
                sign = "+" if mean_err >= 0 else ""
                line = f"  Bar {measure_idx + 1}: \u03bc={sign}{mean_err:.0f}ms  \u03c3={std_dev:.0f}ms"
                worst_color = t.timing_late if mean_err > 5 else (
                    t.timing_early if mean_err < -5 else t.timing_on_time
                )
                worst_surf = _font_cache.render(small_font, line, True, worst_color)
                surface.blit(worst_surf, (panel_x, panel_y))
                panel_y += 18
    def _draw_help_overlay(self, surface: pygame.Surface, layout: _Layout) -> None:
        """Draw a help overlay explaining the track, note colors, and controls."""
        t = get_theme()
        w, h = layout.screen_w, layout.screen_h

        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        surface.blit(overlay, (0, 0))

        title_font = _get_font("arial", 28)
        section_font = _get_font("arial", 20)
        body_font = _get_font("arial", 17)
        hint_font = _get_font("arial", 14)

        cx = w // 2
        y = 30

        title_surf = _font_cache.render(title_font, "Help", True, t.hud_accent)
        surface.blit(title_surf, (cx - title_surf.get_width() // 2, y))
        y += 40

        # -- How to read the track --
        lx = cx - 260  # left edge for content
        section = _font_cache.render(section_font, "Reading the Track", True, t.hud_accent)
        surface.blit(section, (lx, y))
        y += 26

        track_lines = [
            "Notes scroll right-to-left toward the hit zone (white vertical line).",
            "The number on each note is the fret to press (0 = open string).",
            "Play the right fret on the right string as the note crosses the line.",
        ]
        for line in track_lines:
            surf = _font_cache.render(body_font, line, True, t.hud_text)
            surface.blit(surf, (lx, y))
            y += 21
        y += 6

        # -- The 6 rows --
        section = _font_cache.render(section_font, "The 6 Rows = 6 Guitar Strings", True, t.hud_accent)
        surface.blit(section, (lx, y))
        y += 26

        row_lines = [
            "Each horizontal row is one guitar string, top to bottom:",
        ]
        for line in row_lines:
            surf = _font_cache.render(body_font, line, True, t.hud_text)
            surface.blit(surf, (lx, y))
            y += 21

        string_info = [
            (1, "Row 1 (top)     = high E  (thinnest)"),
            (2, "Row 2              = B"),
            (3, "Row 3              = G"),
            (4, "Row 4              = D"),
            (5, "Row 5              = A"),
            (6, "Row 6 (bottom) = low E  (thickest)"),
        ]
        for s, label in string_info:
            color = STRING_COLORS.get(s, (180, 180, 180))
            pygame.draw.rect(surface, color, (lx, y + 3, 14, 14),
                             border_radius=2)
            surf = _font_cache.render(body_font, label, True, t.hud_text)
            surface.blit(surf, (lx + 20, y))
            y += 20
        y += 4

        surf = _font_cache.render(body_font, "A note's color tells you which string to play — it matches the row.", True, t.hud_text)
        surface.blit(surf, (lx, y))
        y += 20
        surf = _font_cache.render(body_font, "Dimmed notes have already passed the hit zone.", True, t.hud_text)
        surface.blit(surf, (lx, y))
        y += 24

        # -- Feedback colors --
        section = _font_cache.render(section_font, "Scoring (colors change after you play)", True, t.hud_accent)
        surface.blit(section, (lx, y))
        y += 26

        surf = _font_cache.render(body_font, "When audio is on, notes change color after they pass the hit zone:", True, t.hud_text)
        surface.blit(surf, (lx, y))
        y += 22

        feedback = [
            (t.feedback_hit, "Turns green", "you played the correct note"),
            (t.feedback_close, "Turns yellow", "close, off by 1 semitone"),
            (t.feedback_miss, "Turns red", "you missed it (not played in time)"),
        ]
        for color, label, desc in feedback:
            pygame.draw.rect(surface, color, (lx + 10, y + 3, 14, 14),
                             border_radius=2)
            surf = _font_cache.render(body_font, f"{label} — {desc}", True, t.hud_text)
            surface.blit(surf, (lx + 30, y))
            y += 21
        y += 10

        # -- Controls --
        section = _font_cache.render(section_font, "Controls", True, t.hud_accent)
        surface.blit(section, (lx, y))
        y += 24

        controls = [
            "SPACE: play/pause    LEFT/RIGHT: seek    HOME: restart",
            "A: toggle audio    PgDn/PgUp: tempo    X/C: noise gate",
            "B: backing track    T: theme    I/O: loop markers    P: toggle loop",
            "F: fret limit    F1-F6: toggle strings    V: chord mode    L: loop weakest",
            "W: wait mode (pause until correct note played)",
        ]
        for line in controls:
            surf = _font_cache.render(hint_font, line, True, t.hud_text)
            surface.blit(surf, (lx, y))
            y += 18

        y += 10
        close_surf = _font_cache.render(hint_font, "Press H to close", True, t.hud_accent)
        surface.blit(close_surf, (cx - close_surf.get_width() // 2, y))

    # -- Difficulty filter --

    def _cycle_fret_limit(self) -> None:
        """Cycle through fret limit options."""
        try:
            idx = FRET_LIMITS.index(self._max_fret)
            self._max_fret = FRET_LIMITS[(idx + 1) % len(FRET_LIMITS)]
        except ValueError:
            self._max_fret = FRET_LIMITS[0]
        self._config.max_fret = self._max_fret
        self._config.save()
        self._reset_matcher_for_filter()

    def _toggle_string(self, string: int) -> None:
        """Toggle a string on/off in the difficulty filter."""
        idx = string - 1
        self._active_strings[idx] = not self._active_strings[idx]
        # Don't allow all strings to be off
        if not any(self._active_strings):
            self._active_strings[idx] = True
            return
        self._config.active_strings = list(self._active_strings)
        self._config.save()
        self._reset_matcher_for_filter()

    def _reset_matcher_for_filter(self) -> None:
        """Reset matcher when filter changes mid-song."""
        if self._matcher:
            self._matcher.reset()
            self._matcher.note_filter = self._note_passes_filter
        self._feedback.reset()

    def _filter_hud_text(self) -> str | None:
        """Return difficulty filter text for HUD, or None if default."""
        parts = []
        if self._max_fret < 24:
            parts.append(f"Fret: 0-{self._max_fret}")
        if not all(self._active_strings):
            strs = " ".join(
                str(i + 1) if on else "_"
                for i, on in enumerate(self._active_strings)
            )
            parts.append(f"Strings: {strs}")
        return "  |  ".join(parts) if parts else None

    # -- Theme --

    def _cycle_theme(self) -> None:
        """Toggle between dark and light theme."""
        name = cycle_theme()
        self._config.theme = name
        self._config.save()

    # -- Chord mode --

    def _toggle_chord_mode(self) -> None:
        """Toggle chord partial credit on/off."""
        self._chord_partial_credit = not self._chord_partial_credit
        self._config.chord_partial_credit = self._chord_partial_credit
        self._config.save()
        if self._matcher:
            self._matcher.chord_partial_credit = self._chord_partial_credit

    # -- Wait mode --

    def _toggle_wait_mode(self) -> None:
        """Toggle wait mode on/off."""
        self._wait_mode = not self._wait_mode
        self._config.wait_mode = self._wait_mode
        self._config.save()
        if not self._wait_mode:
            self._wait_mode_frozen = False

    # -- Timing Judge mode --

    def _toggle_timing_judge(self) -> None:
        """Toggle Timing Judge mode on/off."""
        self._timing_judge = not self._timing_judge
        self._config.timing_judge_mode = self._timing_judge
        self._config.save()
        if self._timing_judge and self._timing_overlay is None:
            self._timing_overlay = TimingOverlay()
        if self._matcher:
            self._matcher.timing_judge_enabled = self._timing_judge
        if not self._timing_judge:
            self._last_obs_count = 0

    # -- Pitch strict mode --

    def _toggle_pitch_strict(self) -> None:
        """Toggle pitch strict mode on/off."""
        self._pitch_strict = not self._pitch_strict
        self._config.pitch_strict_mode = self._pitch_strict
        self._config.save()
        if self._matcher:
            self._matcher.pitch_strict = self._pitch_strict

    def _loop_weakest_section(self) -> None:
        """Set loop to weakest section from completion screen.

        When Timing Judge is active, prefer the timing-based worst bar.
        Otherwise, use the accuracy-based weakest section.
        """
        if not self._song_completed:
            return

        # Timing Judge: prefer timing-based worst bar
        if self._timing_judge and self._timing_worst_measures:
            measure_idx = self._timing_worst_measures[0][0]
            measures = self._timeline.measures
            if measures and measure_idx < len(measures):
                start_ms = measures[measure_idx].start_ms
                end_ms = measures[measure_idx].end_ms
                self._loop_start_ms = start_ms
                self._loop_end_ms = end_ms
                self._loop_enabled = True
                self._song_completed = False
                self._is_new_best = False
                self._weakest_sections = []
                self._timing_summary = None
                self._timing_worst_measures = []
                self.seek(start_ms)
                return

        # Accuracy-based weakest section
        weak = getattr(self, "_weakest_sections", [])
        if not weak:
            return
        section = weak[0]
        start_measure, end_measure = section[0], section[1]
        measures = self._timeline.measures
        if not measures or start_measure >= len(measures):
            return
        start_ms = measures[start_measure].start_ms
        end_idx = min(end_measure + 1, len(measures) - 1)
        end_ms = measures[end_idx].end_ms if end_idx < len(measures) else self._timeline.duration_ms
        self._loop_start_ms = start_ms
        self._loop_end_ms = end_ms
        self._loop_enabled = True
        self._song_completed = False
        self._is_new_best = False
        self._weakest_sections = []
        self._timing_summary = None
        self._timing_worst_measures = []
        self.seek(start_ms)

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

        t = get_theme()
        lane_top = int(LANE_TOP_MARGIN)
        lane_bottom = int(LANE_TOP_MARGIN + 6 * layout.lane_height)
        lane_h = lane_bottom - lane_top

        marker_color = t.loop_marker if self._loop_enabled else t.loop_marker_disabled
        region_color = t.loop_region if self._loop_enabled else t.loop_region_disabled

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
        if self._audio_enabled:
            if self._playing:
                self._start_audio()
            else:
                # Start capture for signal monitoring even while paused
                self._start_capture_only()
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
                note_filter=self._note_passes_filter if self._is_filter_active() else None,
                chord_partial_credit=self._chord_partial_credit,
                timing_judge_enabled=self._timing_judge,
                pitch_strict=self._pitch_strict,
            )
            if self._timing_judge and self._timing_overlay is None:
                self._timing_overlay = TimingOverlay()
            self._feedback.reset()
        except Exception as e:
            print(f"Audio start failed: {e}")
            self._audio_enabled = False

    def _start_capture_only(self) -> None:
        """Start audio capture for signal monitoring (no matcher)."""
        try:
            from pickhero.audio.input import AudioCapture
            if self._audio_capture is None:
                self._audio_capture = AudioCapture(self._config)
            self._audio_capture.start()
        except Exception as e:
            print(f"Audio capture start failed: {e}")
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
