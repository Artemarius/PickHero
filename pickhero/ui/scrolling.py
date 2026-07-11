"""Scrolling note display for the playing screen.

Renders 6 string lanes with notes scrolling right-to-left, synchronized
to a playback clock. Optionally captures audio and shows hit/miss feedback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import pygame

from pickhero.audio.midi_playback import BackingTrack, MidiPlayer
from pickhero.audio.verifier_composite import CompositeVerifier
from pickhero.adaptive import AdaptiveDifficultyController
from pickhero.config import Config
from pickhero.matcher import NoteMatcher
from pickhero.audio.match_mode import MatchMode, _coerce_match_mode
from pickhero.progress import ProgressTracker
from pickhero.tabs.timeline import NoteEvent, Timeline
from pickhero.audio.note_utils import freq_to_cents_deviation, midi_to_name
from pickhero.ui.colors import (
    cycle_theme,
    dimmed,
    get_theme,
    string_color,
)
from pickhero.ui.feedback import FeedbackRenderer, TimingOverlay
from pickhero.ui.overlays import (
    CompletionState,
    draw_completion_overlay,
    draw_help_overlay,
    draw_timing_summary,
    draw_why_missed,
)
from pickhero.timing import TimingStats, TimingVerdict

if TYPE_CHECKING:
    from pickhero.audio.input import AudioCapture

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
                 song_key: str = "", on_error: Callable[[str], None] | None = None):
        self._timeline = timeline
        self._visible_beats = visible_beats
        self._hit_zone_fraction = hit_zone_fraction
        self._config = config or Config()
        self._on_error = on_error
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

        self._audio_capture: AudioCapture | None = None  # created on demand
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
        self._weakest_sections: list[tuple[int, int, float]] = []
        self._recommendations: list[str] = []

        # MIDI backing track
        self._backing_muted: bool = False
        self._midi_player: MidiPlayer | None = None
        self._tempo_factor = max(0.5, min(1.0, self._config.tempo_factor))

        self._num_strings: int = max(1, min(12, self._timeline.metadata.num_strings))
        self._max_fret: int = self._config.max_fret
        # Active-string filter mirrors global config, but sized to the track.
        self._active_strings: list[bool] = list(self._config.active_strings[:self._num_strings])
        while len(self._active_strings) < self._num_strings:
            self._active_strings.append(True)

        # Signal level meter
        self._signal_db: float = -120.0
        self._signal_db_smooth: float = -120.0

        # Tuner display
        self._tuner_freq: float = 0.0
        self._tuner_confidence: float = 0.0
        self._tuner_freq_smooth: float = 0.0
        self._tuner_displayed_note: int = -1
        self._tuner_note_stable_frames: int = 0

        # Match mode (replaces chord_partial_credit, timing_judge, pitch_strict).
        # HighAccuracy profile implies JUDGE mode — strict matching for strict audio.
        if self._config.audio.profile == "high_accuracy" and self._config.match_mode != "judge":
            self._config.match_mode = "judge"
            self._config.save()
        self._match_mode: MatchMode = _coerce_match_mode(self._config.match_mode)
        # Derived booleans for read sites that still check them.
        self._chord_partial_credit: bool = self._match_mode != MatchMode.ARCADE
        self._timing_judge: bool = self._match_mode == MatchMode.JUDGE
        self._pitch_strict: bool = self._match_mode == MatchMode.JUDGE

        # Help overlay
        self._show_help: bool = False

        # Wait mode
        self._wait_mode: bool = self._config.wait_mode
        self._wait_mode_frozen: bool = False
        self._timing_summary: TimingStats | None = None
        self._last_obs_count: int = 0
        self._timing_overlay: TimingOverlay | None = None
        self._timing_summary = None
        self._timing_worst_measures: list[tuple[int, float, float]] = []

        # Guided practice
        self._guided_practice: bool = False
        self._guided_loop_count: int = 0
        self._guided_target_accuracy: float = 90.0
        self._guided_consecutive_successes: int = 0
        self._guided_start_tempo: float = 0.0

        # Last detected technique (for HUD display) + recent verdict explanations
        self._last_technique: str | None = None
        self._recent_verdicts: list = []

        # Phrase-level dynamic difficulty. Persisted mastery is loaded per song
        # and the same predicate is used for rendering and scoring.
        persisted_mastery: dict[str, dict] = {}
        if self._progress_tracker is not None and self._song_key:
            record = self._progress_tracker.get_best(self._song_key)
            if record is not None:
                persisted_mastery = record.phrase_mastery
        self._adaptive = AdaptiveDifficultyController(
            self._timeline,
            enabled=self._config.dynamic_difficulty_enabled,
            initial_level=self._config.dynamic_difficulty_start_level,
            target_accuracy=self._config.dynamic_difficulty_target_accuracy / 100.0,
            persisted=persisted_mastery,
        )

    def _note_passes_filter(self, note: NoteEvent) -> bool:
        """Check if a note passes the difficulty filter."""
        if note.fret > self._max_fret:
            return False
        if not self._active_strings[note.string - 1]:
            return False
        if not self._adaptive.accepts(note):
            return False
        return True

    def _is_filter_active(self) -> bool:
        """Check if any difficulty filter is active."""
        return (
            self._max_fret < 24
            or not all(self._active_strings)
            or self._adaptive.enabled
        )

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
            self._last_technique = None
            self._recent_verdicts = []
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
            self._last_technique = None

        self._playing = not self._playing
        if self._playing:
            self._last_tick = time.perf_counter()
            # Only start audio capture when past count-in
            if self._audio_enabled and self._playback_ms >= 0:
                self._start_audio()
                if self._audio_capture is not None:
                    self._audio_capture.clock.set_segment(
                        self._playback_ms, 0.0, self._tempo_factor
                    )
            if self._midi_player is not None:
                if self._playback_ms >= 0:
                    self._midi_player.seek(self._playback_ms)
        else:
            self._last_tick = None
            self._stop_audio()
            if self._midi_player is not None:
                self._midi_player.pause()

    def seek(self, ms: float) -> None:
        """Seek to an absolute position in ms, clamped to [0, duration].

        Playback mapping changes via clock.set_segment; the input stream
        and detector clock remain uninterrupted.
        """
        self._playback_ms = max(0.0, min(ms, self._timeline.duration_ms))
        if self._matcher:
            self._matcher.reset()
        self._feedback.reset()
        if self._midi_player is not None:
            self._midi_player.seek(self._playback_ms)
        # Remap the clock without restarting the input stream.
        if self._audio_capture is not None and self._audio_enabled and self._playing:
            stream_ms = self._audio_capture.stream_time_ms()
            self._audio_capture.clock.set_segment(
                self._playback_ms, stream_ms, self._tempo_factor
            )

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
        if self._audio_capture is not None:
            stream_ms = self._audio_capture.stream_time_ms()
            self._audio_capture.clock.set_segment(
                self._playback_ms, stream_ms, self._tempo_factor
            )
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
                if not self._wait_mode_frozen:
                    # First freeze frame: pin song position and add clock segment
                    self._playback_ms = prev_ms
                    self._last_tick = now
                    self._wait_mode_frozen = True
                    if self._midi_player is not None and not self._backing_muted:
                        self._midi_player.pause()
                    if self._audio_capture is not None:
                        stream_ms = self._audio_capture.stream_time_ms()
                        self._audio_capture.clock.refresh_frozen_anchor(
                            self._playback_ms, stream_ms
                        )
                else:
                    # Keep one frozen segment anchored to the newest captured
                    # samples; this makes repeated attempts judgeable forever.
                    if self._audio_capture is not None:
                        self._audio_capture.clock.refresh_frozen_anchor(
                            self._playback_ms,
                            self._audio_capture.stream_time_ms(),
                        )
            elif self._wait_mode_frozen:
                self._wait_mode_frozen = False
                if self._midi_player is not None and not self._backing_muted:
                    self._midi_player.seek(self._playback_ms)
                if self._audio_capture is not None:
                    stream_ms = self._audio_capture.stream_time_ms()
                    self._audio_capture.clock.set_segment(
                        self._playback_ms, stream_ms, self._tempo_factor
                    )

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
            clock = self._audio_capture.clock
            capture_stream_ms = self._audio_capture.stream_time_ms()
            # Positive input latency means the newest captured sample belongs
            # to an earlier chart position. Never ask the verifier for audio
            # beyond this captured scoring horizon.
            scoring_playback_ms = min(
                self._playback_ms,
                clock.stream_to_song_ms(capture_stream_ms),
            )

            window = self._config.timing_window_ms * 2
            nearby = self._timeline.get_notes_in_range(
                scoring_playback_ms - window,
                scoring_playback_ms + window,
            )
            expected_midi = [n.midi_note for n in nearby]
            self._audio_capture.set_tab_context(expected_midi, scoring_playback_ms)
            detected = self._audio_capture.get_notes()
            for d in detected:
                d.timestamp_ms = clock.stream_to_song_ms(d.timestamp_ms)
            timing_window_ms = self._config.timing_window_ms
            judge_ms = scoring_playback_ms - timing_window_ms
            window_start_song_ms = judge_ms - timing_window_ms - 50.0
            window_end_song_ms = scoring_playback_ms
            window_start_stream_ms = clock.song_to_stream_ms(window_start_song_ms)
            window_end_stream_ms = clock.song_to_stream_ms(window_end_song_ms)
            audio_window = self._audio_capture.get_window_between(
                window_start_stream_ms, window_end_stream_ms
            )

            def audio_for_song_range(start_song_ms: float, end_song_ms: float):
                return self._audio_capture.get_window_between(
                    clock.song_to_stream_ms(start_song_ms),
                    clock.song_to_stream_ms(end_song_ms),
                )

            # Single scoring authority: unified event state machine.
            # Replaces the three legacy paths (verify_hit_zone, process_detected_notes,
            # verify_chord_at) that produced inconsistent judgments per event.
            if self._wait_mode_frozen and detected:
                # While frozen in wait mode, pin detected timestamps to the frozen
                # playback position so matching hits the notes at the hit zone,
                # not future notes that drift ahead as real time passes.
                pinned_ts = self._playback_ms
                for d in detected:
                    d.timestamp_ms = pinned_ts
            results = list(self._matcher.advance_state_machine(
                playback_ms=scoring_playback_ms,
                audio_window=audio_window,
                detected_notes=detected,
            ))
            has_onset = any(d.note.is_onset for d in detected) if detected else False
            self._feedback.add_results(results, self._playback_ms)

            # Track last detected technique for HUD display.
            # Clear if an onset fires without a technique detected.
            new_technique = None
            for d in detected:
                if d.note.performance is not None and d.note.performance.technique_candidates:
                    new_technique = d.note.performance.technique_candidates[0].kind
                    break
            if new_technique:
                self._last_technique = new_technique
            elif has_onset:
                self._last_technique = None

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
            stats: dict = {}
            phrase_stats: dict[int, dict[str, float | int]] = {}
            if self._matcher:
                # Snapshot the completed iteration before reset or _start_audio
                # replaces the matcher.
                stats = self._matcher.get_statistics()
                phrase_stats = self._matcher.get_phrase_statistics()
            self._playback_ms = self._loop_start_ms
            self._last_tick = time.perf_counter()
            if self._matcher:
                self._matcher.reset()
            self._feedback.reset()
            if self._midi_player is not None:
                self._midi_player.seek(self._loop_start_ms)
            # Remap the clock without restarting the input stream.
            if self._audio_capture is not None and self._audio_enabled and self._playing:
                stream_ms = self._audio_capture.stream_time_ms()
                self._audio_capture.clock.set_segment(
                    self._loop_start_ms, stream_ms, self._tempo_factor
                )

            # Phrase mastery updates on every completed loop, not only while
            # guided-practice tempo automation is enabled. This makes repeated
            # Riff Repeater-style practice drive arrangement density directly.
            if phrase_stats:
                for phrase_id, phrase in phrase_stats.items():
                    self._adaptive.update_phrase(
                        phrase_id, float(phrase.get("accuracy", 0.0))
                    )
                if self._progress_tracker is not None and self._song_key:
                    self._progress_tracker.update_phrase_mastery(
                        self._song_key, self._adaptive.export()
                    )

            # Guided practice controls tempo independently from arrangement
            # density, as in Rocksmith's phrase mastery model.
            if self._guided_practice and stats:
                section_acc = stats.get("accuracy_percent", 0.0)
                self._guided_loop_count += 1
                if section_acc >= self._guided_target_accuracy:
                    self._guided_consecutive_successes += 1
                    if self._guided_consecutive_successes >= 3:
                        # Auto-increase tempo
                        new_factor = min(1.0, self._tempo_factor + 0.05)
                        self.set_tempo_factor(new_factor)
                        self._guided_consecutive_successes = 0
                else:
                    self._guided_consecutive_successes = 0
                    if section_acc < 60.0 and self._tempo_factor > 0.5:
                        self.set_tempo_factor(max(0.5, self._tempo_factor - 0.05))
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
                        phrase_stats = self._matcher.get_phrase_statistics()
                        for phrase_id, phrase in phrase_stats.items():
                            self._adaptive.update_phrase(
                                phrase_id, float(phrase.get("accuracy", 0.0))
                            )
                        self._progress_tracker.update_phrase_mastery(
                            self._song_key, self._adaptive.export()
                        )
                        weakest = self._matcher.get_weakest_sections()
                        self._is_new_best, self._recommendations = (
                            self._progress_tracker.record_detailed_result(
                                self._song_key, stats,
                                weakest, self._tempo_factor,
                                song_bpm=getattr(self._timeline.metadata, 'tempo', 0) or 0,
                            )
                        )
                        self._weakest_sections = weakest
                        self._song_completed = True
                        # Run the after-take analyzer to grade techniques
                        self._analyze_and_build_heatmap()
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
            self.toggle_string(1)
        elif event.key == pygame.K_F2:
            self.toggle_string(2)
        elif event.key == pygame.K_F3:
            self.toggle_string(3)
        elif event.key == pygame.K_F4:
            self.toggle_string(4)
        elif event.key == pygame.K_F5:
            self.toggle_string(5)
        elif event.key == pygame.K_F6:
            self.toggle_string(6)
        elif event.key == pygame.K_j:
            self._cycle_match_mode()
        elif event.key == pygame.K_l:
            self._loop_weakest_section()
        elif event.key == pygame.K_g:
            self._toggle_guided_practice()
        elif event.key == pygame.K_w:
            self._toggle_wait_mode()
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
        self._draw_pitch_curve_overlay(surface, layout)
        self._draw_hud(surface, layout)

        if self._show_help:
            self._draw_help_overlay(surface, layout)

    # -- Pure math helpers (testable without display) --
    def _layout(self, surface: pygame.Surface) -> _Layout:
        """Compute layout dimensions for current surface size."""
        w, h = surface.get_size()
        lane_area = h - LANE_TOP_MARGIN - LANE_BOTTOM_MARGIN
        lane_height = lane_area / self._num_strings
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
        for i in range(self._num_strings):
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
        bottom = int(LANE_TOP_MARGIN + self._num_strings * layout.lane_height)
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

            # Y position: string 1-N
            lane_y = LANE_TOP_MARGIN + (note.string - 1) * layout.lane_height
            y = lane_y + layout.lane_height / 2 - layout.note_h / 2

            # Color: feedback color if matched, dimmed if past the hit zone
            past_hit_zone = note.timestamp_ms < self._playback_ms
            base_color = string_color(note.string)
            if self._audio_enabled:
                color = self._feedback.get_note_color(
                    note, base_color, self._playback_ms, past_hit_zone,
                )
            else:
                color = dimmed(base_color) if past_hit_zone else base_color

            # Palm mute: dim the note color to indicate muted technique
            techniques = note.techniques
            tech_kinds = {t.kind for t in techniques}
            if "palm_mute" in tech_kinds:
                color = dimmed(color)

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

            # Technique icon: small letter on the right side of the note.
            # Reads the resolved techniques tuple (direction already set by matcher).
            if techniques:
                art_icons = {
                    "hammer_on": "H",
                    "pull_off": "P",
                    "bend": "b",
                    "vibrato": "~",
                    "slide": "/",
                    "palm_mute": "M",
                    "harmonic": "o",
                    "dead_note": "x",
                }
                # Pick the first technique with an icon (priority order)
                icon_char = ""
                for kind in ("harmonic", "bend", "slide", "hammer_on", "pull_off", "vibrato", "palm_mute", "dead_note"):
                    if kind in tech_kinds:
                        icon_char = art_icons.get(kind, "")
                        break
                if icon_char and rect.width > 24:
                    icon_font = _get_font("consolas", max(10, fret_font_size - 2))
                    icon_text = _font_cache.render(icon_font, icon_char, True, t.hud_accent)
                    ix = rect.right - icon_text.get_width() - 3
                    iy = rect.y + rect.height // 2 - icon_text.get_height() // 2
                    surface.blit(icon_text, (ix, iy))

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

    def _draw_pitch_curve_overlay(self, surface: pygame.Surface, layout: _Layout) -> None:
        """Overlay the detected f0 curve (cents) against the target for the
        just-played note. Drawn in the note's lane, fading over ~500ms.

        Active when a bend/vibrato/slide verdict exists for the most recent
        matched note. This is the visual flagship for lead guitar feedback.
        """
        if not self._recent_verdicts:
            return
        # Find the most recent bend/vibrato/slide verdict
        curve_verdict = None
        for v in reversed(self._recent_verdicts):
            if v.kind in ("bend", "vibrato", "slide"):
                curve_verdict = v
                break
        if curve_verdict is None:
            return
        # We don't have the event's f0_curve directly here without the event;
        # the verdict's metrics carry detected_cents / depth_cents. Render a
        # simple target-vs-detected bar as the Phase-1 visual proxy.
        t = get_theme()
        # Use the hit-zone x as the anchor
        hit_zone_x = layout.hit_zone_x
        y_center = layout.screen_h // 2
        bar_w = 120
        bar_h = 6
        bx = hit_zone_x - bar_w // 2
        by = y_center - bar_h // 2
        # Target line
        target = curve_verdict.metrics.get("target_cents") or curve_verdict.metrics.get("depth_cents") or 0.0
        detected = curve_verdict.metrics.get("detected_cents") or curve_verdict.metrics.get("depth_cents") or 0.0
        scale = max(1.0, abs(target), abs(detected))
        # Draw target (faint) and detected (bright) bars
        bx + bar_w  # compute but unused // 2 + int((target / scale) * (bar_w // 2))
        dx = bx + bar_w // 2 + int((detected / scale) * (bar_w // 2))
        pygame.draw.line(surface, (120, 120, 120), (bx, by), (bx + bar_w, by), 1)
        pygame.draw.line(surface, t.hud_accent, (bx + bar_w // 2, by - 4), (dx, by + 4), 3)

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

        # Guided practice HUD
        if self._guided_practice:
            gp_text = (
                f"GUIDED PRACTICE  |  Loop: {self._guided_loop_count}  "
                f"Streak: {self._guided_consecutive_successes}/3  "
                f"Tempo: {int(self._tempo_factor * 100)}%"
            )
            gp_surf = _font_cache.render(hint_font, gp_text, True, t.hud_accent)
            surface.blit(gp_surf, (w // 2 - gp_surf.get_width() // 2, 56))

        # Technique indicator (top-left, below title) + why-missed verdicts
        if self._last_technique:
            tech_text = self._last_technique.replace("_", " ").upper()
            tech_surf = _font_cache.render(hint_font, tech_text, True, t.hud_accent)
            surface.blit(tech_surf, (12, 36))
        if self._recent_verdicts:
            draw_why_missed(surface, self._recent_verdicts, hint_font, 12, 56)

        # Cents deviation bar: shows live pitch deviation from nearest semitone.
        # Visible whenever audio capture has a confident pitch — the bar oscillates
        # during vibrato and stays near center for steady notes.
        if self._audio_capture is not None:
            freq, conf = self._audio_capture.get_tuner_data()
            if freq > 0 and conf > 0.5:
                from pickhero.audio.note_utils import freq_to_midi, midi_to_freq
                import math
                midi = freq_to_midi(freq)
                target = midi_to_freq(midi)
                if target > 0:
                    cents = 1200 * math.log2(freq / target)
                    bar_w = 100
                    bar_h = 6
                    bar_x = 12
                    bar_y = 54
                    pygame.draw.rect(surface, t.signal_cold, (bar_x, bar_y, bar_w, bar_h))
                    center_x = bar_x + bar_w // 2
                    deviation = max(-25, min(25, int(cents)))
                    fill_w = abs(deviation) * 2
                    if deviation >= 0:
                        fill_color = t.tuner_in_tune
                        pygame.draw.rect(surface, fill_color,
                                       (center_x, bar_y, fill_w, bar_h))
                    else:
                        fill_color = t.tuner_close
                        pygame.draw.rect(surface, fill_color,
                                       (center_x - fill_w, bar_y, fill_w, bar_h))
                    pygame.draw.line(surface, t.hud_text,
                                    (center_x, bar_y - 1), (center_x, bar_y + bar_h + 1), 1)

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
                # XRun counter: show if any audio dropouts occurred
                if self._audio_capture is not None:
                    xruns = self._audio_capture.get_xrun_count()
                    if xruns > 0:
                        xrun_text = f"XRuns: {xruns}"
                        xrun_surf = _font_cache.render(hint_font, xrun_text, True, t.feedback_miss)
                        surface.blit(xrun_surf, (w - xrun_surf.get_width() - 12, stats_bottom_y + 54))
                    self._draw_input_health(surface, hint_font, w, stats_bottom_y + 72)
        elif self._audio_capture is not None:
            # Audio off but capture exists — still show meter and tuner
            self._draw_signal_meter(surface, hint_font, w, stats_bottom_y)
            self._draw_tuner(surface, hint_font, w, stats_bottom_y + 18)
            # XRun counter also shown when audio is off
            xruns = self._audio_capture.get_xrun_count()
            if xruns > 0:
                xrun_text = f"XRuns: {xruns}"
                xrun_surf = _font_cache.render(hint_font, xrun_text, True, t.feedback_miss)
                surface.blit(xrun_surf, (w - xrun_surf.get_width() - 12, stats_bottom_y + 36))
            self._draw_input_health(surface, hint_font, w, stats_bottom_y + 54)

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
        timing_state = f"|  J: {self._match_mode.value}  "
        hint = (
            f"{state}  |  SPACE: play/pause  |  LEFT/RIGHT: seek  "
            f"|  HOME: restart  |  PgDn/PgUp: tempo  |  X/C: gate"
            f"|  A: audio {audio_state}  "
            f"{backing_state}"
            f"{wait_state}"
            f"{timing_state}"
            f"|  I/O: loop {loop_state}  |  P: toggle  |  G: guided  |  ESC: menu"
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

        # Match mode HUD (shown when not the default ARCADE)
        if self._match_mode != MatchMode.ARCADE:
            mode_text = f"Mode: {self._match_mode.value}"
            mode_surf = _font_cache.render(hint_font, mode_text, True, t.hud_accent)
            surface.blit(mode_surf, (12, info_y))


    def _draw_input_health(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        width: int,
        y: int,
    ) -> None:
        if self._audio_capture is None:
            return
        health = self._audio_capture.get_input_health()
        status = health.get("status")
        if status == "clipping":
            clipped = float(health.get("clipped_fraction", 0.0)) * 100.0
            text = f"INPUT CLIPPING ({clipped:.1f}%) — lower gain"
        elif status == "dc_offset":
            dc = float(health.get("dc_offset", 0.0))
            text = f"INPUT DC OFFSET ({dc:+.3f})"
        elif status == "overrun":
            dropped = int(health.get("dropped_samples", 0))
            backlog = int(health.get("worker_backlog", 0))
            text = f"AUDIO WORKER OVERRUN ({dropped} samples, queue {backlog})"
        else:
            return
        t = get_theme()
        warning = _font_cache.render(font, text, True, t.feedback_miss)
        surface.blit(warning, (width - warning.get_width() - 12, y))

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
        state = CompletionState(
            audio_enabled=self._audio_enabled,
            matcher=self._matcher,
            is_new_best=self._is_new_best,
            recommendations=self._recommendations,
            weakest_sections=getattr(self, "_weakest_sections", []),
            timing_judge=self._timing_judge,
            timing_summary=self._timing_summary,
            timing_worst_measures=self._timing_worst_measures,
            technique_heatmap=getattr(self, "_technique_heatmap", {}),
            drill_recommendation=getattr(self, "_drill_recommendation", None),
        )
        draw_completion_overlay(surface, layout, state)

    def _analyze_and_build_heatmap(self) -> None:
        """Run the after-take analyzer and build the technique heatmap + drill."""
        if self._matcher is None:
            self._technique_heatmap = {}
            self._drill_recommendation = None
            return
        # Drain any pending PerformanceEvents from the audio capture
        if self._audio_capture is not None:
            self._audio_capture.get_events()
        # Run the analyzer over collected matched pairs
        tone_profile = None
        if self._config is not None:
            tone_profile = self._config.get_active_tone_profile()
        graded = self._matcher.analyze_performance(tone_profile)
        # Collect recent failed verdicts for the "why did I miss" HUD display
        all_verdicts: list = []
        for ev in graded:
            all_verdicts.extend(ev.verdicts)
        # Patch 6d: offline polyphonic pass (unison bends, pinch verification).
        # Runs only when the preset armed it AND a take was recorded.
        if (
            self._config is not None
            and getattr(self._config, "offline_deep_analysis", False)
            and self._audio_capture is not None
        ):
            raw = self._audio_capture.stop_take_recording()
            if raw is not None and len(raw) > 0:
                sr = self._audio_capture.detector.sample_rate
                offline_verdicts = self._matcher.analyze_performance_offline(
                    raw, sr, tone_profile,
                )
                all_verdicts.extend(offline_verdicts)
        self._recent_verdicts = [v for v in all_verdicts if v.grade in ("missed", "weak")][-3:]
        # Build the heatmap: kind -> {accuracy, count}
        heatmap: dict[str, dict[str, float]] = {}
        for v in all_verdicts:
            entry = heatmap.setdefault(v.kind, {"accuracy": 0.0, "count": 0, "_correct": 0})
            entry["count"] += 1
            if v.grade in ("good", "ok"):
                entry["_correct"] += 1
        for kind, entry in heatmap.items():
            cnt = entry["count"]
            entry["accuracy"] = (entry["_correct"] / cnt * 100.0) if cnt > 0 else 0.0
            del entry["_correct"]
        self._technique_heatmap = heatmap
        # Build the drill recommendation
        from pickhero.recommendations import recommend_drill
        self._drill_recommendation = recommend_drill(heatmap, self._weakest_sections)

        # Dump the debug match log to stderr when PICKHERO_DEBUG_MATCH=1.
        if self._matcher is not None and getattr(self._matcher, "_match_log", None):
            import sys
            print("=== MATCH LOG ===", file=sys.stderr)
            for line in self._matcher.get_match_log():
                print(line, file=sys.stderr)
            print("=== END MATCH LOG ===", file=sys.stderr)

    def get_timing_stats(self):
        """Return the matcher's current TimingStats, or None if no matcher."""
        if self._matcher is None:
            return None
        return self._matcher.get_timing_stats()

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
        stats = self._timing_summary
        if stats is None:
            return
        observations = (
            self._matcher.get_timing_observations()
            if self._matcher is not None
            else []
        )
        draw_timing_summary(
            surface,
            layout,
            stats,
            self._timing_worst_measures,
            observations,
        )

    def _draw_help_overlay(self, surface: pygame.Surface, layout: _Layout) -> None:
        """Draw a help overlay explaining the track, note colors, and controls."""
        draw_help_overlay(surface, layout)

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

    def toggle_string(self, string: int) -> None:
        """Toggle a string lane on/off."""
        if not 1 <= string <= self._num_strings:
            return
        idx = string - 1
        self._active_strings[idx] = not self._active_strings[idx]
        # Don't allow all strings to be off
        if not any(self._active_strings):
            self._active_strings[idx] = True
            return
        # Persist as many entries as the global config supports, padding if needed.
        padded = list(self._active_strings)
        while len(padded) < len(self._config.active_strings):
            padded.append(True)
        self._config.active_strings = padded[:len(self._config.active_strings)]
        self._config.save()
        self._reset_matcher_for_filter()

    def _filter_status_text(self) -> str | None:
        """Return a status text snippet if difficulty filters are active."""
        parts: list[str] = []
        if self._max_fret < 24:
            parts.append(f"Fret: 0-{self._max_fret}")
        if not all(self._active_strings):
            strs = " ".join(
                str(i + 1) if on else "_"
                for i, on in enumerate(self._active_strings)
            )
            parts.append(f"Strings: {strs}")
        if self._adaptive.enabled:
            levels = [state.level for state in self._adaptive.phrases.values()]
            if levels:
                low, high = min(levels), max(levels)
                level_text = str(low) if low == high else f"{low}-{high}"
                parts.append(f"Dynamic difficulty: {level_text}/5")
        return "  |  ".join(parts) if parts else None

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
        if self._adaptive.enabled:
            levels = [state.level for state in self._adaptive.phrases.values()]
            if levels:
                low, high = min(levels), max(levels)
                level_text = str(low) if low == high else f"{low}-{high}"
                parts.append(f"Dynamic difficulty: {level_text}/5")
        return "  |  ".join(parts) if parts else None

    # -- Theme --

    def _cycle_theme(self) -> None:
        """Toggle between dark and light theme."""
        name = cycle_theme()
        self._config.theme = name
        self._config.save()

    # -- Match mode (replaces chord/timing/pitch toggles) --

    _MODE_CYCLE = [MatchMode.ARCADE, MatchMode.PRACTICE, MatchMode.JUDGE]

    def _cycle_match_mode(self) -> None:
        """Cycle ARCADE → PRACTICE → JUDGE → ARCADE.

        Replaces the separate chord_partial_credit, timing_judge, and
        pitch_strict toggles with a single strictness profile.
        """
        idx = self._MODE_CYCLE.index(self._match_mode)
        self._match_mode = self._MODE_CYCLE[(idx + 1) % len(self._MODE_CYCLE)]
        # Sync derived booleans and persist
        self._chord_partial_credit = self._match_mode != MatchMode.ARCADE
        self._timing_judge = self._match_mode == MatchMode.JUDGE
        self._pitch_strict = self._match_mode == MatchMode.JUDGE
        self._config.match_mode = self._match_mode.value
        self._config.save()
        if self._match_mode == MatchMode.JUDGE and self._timing_overlay is None:
            self._timing_overlay = TimingOverlay()
        if self._match_mode != MatchMode.JUDGE:
            self._last_obs_count = 0
        if self._matcher:
            self._matcher.match_mode = self._match_mode
        if self._audio_capture is not None:
            self._audio_capture.set_match_mode(self._match_mode)

    def _toggle_guided_practice(self) -> None:
        """Toggle guided practice mode on/off."""
        if not self._song_completed and not self._weakest_sections:
            # Need a completed run to know the weakest section
            return
        self._guided_practice = not self._guided_practice
        if self._guided_practice:
            self._start_guided_practice()

    def _start_guided_practice(self) -> None:
        """Start guided practice: loop weakest section at reduced tempo."""
        weak = getattr(self, "_weakest_sections", [])
        if not weak:
            return
        section = weak[0]
        start_measure, end_measure = section[0], section[1]
        measures = self._timeline.measures
        if not measures or start_measure >= len(measures):
            return
        # Set loop markers
        start_ms = measures[start_measure].start_ms
        end_idx = min(end_measure + 1, len(measures) - 1)
        end_ms = measures[end_idx].end_ms if end_idx < len(measures) else self._timeline.duration_ms
        self._loop_start_ms = start_ms
        self._loop_end_ms = end_ms
        self._loop_enabled = True
        self._song_completed = False
        # Set tempo to 50% or cliff_bpm - 10 if available
        cliff = None
        if self._progress_tracker is not None and self._song_key:
            record = self._progress_tracker.get_best(self._song_key)
            if record is not None:
                cliff = record.cliff_bpm
        if cliff and self._timeline.metadata.tempo > 0:
            target_factor = max(0.5, min(1.0, (cliff - 10) / self._timeline.metadata.tempo))
        else:
            target_factor = 0.5
        self.set_tempo_factor(target_factor)
        self._guided_consecutive_successes = 0
        self._guided_loop_count = 0
        self.seek(start_ms)

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
        lane_bottom = int(LANE_TOP_MARGIN + self._num_strings * layout.lane_height)
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
            stream_ms = self._audio_capture.stream_time_ms()
            self._audio_capture.clock.set_segment(
                self._playback_ms, stream_ms, self._tempo_factor
            )
            # Patch 6b/d: arm raw-take recording for offline polyphonic analysis
            # when the preset requests it.
            if getattr(self._config, "offline_deep_analysis", False):
                self._audio_capture.start_take_recording()
            chord_fft = getattr(self._config, 'preset_flags', {}).get('chord_fft_size', 8192)
            verifier = CompositeVerifier(
                sample_rate=self._audio_capture.detector.sample_rate,
                fft_size=chord_fft,
            )
            self._matcher = NoteMatcher(
                self._timeline,
                timing_window_ms=self._config.timing_window_ms,
                audio_offset_ms=self._config.get_audio_latency_offset(),
                chord_threshold_ms=self._config.chord_threshold_ms,
                note_filter=self._note_passes_filter,
                mode=self._match_mode,
                verifier=verifier,
            )
            if self._timing_judge and self._timing_overlay is None:
                self._timing_overlay = TimingOverlay()
            self._feedback.reset()
        except Exception as e:
            if self._on_error:
                self._on_error(f"Audio start failed: {e}")
            self._audio_enabled = False

    def _start_capture_only(self) -> None:
        """Start audio capture for signal monitoring (no matcher)."""
        try:
            from pickhero.audio.input import AudioCapture
            if self._audio_capture is None:
                self._audio_capture = AudioCapture(self._config)
            self._audio_capture.start()
        except Exception as e:
            if self._on_error:
                self._on_error(f"Audio capture start failed: {e}")
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
            if self._on_error:
                self._on_error(f"MIDI player init failed: {e}")

    def _toggle_backing(self) -> None:
        """Toggle backing track mute on/off."""
        if self._midi_player is None:
            return
        self._backing_muted = not self._backing_muted
        self._midi_player.set_muted(self._backing_muted)

    def _toggle_wait_mode(self) -> None:
        """Toggle wait mode on/off."""
        self._wait_mode = not self._wait_mode
        self._config.wait_mode = self._wait_mode
        self._config.save()
