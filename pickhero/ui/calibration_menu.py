"""Guitar calibration screen.

Guides the user through calibrating each open string. Measures noise floor
then collects pitch samples for strings 6-1 (low E first). Results are
stored in Config.calibration for octave-correction in the detector.
"""

from __future__ import annotations

import statistics
import time

import pygame

from pickhero.audio.input import AudioCapture
from pickhero.audio.note_utils import freq_to_midi, midi_to_name, STANDARD_TUNING
from pickhero.config import Config, StringCalibration
from pickhero.ui.colors import get_theme


# String order for calibration: 6 (low E) → 1 (high E)
STRING_ORDER = [6, 5, 4, 3, 2, 1]

# Collection time per string (seconds)
COLLECT_SECONDS = 3.0
# Noise floor measurement time (seconds)
NOISE_SECONDS = 2.0

# Minimum confidence for a sample to count
MIN_CONFIDENCE = 0.5

# Signal must exceed noise floor by this many dB to trigger start
SPIKE_THRESHOLD_DB = 15.0
# Cooldown after entering waiting state before accepting signal (seconds)
WAITING_COOLDOWN = 0.6


def _get_font(name: str, size: int) -> pygame.font.Font:
    for family in (name, "Courier New", "monospace"):
        font = pygame.font.SysFont(family, size)
        if font:
            return font
    return pygame.font.Font(None, size)


class CalibrationMenuScreen:
    """Step-by-step guitar calibration UI."""

    def __init__(self, config: Config):
        self._config = config
        self._capture: AudioCapture | None = None
        self._state = "intro"  # intro, noise, waiting, listen, confirm, done
        self._string_idx = 0  # index into STRING_ORDER
        self._start_time: float = 0.0
        self._waiting_start: float = 0.0

        # Noise floor
        self._noise_samples: list[float] = []
        self._noise_floor_db: float = -80.0

        # Per-string collection
        self._freq_samples: list[tuple[float, float]] = []  # (freq, confidence)

        # Results per string number
        self._results: dict[int, StringCalibration] = {}

    @property
    def _current_string(self) -> int:
        """String number currently being calibrated."""
        return STRING_ORDER[self._string_idx]

    @property
    def _current_note_name(self) -> str:
        """Expected open-string note name."""
        midi = STANDARD_TUNING.get(self._current_string, 0)
        return midi_to_name(midi)

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Process input. Returns 'back' on cancel, 'complete' when done."""
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            self._stop_capture()
            return "back"

        if self._state == "intro":
            if event.key == pygame.K_RETURN:
                self._start_capture()
                self._begin_noise()
            return None

        if self._state == "confirm":
            if event.key == pygame.K_RETURN:
                # Accept and move to next string or done
                self._string_idx += 1
                if self._string_idx >= len(STRING_ORDER):
                    self._stop_capture()
                    self._state = "done"
                else:
                    self._begin_waiting()
            elif event.key == pygame.K_r:
                # Retry current string
                self._begin_waiting()
            return None

        if self._state == "done":
            if event.key == pygame.K_RETURN:
                self._stop_capture()
                return "complete"
            return None

        return None

    def update(self) -> None:
        """Called each frame to advance calibration timers."""
        if self._capture is None:
            return

        elapsed = time.perf_counter() - self._start_time

        if self._state == "noise":
            # Collect noise floor samples
            db = float(self._capture.get_signal_db())
            self._noise_samples.append(db)
            if elapsed >= NOISE_SECONDS:
                self._noise_floor_db = statistics.median(self._noise_samples) if self._noise_samples else -80.0
                self._begin_waiting()

        elif self._state == "waiting":
            # Wait for a strong pick — require cooldown + signal spike above noise floor
            wait_elapsed = time.perf_counter() - self._waiting_start
            if wait_elapsed < WAITING_COOLDOWN:
                return  # let previous string's sound fade
            db = float(self._capture.get_signal_db())
            freq, conf = self._capture.get_tuner_data()
            spike = db > self._noise_floor_db + SPIKE_THRESHOLD_DB
            if spike and freq > 0 and conf > MIN_CONFIDENCE:
                self._begin_listen()

        elif self._state == "listen":
            # Collect pitch samples
            freq, conf = self._capture.get_tuner_data()
            if freq > 0 and conf > MIN_CONFIDENCE:
                self._freq_samples.append((float(freq), float(conf)))
            if elapsed >= COLLECT_SECONDS:
                self._finish_listen()

    def render(self, surface: pygame.Surface) -> None:
        """Draw the calibration screen."""
        t = get_theme()
        surface.fill(t.menu_bg)
        w, h = surface.get_size()

        title_font = _get_font("arial", 36)
        body_font = _get_font("arial", 22)
        hint_font = _get_font("arial", 16)
        big_font = _get_font("arial", 48)

        # Title
        title_surf = title_font.render("Guitar Calibration", True, t.hud_accent)
        surface.blit(title_surf, (w // 2 - title_surf.get_width() // 2, 24))

        cy = h // 2 - 60

        if self._state == "intro":
            self._render_centered(surface, body_font,
                                  "Calibrate your guitar for more accurate detection.", t.hud_text, cy)
            self._render_centered(surface, body_font,
                                  "Play each open string when prompted.", t.hud_text, cy + 32)
            self._render_centered(surface, hint_font,
                                  "ENTER: begin  |  ESC: cancel", t.hud_text, h - 36)

        elif self._state == "noise":
            self._render_centered(surface, body_font,
                                  "Stay quiet for 2 seconds...", t.hud_accent, cy)
            self._draw_progress_bar(surface, w, cy + 50,
                                    (time.perf_counter() - self._start_time) / NOISE_SECONDS)
            db = self._capture.get_signal_db() if self._capture else -80
            self._render_centered(surface, hint_font,
                                  f"Noise level: {int(db)} dB", t.hud_text, cy + 80)

        elif self._state == "waiting":
            string_num = self._current_string
            note_name = self._current_note_name
            step = f"String {string_num} — play {note_name} open"
            self._render_centered(surface, big_font, step, t.hud_accent, cy - 20)
            self._render_centered(surface, body_font,
                                  "Pick the string when ready...", t.hud_text, cy + 50)
            # Show signal level vs threshold
            if self._capture:
                db = float(self._capture.get_signal_db())
                trigger_db = self._noise_floor_db + SPIKE_THRESHOLD_DB
                wait_elapsed = time.perf_counter() - self._waiting_start
                if wait_elapsed < WAITING_COOLDOWN:
                    status = "Waiting for silence..."
                    color = t.hud_text
                elif db > trigger_db:
                    status = f"Signal: {int(db)} dB  (trigger: {int(trigger_db)} dB)"
                    color = t.tuner_in_tune
                else:
                    status = f"Signal: {int(db)} dB  (need > {int(trigger_db)} dB)"
                    color = t.hud_text
                self._render_centered(surface, hint_font, status, color, cy + 85)
            self._render_centered(surface, hint_font,
                                  "ESC: cancel", t.hud_text, h - 36)

        elif self._state == "listen":
            string_num = self._current_string
            note_name = self._current_note_name
            step = f"String {string_num} — play {note_name} open"
            self._render_centered(surface, big_font, step, t.hud_accent, cy - 20)

            # Progress bar
            elapsed = time.perf_counter() - self._start_time
            self._draw_progress_bar(surface, w, cy + 50, elapsed / COLLECT_SECONDS)

            # Live frequency display
            if self._capture:
                freq, conf = self._capture.get_tuner_data()
                if freq > 0 and conf > MIN_CONFIDENCE:
                    detected = midi_to_name(freq_to_midi(freq))
                    info = f"{detected}  {freq:.1f} Hz  (conf: {conf:.2f})"
                    color = t.tuner_in_tune if conf > 0.8 else t.tuner_close
                else:
                    info = "Listening..."
                    color = t.hud_text
                self._render_centered(surface, body_font, info, color, cy + 80)

            # Sample count
            count = len(self._freq_samples)
            self._render_centered(surface, hint_font,
                                  f"{count} samples collected", t.hud_text, cy + 115)

        elif self._state == "confirm":
            string_num = self._current_string
            cal = self._results.get(string_num)
            if cal:
                note_name = midi_to_name(cal.midi_note)
                self._render_centered(surface, big_font,
                                      f"String {string_num}: {note_name}", t.hud_accent, cy - 20)
                self._render_centered(surface, body_font,
                                      f"{cal.frequency:.1f} Hz  (MIDI {cal.midi_note})", t.hud_text, cy + 40)
            else:
                self._render_centered(surface, big_font,
                                      f"String {string_num}: No data!", t.feedback_miss, cy - 20)
                self._render_centered(surface, body_font,
                                      "Try again — play louder or check your cable.", t.hud_text, cy + 40)
            self._render_centered(surface, hint_font,
                                  "ENTER: accept  |  R: retry  |  ESC: cancel", t.hud_text, h - 36)

        elif self._state == "done":
            self._render_centered(surface, big_font,
                                  "Calibration Complete!", t.hud_accent, cy - 40)
            # Summary
            summary_y = cy + 20
            for s in STRING_ORDER:
                cal = self._results.get(s)
                if cal:
                    name = midi_to_name(cal.midi_note)
                    line = f"String {s}: {name}  ({cal.frequency:.1f} Hz)"
                    color = t.tuner_in_tune
                else:
                    line = f"String {s}: not calibrated"
                    color = t.feedback_miss
                self._render_centered(surface, body_font, line, color, summary_y)
                summary_y += 30
            self._render_centered(surface, hint_font,
                                  "ENTER: save and return  |  ESC: discard", t.hud_text, h - 36)

    def get_results(self) -> dict[int, StringCalibration]:
        """Return calibration results (string_num → StringCalibration)."""
        return dict(self._results)

    # -- Internal helpers --

    def _render_centered(self, surface: pygame.Surface, font: pygame.font.Font,
                         text: str, color: tuple, y: int) -> None:
        surf = font.render(text, True, color)
        w = surface.get_width()
        surface.blit(surf, (w // 2 - surf.get_width() // 2, y))

    def _draw_progress_bar(self, surface: pygame.Surface, screen_w: int,
                           y: int, fraction: float) -> None:
        t = get_theme()
        bar_w = 300
        bar_h = 12
        bar_x = screen_w // 2 - bar_w // 2
        fraction = max(0.0, min(1.0, fraction))

        pygame.draw.rect(surface, t.signal_cold, (bar_x, y, bar_w, bar_h))
        fill_w = int(fraction * bar_w)
        if fill_w > 0:
            pygame.draw.rect(surface, t.hud_accent, (bar_x, y, fill_w, bar_h))
        pygame.draw.rect(surface, t.hud_text, (bar_x, y, bar_w, bar_h), 1)

    def _start_capture(self) -> None:
        """Start audio capture for calibration."""
        if self._capture is None:
            self._capture = AudioCapture(self._config)
        self._capture.start()

    def _stop_capture(self) -> None:
        """Stop audio capture."""
        if self._capture is not None:
            self._capture.stop()
            self._capture = None

    def _begin_noise(self) -> None:
        """Enter noise measurement state."""
        self._state = "noise"
        self._noise_samples = []
        self._start_time = time.perf_counter()

    def _begin_waiting(self) -> None:
        """Enter waiting state — show prompt, start collecting on first pick."""
        self._state = "waiting"
        self._freq_samples = []
        self._waiting_start = time.perf_counter()

    def _begin_listen(self) -> None:
        """Enter listening state for current string (timer starts now)."""
        self._state = "listen"
        self._freq_samples = []
        self._start_time = time.perf_counter()

    def _finish_listen(self) -> None:
        """Process collected samples and move to confirm state."""
        string_num = self._current_string

        if self._freq_samples:
            freqs = [f for f, c in self._freq_samples]
            median_freq = statistics.median(freqs)
            midi_note = freq_to_midi(median_freq)
            self._results[string_num] = StringCalibration(
                midi_note=midi_note,
                frequency=round(median_freq, 2),
                noise_floor_db=round(self._noise_floor_db, 1),
            )
        # If no samples were collected, _results won't have an entry for this string

        self._state = "confirm"
