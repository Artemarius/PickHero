"""Guitar calibration screen.

Guides the user through calibrating each open string. Measures noise floor
then collects pitch samples for strings 6-1 (low E first). Results are
stored in Config.calibration for octave-correction in the detector.

Latency calibration is a manual-offset nudge UI backed by the Timing Judge's
early/late distribution. The previous flash-based measurement mixed two
unrelated clocks (session-relative onset ms vs. absolute perf_counter) and
measured reaction time + display + input latency, not audio latency alone.
"""

from __future__ import annotations

import statistics
import threading
import time
from typing import Callable, TYPE_CHECKING

import pygame

from pickhero.audio.input import AudioCapture
from pickhero.audio.note_utils import freq_to_midi, midi_to_name, STANDARD_TUNING
from pickhero.config import Config, StringCalibration
from pickhero.ui.colors import get_theme

if TYPE_CHECKING:
    from pickhero.timing import TimingStats

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

# Latency nudge step (ms) for the manual offset adjustment UI.
LATENCY_NUDGE_MS = 5.0


def _get_font(name: str, size: int) -> pygame.font.Font:
    for family in (name, "Courier New", "monospace"):
        font = pygame.font.SysFont(family, size)
        if font:
            return font
    return pygame.font.Font(None, size)

class CalibrationMenuScreen:
    """Step-by-step guitar/bass calibration UI."""

    def __init__(self, config: Config, num_strings: int = 6,
                 tuning: dict[int, int] | None = None,
                 timing_stats_provider: Callable[[], "TimingStats | None"] | None = None):
        self._config = config
        self._num_strings = max(1, min(12, num_strings))
        self._tuning = tuning if tuning else dict(STANDARD_TUNING)
        self._capture: AudioCapture | None = None
        # States: intro, noise, waiting, listen, confirm, done, latency_nudge
        self._state = "intro"
        self._string_idx = 0  # index into _string_order
        self._start_time: float = 0.0
        self._waiting_start: float = 0.0
        self._noise_samples: list[float] = []
        self._noise_floor_db: float = -80.0

        # Per-string collection
        self._freq_samples: list[tuple[float, float]] = []  # (freq, confidence)

        # Results per string number
        self._results: dict[int, StringCalibration] = {}

        # Latency nudge UI: optional provider of TimingStats from the last run,
        # used to render a live early/late histogram. None when no run exists.
        self._timing_stats_provider = timing_stats_provider
        self._latency_thread: threading.Thread | None = None
        self._latency_result = None
        self._latency_error: str = ""

    @property
    def _string_order(self) -> list[int]:
        """String numbers to calibrate, lowest (thickest) to highest."""
        return list(range(self._num_strings, 0, -1))

    @property
    def _current_string(self) -> int:
        """String number currently being calibrated."""
        return self._string_order[self._string_idx]

    @property
    def _current_note_name(self) -> str:
        """Expected open-string note name."""
        midi = self._tuning.get(self._current_string, 0)
        if midi <= 0:
            return "?"
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
            elif event.key == pygame.K_y:
                self._state = "latency_nudge"
            elif event.key == pygame.K_LEFT and self._num_strings > 1:
                self._num_strings = max(1, self._num_strings - 1)
            elif event.key == pygame.K_RIGHT and self._num_strings < 12:
                self._num_strings = min(12, self._num_strings + 1)
            return None

        if self._state == "latency_nudge":
            # LEFT/RIGHT nudge the audio latency offset by ±LATENCY_NUDGE_MS.
            # ENTER/ESC returns to the intro screen (offset is saved on change).
            if event.key == pygame.K_LEFT:
                self._nudge_latency(-LATENCY_NUDGE_MS)
            elif event.key == pygame.K_RIGHT:
                self._nudge_latency(+LATENCY_NUDGE_MS)
            elif event.key == pygame.K_a:
                self._begin_auto_latency()
            elif event.key == pygame.K_RETURN:
                self._state = "intro"
            return None

        if self._state == "confirm":
            if event.key == pygame.K_RETURN:
                # Accept and move to next string or done
                self._string_idx += 1
                if self._string_idx >= len(self._string_order):
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
                                  f"Strings: {self._num_strings}  (LEFT / RIGHT to change)", t.hud_text, cy + 32)
            self._render_centered(surface, body_font,
                                  "Play each open string when prompted.", t.hud_text, cy + 64)
            self._render_centered(surface, hint_font,
                                  "ENTER: begin  |  Y: Latency Offset (manual)  |  ESC: cancel", t.hud_text, h - 36)

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

        elif self._state == "latency_nudge":
            # Manual offset nudge UI, replacing the broken flash-based measurement.
            # The flash mixed session-relative onset ms with absolute perf_counter ms
            # and measured reaction+display+input latency, not audio latency.
            self._render_centered(surface, big_font,
                                  "Latency Offset (manual)", t.hud_accent, cy - 60)
            current_offset = self._config.get_audio_latency_offset()
            sign = "+" if current_offset >= 0 else ""
            self._render_centered(surface, body_font,
                                  f"Audio offset: {sign}{current_offset:.0f} ms",
                                  t.hud_accent, cy - 10)
            self._render_centered(surface, hint_font,
                                  "LEFT: earlier  |  RIGHT: later  (±5 ms per press)",
                                  t.hud_text, cy + 24)
            self._render_centered(surface, hint_font,
                                  "A: automatic loopback probe  |  ENTER: back",
                                  t.hud_text, cy + 48)
            auto_y = cy + 72
            if self._latency_thread is not None and self._latency_thread.is_alive():
                self._render_centered(surface, hint_font,
                                      "Measuring... keep the output-to-input path connected",
                                      t.hud_accent, auto_y)
            elif self._latency_result is not None:
                result = self._latency_result
                state = "applied" if result.accepted else "rejected (low confidence)"
                self._render_centered(
                    surface, hint_font,
                    f"Automatic: {result.delay_ms:.1f} ms, confidence {result.confidence:.0%} — {state}",
                    t.tuner_in_tune if result.accepted else t.feedback_miss, auto_y,
                )
            elif self._latency_error:
                self._render_centered(surface, hint_font,
                                      f"Automatic calibration failed: {self._latency_error}",
                                      t.feedback_miss, auto_y)
            else:
                self._render_centered(surface, hint_font,
                                      "Best result: physical loopback cable; acoustic probe is less reliable",
                                      t.hud_text, auto_y)

            # Live early/late histogram from the Timing Judge, when available.
            stats = self._timing_stats_provider() if self._timing_stats_provider else None
            hist_y = cy + 104
            if stats is not None and stats.count > 0:
                self._render_centered(surface, hint_font,
                                      f"Early/late distribution ({stats.count} obs):",
                                      t.hud_text, hist_y)
                self._draw_histogram(surface, w, hist_y + 24, stats)
            else:
                self._render_centered(surface, hint_font,
                                      "Play a section to see early/late distribution",
                                      t.hud_text, hist_y)

            self._render_centered(surface, hint_font,
                                  "ENTER: back  |  ESC: cancel", t.hud_text, h - 36)

        elif self._state == "done":
            self._render_centered(surface, big_font,
                                  "Calibration Complete!", t.hud_accent, cy - 40)
            # Summary
            summary_y = cy + 20
            for s in self._string_order:
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

    # -- Tone profile template recording (Phase 1) --

    def record_tone_template(
        self, template_type: str, audio_chunks: list, sample_rate: int = 44100,
    ) -> dict:
        """Record a tone-profile template for the given technique type.

        Captures ~2s of audio (passed as pre-collected chunks) and computes
        ``decay_halflife_ms`` (from the RMS envelope) and ``centroid_hz`` (mean
        spectral centroid over onset frames). Returns a template dict suitable
        for storage in a :class:`~pickhero.config.ToneProfile`.

        Args:
            template_type: one of ``normal``, ``palm_mute``, ``dead_note``,
                ``harmonic``, ``bend``, ``vibrato``.
            audio_chunks: list of float32 numpy arrays (mono audio frames).
            sample_rate: capture sample rate.

        Returns:
            ``{"decay_halflife_ms": float, "centroid_hz": float,
            "harmonic_strength": float}``.
        """
        import numpy as np
        if not audio_chunks:
            return {"decay_halflife_ms": 0.0, "centroid_hz": 0.0, "harmonic_strength": 0.0}
        # Compute RMS envelope per chunk
        rms_values = []
        centroids = []
        for chunk in audio_chunks:
            arr = np.asarray(chunk, dtype=np.float32)
            if len(arr) < 2:
                continue
            rms = float(np.sqrt(np.mean(arr ** 2)))
            rms_values.append(rms)
            # Spectral centroid
            window = np.hanning(len(arr))
            spectrum = np.abs(np.fft.rfft(arr * window))
            freqs = np.fft.rfftfreq(len(arr), 1.0 / sample_rate)
            total = float(np.sum(spectrum))
            if total > 0:
                centroids.append(float(np.sum(freqs * spectrum) / total))
        halflife = 0.0
        if rms_values:
            peak = max(rms_values)
            if peak > 0:
                half = peak / 2.0
                chunk_ms = (len(audio_chunks[0]) / sample_rate * 1000.0) if audio_chunks else 11.6
                for i, r in enumerate(rms_values):
                    if r <= half:
                        halflife = i * chunk_ms
                        break
                if halflife == 0.0:
                    halflife = len(rms_values) * chunk_ms
        centroid = statistics.fmean(centroids) if centroids else 0.0
        return {
            "decay_halflife_ms": round(halflife, 1),
            "centroid_hz": round(centroid, 1),
            "harmonic_strength": 0.0,
        }

    def save_tone_profile(self, guitar: str, pickup: str, gain: str,
                          templates: dict[str, dict]) -> None:
        """Build a ToneProfile from recorded templates and persist it to Config.

        Sets ``active_tone_profile`` to the new profile's name.
        """
        from pickhero.config import ToneProfile
        profile = ToneProfile(
            guitar=guitar, pickup=pickup, gain=gain, templates=templates,
        )
        self._config.add_tone_profile(profile)
        self._config.active_tone_profile = profile.name
        self._config.save()

    # -- Latency measurement --

    def _begin_auto_latency(self) -> None:
        """Run a non-blocking output-to-input loopback measurement."""
        if self._latency_thread is not None and self._latency_thread.is_alive():
            return
        self._latency_result = None
        self._latency_error = ""

        def worker() -> None:
            try:
                from pickhero.audio.latency_calibrator import measure_roundtrip_latency
                ac = self._config.audio
                input_device = ac.device_name or ac.device_index
                result = measure_roundtrip_latency(
                    sample_rate=ac.sample_rate,
                    input_device=input_device,
                    input_channel=ac.input_channel,
                )
                self._latency_result = result
                self._config.set_audio_latency_measurement(
                    result.to_dict(), apply=result.accepted
                )
                self._config.save()
            except Exception as exc:
                self._latency_error = str(exc)

        self._latency_thread = threading.Thread(target=worker, daemon=True)
        self._latency_thread.start()

    # -- Latency offset nudge (manual) --

    def _nudge_latency(self, delta_ms: float) -> None:
        """Adjust audio_latency_offset_ms by delta_ms and persist.

        Replaces the flash-based measurement which mixed session-relative
        onset ms with absolute perf_counter ms. The offset is now a small
        signed value set by ear using the early/late histogram.
        """
        current = self._config.get_audio_latency_offset()
        self._config.set_audio_latency_offset(current + delta_ms)
        self._config.save()

    def _draw_histogram(self, surface: pygame.Surface, screen_w: int,
                        y: int, stats: "TimingStats") -> None:
        """Render the early/late timing-error histogram from TimingStats.

        Bins cover -100..+100 ms in 10 ms steps (HISTOGRAM_NUM_BINS=20).
        Negative = early (player ahead), positive = late (player behind).
        """
        from pickhero.timing import HISTOGRAM_NUM_BINS, HISTOGRAM_RANGE_MS

        t = get_theme()
        bins = stats.histogram_bins or []
        max_count = max(bins) if bins else 0
        bar_w = 12
        gap = 2
        total_w = HISTOGRAM_NUM_BINS * (bar_w + gap)
        start_x = screen_w // 2 - total_w // 2
        # Baseline at y + max_height
        max_h = 60
        baseline_y = y + max_h

        for i in range(HISTOGRAM_NUM_BINS):
            count = bins[i] if i < len(bins) else 0
            h = int((count / max_count) * max_h) if max_count > 0 else 0
            bin_center_ms = -HISTOGRAM_RANGE_MS / 2 + (i + 0.5) * (HISTOGRAM_RANGE_MS / HISTOGRAM_NUM_BINS)
            # Color: early=cool, on-time=accent, late=warm
            if bin_center_ms < -25:
                color = t.signal_cold
            elif bin_center_ms > 25:
                color = t.feedback_miss
            else:
                color = t.hud_accent
            bx = start_x + i * (bar_w + gap)
            pygame.draw.rect(surface, t.signal_cold, (bx, y, bar_w, max_h))
            if h > 0:
                pygame.draw.rect(surface, color, (bx, baseline_y - h, bar_w, h))

        # Center line marker
        center_x = start_x + (HISTOGRAM_NUM_BINS // 2) * (bar_w + gap) + bar_w // 2
        pygame.draw.line(surface, t.hud_text, (center_x, y - 4), (center_x, baseline_y + 4), 1)

        hint_font = _get_font("arial", 12)
        early_surf = hint_font.render("early", True, t.hud_text)
        late_surf = hint_font.render("late", True, t.hud_text)
        surface.blit(early_surf, (start_x - early_surf.get_width() - 6, y + max_h // 2))
        surface.blit(late_surf, (start_x + total_w + 6, y + max_h // 2))
# Backwards-compatible alias used by some verification scripts.
CalibrationMenu = CalibrationMenuScreen
