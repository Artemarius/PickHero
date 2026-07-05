"""Real-time guitar articulation detection.

The detector layers on the existing pitch detector (yinfast) and onset
detector, building a :class:`~pickhero.audio.performance.PerformanceEvent`
frame-by-frame while a note is sounding. When the next onset fires, the
in-progress event is closed (``release_ms`` set) and pushed onto a completed
list drained by :class:`~pickhero.audio.detector.PitchDetector.drain_events`.

Detection gates (left-hand pitch-contour + right-hand spectral) come from
published research:

Left-hand (pitch-contour based):
  - hammer_on: abrupt ascending pitch change without onset
  - pull_off: abrupt descending pitch change without onset
  - bend: gradual monotonic pitch drift >35 cents over >80ms
  - vibrato: periodic pitch oscillation at 3-8 Hz, 10-60 cents
  - slide: staircase pitch profile, monotonic, <50 cents/frame

Right-hand (spectral, on onset frames):
  - palm_mute: fast energy decay + low spectral centroid
  - harmonic: weak fundamental, strong overtones at 1.5×F0

Compute cost: <0.05ms per frame for pitch-based detectors,
+0.3ms on onset frames for spectral detectors. The heavy spectral work
(``flux``/``flatness``/``hnr``) is gated behind ``is_onset`` so it never runs
in the audio callback path.
"""

from __future__ import annotations

import math
import numpy as np

from pickhero.audio.performance import (
    PerformanceEvent,
)


# --- Thresholds (detection gates — distinct from Judge coaching thresholds) ---

# Legato (hammer-on / pull-off)
_LEGATO_JUMP_CENTS = 150.0      # abrupt pitch jump threshold per frame (1.5 semitones — YIN jitter on low E routinely exceeds 50 cents between frames)
_LEGATO_MIN_SEMITONE = 1.0     # minimum semitone difference
_LEGATO_GAP_THRESHOLD_MS = 40.0 # transition must be faster than this (no pick gap)

# Bend
_BEND_THRESHOLD_CENTS = 35.0   # sustained deviation to count as bend
_BEND_MIN_DURATION_MS = 80.0   # minimum sustain duration
_BEND_MAX_FRAME_CENTS = 50.0   # max cents between consecutive frames (smooth)

# Vibrato — DETECTION band. The VibratoJudge coaching band (4-8 Hz / 30-80¢)
# lives in analyzer.py and is intentionally stricter. Do not conflate.
_VIB_RATE_MIN_HZ = 3.0
_VIB_RATE_MAX_HZ = 8.0
_VIB_AMP_MIN_CENTS = 10.0
_VIB_AMP_MAX_CENTS = 60.0
_VIB_BUFFER_FRAMES = 24         # ~280ms at hop=512/44100Hz
_VIB_MIN_FRAMES = 10

# Slide
_SLIDE_SUDDEN_JUMP_CENTS = 50.0  # reject if any pair exceeds this
_SLIDE_TRIGGER_CENTS = 100.0      # crossed a semitone boundary
_SLIDE_MONOTONIC_TOL = 2.0        # cents tolerance for "same direction"
_SLIDE_CHECK_LEN = 5              # frames to check for monotonic trend
_SLIDE_MIN_FRAMES = 3             # minimum frames in same direction
_SLIDE_LANDING_STABILITY_CENTS = 5.0  # cents delta below which pitch has "landed"

# Palm mute
_PALM_MUTE_DECAY_SLOPE = -0.05    # energy decay ratio threshold
_PALM_MUTE_LOWPASS_HZ = 500.0     # frequency band for energy analysis
_PALM_MUTE_FRAMES = 4             # frames after onset to analyze
_PALM_MUTE_CENTROID_THRESHOLD = 1500.0  # Hz — muted notes are dull

# Harmonic
_HARMONIC_ATTACK_MS = 40.0        # attack analysis window
_HARMONIC_SUBHARMONIC_RATIO_DB = 0.0  # F0 must be weaker than the 1.5× subharmonic
_HARMONIC_OVERTONE_RATIO = 2.0    # 2nd overtone / fundamental magnitude (unused — kept for reference)


class ArticulationDetector:
    """Real-time guitar articulation detector.

    Processes per-frame pitch and onset data to build a
    :class:`PerformanceEvent` per note. Technique candidates are appended to
    the *active* event as they fire; the event is closed and pushed to the
    completed list when the next onset arrives.

    Usage::

        detector = ArticulationDetector(sample_rate=44100, hop_size=512)
        detector.process(freq, confidence, is_onset, audio_buffer, timestamp_ms)
        completed = detector.drain_completed()  # list[PerformanceEvent]
        active = detector.active_event         # the in-progress note, or None
    """

    def __init__(self, sample_rate: int = 44100, hop_size: int = 512, buf_size: int = 2048):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.buf_size = buf_size
        self._frame_duration_ms = (hop_size / sample_rate) * 1000.0

        # Pitch contour history: (timestamp_ms, freq, cents_from_base, is_onset)
        self._pitch_history: list[tuple[float, float, float, bool]] = []
        self._base_freq: float = 0.0  # the note's base frequency (set on onset)

        # Bend state
        self._bend_active: bool = False
        self._bend_start_ms: float = 0.0

        # Vibrato buffer: cents deviation values
        self._vib_buffer: list[float] = []
        self._vib_peak_times: list[float] = []  # for regularity metric

        # Slide state
        self._slide_cents_history: list[float] = []

        # Palm mute: energy history after onset
        self._pm_energy_history: list[float] = []
        self._pm_onset_active: bool = False

        # Harmonic: attack buffer
        self._harm_attack_samples = int(sample_rate * (_HARMONIC_ATTACK_MS / 1000.0))
        self._harm_audio_buffer: list[np.ndarray] = []

        # Spectral flux: previous magnitude spectrum
        self._prev_spectrum: np.ndarray | None = None

        # Active + completed PerformanceEvents.
        self._active: PerformanceEvent | None = None
        self._completed: list[PerformanceEvent] = []

        # Previous event's midi_note (for real-time legato direction proxy).
        self._prev_event_midi: int | None = None

    # ------------------------------------------------------------------ state

    @property
    def active_event(self) -> PerformanceEvent | None:
        """The PerformanceEvent currently being built (the sounding note)."""
        return self._active

    def reset(self) -> None:
        """Clear all internal state. Call on new song/session."""
        self._pitch_history.clear()
        self._base_freq = 0.0
        self._bend_active = False
        self._bend_start_ms = 0.0
        self._vib_buffer.clear()
        self._vib_peak_times.clear()
        self._slide_cents_history.clear()
        self._pm_energy_history.clear()
        self._pm_onset_active = False
        self._harm_audio_buffer.clear()
        self._prev_spectrum = None
        self._active = None
        self._completed.clear()
        self._prev_event_midi = None

    def drain_completed(self) -> list[PerformanceEvent]:
        """Return and clear the list of PerformanceEvents completed since the
        last call. A note is *completed* when the next onset fires (its
        ``release_ms`` is set at that point)."""
        drained = self._completed
        self._completed = []
        return drained

    # ----------------------------------------------------------------- process

    def process(
        self,
        freq: float,
        confidence: float,
        is_onset: bool,
        audio_buffer: np.ndarray | None,
        timestamp_ms: float,
    ) -> list[PerformanceEvent]:
        """Process one frame of pitch/onset data.

        Appends to the active :class:`PerformanceEvent`'s ``f0_curve``,
        ``energy_envelope``, and (on onset frames) ``spectral_features``.
        On a new onset, closes the previous event and starts a new one.

        Args:
            freq: Detected frequency (Hz), after octave correction.
            confidence: Pitch confidence (0.0-1.0).
            is_onset: Whether an onset was detected this frame.
            audio_buffer: Raw audio samples for this frame (for spectral
                analysis). May be None in tests.
            timestamp_ms: Timestamp of this frame (ms from session start).

        Returns:
            The list of PerformanceEvents *newly completed* this frame (a
            note that just released). Usually empty; non-empty only on the
            frame where a new onset closes the previous note. Phase-1
            detector emits at most one :class:`TechniqueCandidate` per note.
        """
        newly_completed: list[PerformanceEvent] = []

        # --- Onset: close the previous event, start a new one ---
        if is_onset and freq > 0 and confidence > 0.3:
            if self._active is not None:
                self._active.release_ms = timestamp_ms
                self._prev_event_midi = self._active.midi_note
                self._completed.append(self._active)
                newly_completed.append(self._active)

            # Start fresh event
            self._active = PerformanceEvent(
                onset_ms=timestamp_ms,
                midi_note=_freq_to_midi(freq),
                confidence=confidence,
            )
            self._base_freq = freq
            self._pitch_history.clear()
            self._vib_buffer.clear()
            self._vib_peak_times.clear()
            self._slide_cents_history.clear()
            self._bend_active = False
            self._bend_start_ms = 0.0

            # Spectral features for the onset frame (gated behind is_onset).
            if audio_buffer is not None:
                self._record_spectral_features(audio_buffer, freq, timestamp_ms)
                # Harmonic detection disabled in real-time pipeline.
                # The 1.5×F0 subharmonic check fires on every normal guitar
                # note (spectral leakage + inharmonicity produce energy at
                # non-harmonic frequencies). Harmonic detection must be
                # tab-conditioned (only run when the tab expects a harmonic)
                # — a spectral heuristic alone can't distinguish a real
                # harmonic from a normal pick. See Patch 4d/5c for the
                # tab-conditioned path in the analyzer.
                # if freq > 0 and self._detect_harmonic(audio_buffer, freq):
                #     self._active.upsert_technique_candidate(...)
                # Onset features (pick transient / noise burst proxies)
                self._active.onset_features = self._compute_onset_features(audio_buffer)

            # Begin palm-mute window
            self._pm_onset_active = True
            self._pm_energy_history.clear()
            if audio_buffer is not None:
                rms = float(np.sqrt(np.mean(audio_buffer ** 2))) if len(audio_buffer) else 0.0
                self._pm_energy_history.append(rms)
                self._active.energy_envelope.append((timestamp_ms, rms))

            # Record f0 point at onset
            self._active.f0_curve.append((timestamp_ms, freq, 0.0))
            self._pitch_history.append((timestamp_ms, freq, 0.0, True))
            return newly_completed

        # No onset but an active event: accumulate the curve.
        if self._active is None:
            # No onset yet — nothing to attribute this frame to. We still keep
            # the palm-mute / harmonic buffers clean.
            return newly_completed

        # Skip if no valid pitch or low confidence (curve keeps its last value;
        # we don't append garbage points).
        if freq <= 0 or confidence < 0.3 or self._base_freq <= 0:
            # Still track energy for the envelope (dead-note / release tail).
            if audio_buffer is not None:
                rms = float(np.sqrt(np.mean(audio_buffer ** 2))) if len(audio_buffer) else 0.0
                self._active.energy_envelope.append((timestamp_ms, rms))
            return newly_completed

        cents = 1200.0 * math.log2(freq / self._base_freq) if self._base_freq > 0 else 0.0

        # Record curves
        self._active.f0_curve.append((timestamp_ms, freq, cents))
        if audio_buffer is not None:
            rms = float(np.sqrt(np.mean(audio_buffer ** 2))) if len(audio_buffer) else 0.0
            self._active.energy_envelope.append((timestamp_ms, rms))

        self._pitch_history.append((timestamp_ms, freq, cents, is_onset))
        if len(self._pitch_history) > 30:
            self._pitch_history.pop(0)

        # --- Classification (appends candidates to the active event) ---
        # All detectors run per frame (no priority-return chain) so a note can
        # accumulate compound tags (e.g. bend + vibrato). De-dup via upsert.
        #
        # Per-frame reset for legato_transition only: it's a one-shot transition
        # (the frame where the pitch jumps without a pick). Without this reset,
        # it sticks for the entire note sustain (~5s) and floods the matcher.
        # slide_landing and bend_target are *states* — once the slide has landed
        # or the bend has reached its target, that state persists.
        if self._active.event_kind == "legato_transition":
            self._active.event_kind = "sustain_update"
        legato = self._detect_legato(cents, timestamp_ms)
        if legato is not None:
            # Real-time direction proxy: pitch delta vs previous event's midi.
            kind = self._resolve_legato_kind(legato, freq)
            self._active.upsert_technique_candidate(
                kind, 0.7,
                detected_cents=cents,
                metrics={"transition_ms": self._frame_duration_ms},
            )
            # The destination note is sounding without a pick — this event IS
            # the legato transition (hammer-on / pull-off landing).
            self._active.event_kind = "legato_transition"

        vib_metrics = self._detect_vibrato(cents, timestamp_ms)
        if vib_metrics is not None:
            self._active.upsert_technique_candidate(
                "vibrato", 0.7,
                metrics=vib_metrics,
            )

        slide = self._detect_slide(cents, timestamp_ms)
        if slide is not None:
            self._active.upsert_technique_candidate(
                "slide", 0.6,
                detected_cents=cents,
                metrics={"direction": slide},
            )
            # While the pitch is still sliding, leave event_kind as pick_onset.
            # When the cents delta flattens (pitch arrived at destination fret),
            # mark the landing so the matcher can pair it with the slide spec.
            if len(self._slide_cents_history) >= 2:
                last_delta = abs(self._slide_cents_history[-1] - self._slide_cents_history[-2])
                if last_delta < _SLIDE_LANDING_STABILITY_CENTS:
                    self._active.event_kind = "slide_landing"

        bend = self._detect_bend(cents, timestamp_ms)
        if bend is not None and self._active.event_kind != "slide_landing":
            # A monotonic drift can trigger both slide and bend detectors.
            # If the event is already classified as slide_landing, don't
            # append a bend candidate — it would produce a conflicting
            # bend verdict on a note the tab expects as a slide (Judge B).
            self._active.upsert_technique_candidate(
                "bend", 0.8,
                detected_cents=cents,
                target_cents=None,  # real-time can't know tab target
            )
            # _detect_bend sets _bend_active=True when the sustained drift
            # crosses the duration threshold — that's the plateau (target reached).
            # Don't override a more specific event_kind already set this note
            # (e.g. slide_landing on the same contour).
            if self._bend_active and self._active.event_kind == "pick_onset":
                self._active.event_kind = "bend_target"

        # Palm mute: spectral, runs over the post-onset window
        if self._pm_onset_active and audio_buffer is not None:
            pm = self._detect_palm_mute(audio_buffer, timestamp_ms)
            if pm:
                self._pm_onset_active = False
                # Compute metrics for the candidate
                halflife = self._decay_halflife_ms()
                centroid = self._spectral_centroid(audio_buffer)
                self._active.upsert_technique_candidate(
                    "palm_mute", 0.7,
                    metrics={
                        "decay_halflife_ms": halflife,
                        "centroid_hz": centroid,
                    },
                )

        return newly_completed

    # --------------------------------------------------------- spectral helpers

    def _record_spectral_features(
        self, audio_buffer: np.ndarray, f0: float, timestamp_ms: float
    ) -> None:
        """Append a per-frame spectral feature dict to the active event."""
        if self._active is None or len(audio_buffer) < 2:
            return
        window = np.hanning(len(audio_buffer))
        spectrum = np.abs(np.fft.rfft(audio_buffer * window))
        freqs = np.fft.rfftfreq(len(audio_buffer), 1.0 / self.sample_rate)

        centroid = float(
            np.sum(freqs * spectrum) / np.sum(spectrum)
        ) if np.sum(spectrum) > 0 else 0.0
        flatness = self._spectral_flatness(spectrum)
        flux = self._spectral_flux(spectrum)
        hnr = self._hnr(audio_buffer, f0) if f0 > 0 else 0.0

        self._active.spectral_features.append({
            "time_ms": timestamp_ms,
            "centroid": centroid,
            "flux": flux,
            "flatness": flatness,
            "hnr": hnr,
        })

    def _compute_onset_features(self, audio_buffer: np.ndarray) -> dict:
        """Compute onset-frame features used by the LegatoJudge / DeadNoteJudge.

        - ``pick_transient``: peak RMS in the first ~10ms (attack sharpness).
        - ``noise_burst``: spectral flatness of the attack (high = noisy/dead).
        - ``fret_transient``: reserved (Phase 2).
        """
        n = len(audio_buffer)
        if n == 0:
            return {}
        attack_samples = min(n, int(self.sample_rate * 0.010))
        attack = audio_buffer[:attack_samples]
        peak_rms = float(np.sqrt(np.mean(attack ** 2))) if attack_samples else 0.0
        # Noise burst: spectral flatness of the full frame
        window = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(audio_buffer * window))
        noise = self._spectral_flatness(spectrum)
        return {
            "pick_transient": peak_rms,
            "pick_transient_strength": min(1.0, peak_rms * 4.0),  # rough normalization
            "noise_burst": noise,
            "fret_transient": 0.0,
        }

    # --- Legato (hammer-on / pull-off) ---

    def _detect_legato(self, cents: float, timestamp_ms: float) -> str | None:
        """Detect hammer-on or pull-off from abrupt pitch change without onset."""
        if len(self._pitch_history) < 2:
            return None

        prev_ts, prev_freq, prev_cents, prev_onset = self._pitch_history[-2]
        curr_ts, curr_freq, curr_cents, curr_onset = self._pitch_history[-1]

        # Must not be an onset frame (legato = no pick)
        if curr_onset or prev_onset:
            return None

        # Pitch must jump rapidly
        cents_delta = abs(curr_cents - prev_cents)
        if cents_delta < _LEGATO_JUMP_CENTS:
            return None

        # Must be at least 1 semitone change
        if abs(curr_cents) < 100.0 * _LEGATO_MIN_SEMITONE and abs(prev_cents) < 100.0 * _LEGATO_MIN_SEMITONE:
            # The delta itself must cross a semitone
            if cents_delta < 100.0:
                return None

        # Transition must be fast (no amplitude gap from picking)
        gap_ms = curr_ts - prev_ts
        if gap_ms > _LEGATO_GAP_THRESHOLD_MS * 2:
            return None

        # Direction: ascending = hammer-on, descending = pull-off
        if curr_cents > prev_cents:
            return "hammer_on"
        else:
            return "pull_off"

    def _resolve_legato_kind(self, detected: str, freq: float) -> str:
        """Real-time direction proxy: compare current pitch to the previous
        event's midi_note. The matcher does the authoritative resolution later
        (Step 9) from the tab's neighbor NoteEvent."""
        if self._prev_event_midi is None:
            return detected
        cur_midi = _freq_to_midi(freq)
        if cur_midi is None:
            return detected
        if cur_midi > self._prev_event_midi:
            return "hammer_on"
        elif cur_midi < self._prev_event_midi:
            return "pull_off"
        return detected

    # --- Vibrato ---

    def _detect_vibrato(self, cents: float, timestamp_ms: float) -> dict | None:
        """Detect vibrato via zero-crossing rate of detrended pitch contour.

        Returns a metrics dict (``rate_hz``, ``depth_cents``, ``regularity``)
        when vibrato is detected, else None.
        """
        self._vib_buffer.append(cents)
        if len(self._vib_buffer) > _VIB_BUFFER_FRAMES:
            self._vib_buffer.pop(0)

        if len(self._vib_buffer) < _VIB_MIN_FRAMES:
            return None

        ordered = self._vib_buffer
        mean = sum(ordered) / len(ordered)

        # Count zero-crossings of the mean
        crossings = 0
        for i in range(len(ordered) - 1):
            if (ordered[i] - mean) * (ordered[i + 1] - mean) < 0:
                crossings += 1

        duration_s = len(ordered) * self._frame_duration_ms / 1000.0
        if duration_s < 0.05:
            return None

        rate_hz = crossings / (2.0 * duration_s)
        if rate_hz < _VIB_RATE_MIN_HZ or rate_hz > _VIB_RATE_MAX_HZ:
            return None

        amplitude = max(abs(c - mean) for c in ordered)
        if amplitude < _VIB_AMP_MIN_CENTS or amplitude > _VIB_AMP_MAX_CENTS:
            return None

        # Regularity: 1 - cv of inter-peak intervals
        peaks = self._find_vibrato_peaks(ordered, mean)
        regularity = self._peak_regularity(peaks)

        # Track peak times for regularity over the full sustain
        if peaks:
            self._vib_peak_times.append(timestamp_ms)

        depth_cents = amplitude
        return {
            "rate_hz": rate_hz,
            "depth_cents": depth_cents,
            "regularity": regularity,
            "center_offset_cents": mean,
        }

    @staticmethod
    def _find_vibrato_peaks(buffer: list[float], mean: float) -> list[int]:
        """Return indices of local maxima above the mean."""
        peaks = []
        for i in range(1, len(buffer) - 1):
            if buffer[i] > buffer[i - 1] and buffer[i] > buffer[i + 1] and buffer[i] > mean:
                peaks.append(i)
        return peaks

    @staticmethod
    def _peak_regularity(peaks: list[int]) -> float:
        """Regularity = 1 - cv of inter-peak intervals (1.0 = perfectly regular)."""
        if len(peaks) < 2:
            return 0.0
        intervals = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
        mean = sum(intervals) / len(intervals)
        if mean <= 0:
            return 0.0
        var = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        std = math.sqrt(var)
        cv = std / mean
        return max(0.0, 1.0 - cv)

    # --- Slide ---

    def _detect_slide(self, cents: float, timestamp_ms: float) -> str | None:
        """Detect slide via monotonic staircase pitch pattern.

        Returns ``"up"`` or ``"down"`` when a slide is detected, else None.
        """
        self._slide_cents_history.append(cents)
        if len(self._slide_cents_history) > 8:
            self._slide_cents_history.pop(0)

        if len(self._slide_cents_history) < _SLIDE_CHECK_LEN:
            return None

        recent = self._slide_cents_history[-_SLIDE_CHECK_LEN:]

        # Reject if any consecutive pair has a sudden jump (that's legato, not slide)
        for i in range(len(recent) - 1):
            if abs(recent[i + 1] - recent[i]) > _SLIDE_SUDDEN_JUMP_CENTS:
                return None

        # Check for monotonic trend
        increasing = sum(
            1 for i in range(len(recent) - 1)
            if recent[i + 1] - recent[i] > _SLIDE_MONOTONIC_TOL
        )
        decreasing = sum(
            1 for i in range(len(recent) - 1)
            if recent[i + 1] - recent[i] < -_SLIDE_MONOTONIC_TOL
        )

        is_monotonic_up = increasing >= _SLIDE_MIN_FRAMES and decreasing == 0
        is_monotonic_down = decreasing >= _SLIDE_MIN_FRAMES and increasing == 0

        if not (is_monotonic_up or is_monotonic_down):
            return None

        # Trigger when crossing a semitone boundary
        if abs(cents) >= _SLIDE_TRIGGER_CENTS:
            return "up" if is_monotonic_up else "down"

        return None

    # --- Bend ---

    def _detect_bend(self, cents: float, timestamp_ms: float) -> str | None:
        """Detect bend via gradual monotonic pitch drift."""
        # Need pitch history
        if len(self._pitch_history) < 3:
            return None

        # Get recent cents values
        recent = [h[2] for h in self._pitch_history[-min(len(self._pitch_history), 8):]]

        # Check max deviation
        max_abs = max(abs(c) for c in recent)

        if max_abs < _BEND_THRESHOLD_CENTS:
            if self._bend_active:
                # Bend released
                self._bend_active = False
            return None

        # Reject if any frame-to-frame jump is too large (that's legato, not bend)
        for i in range(1, len(recent)):
            if abs(recent[i] - recent[i - 1]) > _BEND_MAX_FRAME_CENTS:
                return None

        # Check duration
        if len(self._pitch_history) >= 2:
            duration = timestamp_ms - self._pitch_history[0][0]
            if duration >= _BEND_MIN_DURATION_MS:
                self._bend_active = True
                return "bend"

        return None

    # --- Palm mute (spectral) ---

    def _detect_palm_mute(self, audio_buffer: np.ndarray, timestamp_ms: float) -> bool:
        """Detect palm mute via fast energy decay and low spectral centroid."""
        # Compute RMS energy
        rms = float(np.sqrt(np.mean(audio_buffer ** 2)))
        self._pm_energy_history.append(rms)

        if len(self._pm_energy_history) > _PALM_MUTE_FRAMES:
            self._pm_energy_history = self._pm_energy_history[-_PALM_MUTE_FRAMES:]

        if len(self._pm_energy_history) < _PALM_MUTE_FRAMES:
            return False

        # Check decay slope
        peak_idx = int(np.argmax(self._pm_energy_history))
        if peak_idx >= len(self._pm_energy_history) - 1:
            return False

        peak_energy = self._pm_energy_history[peak_idx]
        next_energy = self._pm_energy_history[peak_idx + 1]

        if peak_energy <= 0:
            return False

        decay_ratio = (next_energy - peak_energy) / peak_energy
        if decay_ratio >= _PALM_MUTE_DECAY_SLOPE:
            return False

        # Confirm with spectral centroid (muted = dull)
        centroid = self._spectral_centroid(audio_buffer)
        if centroid > _PALM_MUTE_CENTROID_THRESHOLD:
            return False

        return True

    def _decay_halflife_ms(self) -> float:
        """Estimate the energy decay halflife from the palm-mute window."""
        if len(self._pm_energy_history) < 2:
            return 0.0
        peak = max(self._pm_energy_history)
        if peak <= 0:
            return 0.0
        half = peak / 2.0
        # Frames until energy drops below half
        for i, e in enumerate(self._pm_energy_history):
            if e <= half:
                return i * self._frame_duration_ms
        return len(self._pm_energy_history) * self._frame_duration_ms

    def _spectral_centroid(self, audio_buffer: np.ndarray) -> float:
        """Compute spectral centroid (brightness) of an audio buffer."""
        if len(audio_buffer) < 2:
            return 0.0
        window = np.hanning(len(audio_buffer))
        spectrum = np.abs(np.fft.rfft(audio_buffer * window))
        freqs = np.fft.rfftfreq(len(audio_buffer), 1.0 / self.sample_rate)
        total_mag = float(np.sum(spectrum))
        if total_mag <= 0:
            return 0.0
        return float(np.sum(freqs * spectrum) / total_mag)

    def _spectral_flux(self, spectrum: np.ndarray) -> float:
        """L1 norm of magnitude delta vs the previous frame's spectrum."""
        if self._prev_spectrum is None:
            self._prev_spectrum = spectrum.copy()
            return 0.0
        # Pad to match lengths
        a, b = self._prev_spectrum, spectrum
        if len(a) != len(b):
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
        flux = float(np.sum(np.abs(b - a)))
        self._prev_spectrum = spectrum.copy()
        return flux

    @staticmethod
    def _spectral_flatness(spectrum: np.ndarray) -> float:
        """Geometric mean / arithmetic mean of magnitude (0=pitched, 1=noise)."""
        if len(spectrum) == 0:
            return 0.0
        mag = np.maximum(spectrum, 1e-12)
        geo = float(np.exp(np.mean(np.log(mag))))
        arith = float(np.mean(mag))
        if arith <= 0:
            return 0.0
        return geo / arith

    def _hnr(self, audio_buffer: np.ndarray, f0: float) -> float:
        """Harmonic-to-noise ratio: energy at integer multiples of f0 vs total.

        Reuses a zero-padded FFT for adequate frequency resolution at low f0.
        """
        if f0 <= 0 or len(audio_buffer) < 64:
            return 0.0
        fft_n = max(4096, len(audio_buffer))
        window = np.hanning(len(audio_buffer))
        padded = np.zeros(fft_n, dtype=np.float32)
        padded[:len(audio_buffer)] = audio_buffer * window
        spectrum = np.abs(np.fft.rfft(padded))
        freqs = np.fft.rfftfreq(fft_n, 1.0 / self.sample_rate)
        total = float(np.sum(spectrum))
        if total <= 0:
            return 0.0
        # Sum energy at integer multiples of f0 (up to Nyquist)
        nyquist = self.sample_rate / 2.0
        harmonic_energy = 0.0
        h = 1
        while f0 * h < nyquist:
            harmonic_energy += self._mag_at_freq(spectrum, freqs, f0 * h)
            h += 1
        return float(harmonic_energy / total)

    # --- Harmonic (spectral) ---

    def _detect_harmonic(self, audio_buffer: np.ndarray, fundamental_freq: float) -> bool:
        """Detect harmonic via weak fundamental and strong overtones."""
        if fundamental_freq <= 0 or len(audio_buffer) < 64:
            return False

        # Zero-pad to at least 4096 samples for adequate frequency resolution.
        attack = audio_buffer
        fft_n = max(4096, len(attack))
        window = np.hanning(len(attack))
        padded = np.zeros(fft_n, dtype=np.float32)
        padded[:len(attack)] = attack * window
        spectrum = np.abs(np.fft.rfft(padded))
        freqs = np.fft.rfftfreq(fft_n, 1.0 / self.sample_rate)
        # Detect harmonic via the 1.5×F0 subharmonic being stronger than F0.
        # The 2nd-overtone check (2×F0 > fundamental) was removed — on a real
        # low E the octave is naturally 2-3× louder than the weak fundamental,
        # causing false positives on every normal pick.
        f0_mag = self._mag_at_freq(spectrum, freqs, fundamental_freq)
        sub_mag = self._mag_at_freq(spectrum, freqs, 1.5 * fundamental_freq)
        if f0_mag > 0 and sub_mag > 0:
            ratio_db = 20.0 * math.log10(f0_mag / sub_mag)
            if ratio_db < _HARMONIC_SUBHARMONIC_RATIO_DB:
                return True
        return False

    @staticmethod
    def _mag_at_freq(
        spectrum: np.ndarray, freqs: np.ndarray, target_freq: float, tolerance: float = 0.05
    ) -> float:
        """Get spectral magnitude at a target frequency (with ±tolerance)."""
        lo = target_freq * (1 - tolerance)
        hi = target_freq * (1 + tolerance)
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            return 0.0
        return float(np.max(spectrum[mask]))


def _freq_to_midi(freq: float) -> int | None:
    """Convert a frequency in Hz to a MIDI note number (or None)."""
    if freq <= 0:
        return None
    midi = round(69 + 12 * math.log2(freq / 440.0))
    if 0 <= midi <= 127:
        return midi
    return None
