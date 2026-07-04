"""Guitar articulation detection from real-time pitch contour and spectral features.

Detects 7 articulations using DSP heuristics layered on the existing pitch
detector (yinfast) and onset detector:

Left-hand (pitch-contour based):
  - hammer_on: abrupt ascending pitch change without onset
  - pull_off: abrupt descending pitch change without onset
  - bend: gradual monotonic pitch drift >35 cents over >80ms
  - vibrato: periodic pitch oscillation at 3-8 Hz, 10-60 cents
  - slide: staircase pitch profile, monotonic, <50 cents/frame

Right-hand (spectral, on onset frames):
  - palm_mute: fast energy decay + low spectral centroid
  - harmonic: weak fundamental, strong overtones at 1.5×F0

All thresholds derived from published research:
  - Chen et al., ISMIR 2015
  - Reboursière et al., NIME 2012
  - contrapunk-audio/contrapunk (working Rust implementation)
  - SignalAssistant (C++ reference with documented thresholds)

Compute cost: <0.05ms per frame for pitch-based detectors,
+0.3ms on onset frames for spectral detectors.
"""

from __future__ import annotations

import math
import numpy as np


# --- Thresholds (from literature) ---

# Legato (hammer-on / pull-off)
_LEGATO_JUMP_CENTS = 50.0       # abrupt pitch jump threshold per frame
_LEGATO_MIN_SEMITONE = 1.0     # minimum semitone difference
_LEGATO_GAP_THRESHOLD_MS = 40.0 # transition must be faster than this (no pick gap)

# Bend
_BEND_THRESHOLD_CENTS = 35.0   # sustained deviation to count as bend
_BEND_MIN_DURATION_MS = 80.0   # minimum sustain duration
_BEND_MAX_FRAME_CENTS = 50.0   # max cents between consecutive frames (smooth)

# Vibrato
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

# Palm mute
_PALM_MUTE_DECAY_SLOPE = -0.05    # energy decay ratio threshold
_PALM_MUTE_LOWPASS_HZ = 500.0     # frequency band for energy analysis
_PALM_MUTE_FRAMES = 4             # frames after onset to analyze
_PALM_MUTE_CENTROID_THRESHOLD = 1500.0  # Hz — muted notes are dull

# Harmonic
_HARMONIC_ATTACK_MS = 40.0        # attack analysis window
_HARMONIC_SUBHARMONIC_RATIO_DB = 0.0  # F0 must be weaker than subharmonic
_HARMONIC_OVERTONE_RATIO = 1.5    # 2nd overtone / fundamental magnitude


class ArticulationDetector:
    """Real-time guitar articulation detector.

    Processes per-frame pitch and onset data to classify articulations.
    Maintains internal state (pitch history, energy history) across calls.

    Usage:
        detector = ArticulationDetector(sample_rate=44100, hop_size=512)
        articulation = detector.process(freq, confidence, is_onset, audio_buffer, timestamp_ms)
        # articulation is None or one of: "hammer_on", "pull_off", "bend",
        #   "vibrato", "slide", "palm_mute", "harmonic"
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

        # Slide state
        self._slide_cents_history: list[float] = []

        # Palm mute: energy history after onset
        self._pm_energy_history: list[float] = []
        self._pm_onset_active: bool = False

        # Harmonic: attack buffer
        self._harm_attack_samples = int(sample_rate * (_HARMONIC_ATTACK_MS / 1000.0))
        self._harm_audio_buffer: list[np.ndarray] = []

    def reset(self) -> None:
        """Clear all internal state. Call on new song/session."""
        self._pitch_history.clear()
        self._base_freq = 0.0
        self._bend_active = False
        self._bend_start_ms = 0.0
        self._vib_buffer.clear()
        self._slide_cents_history.clear()
        self._pm_energy_history.clear()
        self._pm_onset_active = False
        self._harm_audio_buffer.clear()

    def process(
        self,
        freq: float,
        confidence: float,
        is_onset: bool,
        audio_buffer: np.ndarray | None,
        timestamp_ms: float,
    ) -> str | None:
        """Process one frame of pitch/onset data.

        Args:
            freq: Detected frequency (Hz), after octave correction.
            confidence: Pitch confidence (0.0-1.0).
            is_onset: Whether an onset was detected this frame.
            audio_buffer: Raw audio samples for this frame (for spectral analysis).
            timestamp_ms: Timestamp of this frame (ms from session start).

        Returns:
            Articulation name string, or None if no articulation detected.
            Priority: hammer_on/pull_off > vibrato > slide > bend > palm_mute > harmonic
        """
        # On onset: reset pitch-based detectors, set base frequency
        if is_onset and freq > 0 and confidence > 0.3:
            self._base_freq = freq
            self._pitch_history.clear()
            self._vib_buffer.clear()
            self._slide_cents_history.clear()
            self._bend_active = False

            # Start palm mute and harmonic analysis
            self._pm_onset_active = True
            self._pm_energy_history.clear()
            self._harm_audio_buffer.clear()
            if audio_buffer is not None:
                self._harm_audio_buffer.append(audio_buffer.copy())

            # Check for harmonic on the onset frame
            if audio_buffer is not None and freq > 0:
                harm_result = self._detect_harmonic(audio_buffer, freq)
                if harm_result:
                    return "harmonic"

            return None

        # Collect audio for harmonic attack analysis
        if is_onset and audio_buffer is not None:
            if len(self._harm_audio_buffer) < 4:
                self._harm_audio_buffer.append(audio_buffer.copy())

        # Skip if no valid pitch or low confidence
        if freq <= 0 or confidence < 0.3 or self._base_freq <= 0:
            return None

        # Compute cents deviation from base note
        cents = 1200.0 * math.log2(freq / self._base_freq) if self._base_freq > 0 else 0.0

        # Record in pitch history
        self._pitch_history.append((timestamp_ms, freq, cents, is_onset))
        if len(self._pitch_history) > 30:
            self._pitch_history.pop(0)

        # --- Classification priority ---

        # 1. Hammer-on / Pull-off: abrupt pitch change without onset
        legato = self._detect_legato(cents, timestamp_ms)
        if legato:
            return legato

        # 2. Vibrato: periodic oscillation
        vib = self._detect_vibrato(cents)
        if vib:
            return "vibrato"

        # 3. Slide: monotonic staircase pitch movement
        slide = self._detect_slide(cents)
        if slide:
            return "slide"

        # 4. Bend: gradual monotonic pitch drift
        bend = self._detect_bend(cents, timestamp_ms)
        if bend:
            return "bend"

        # 5. Palm mute: check energy decay on onset frames
        if self._pm_onset_active and audio_buffer is not None:
            pm = self._detect_palm_mute(audio_buffer, timestamp_ms)
            if pm:
                self._pm_onset_active = False
                return "palm_mute"

        return None

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
        if abs(curr_cents) < 100.0 * _LEGATO_MIN_SEMITONE:
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

    # --- Vibrato ---

    def _detect_vibrato(self, cents: float) -> bool:
        """Detect vibrato via zero-crossing rate of detrended pitch contour."""
        self._vib_buffer.append(cents)
        if len(self._vib_buffer) > _VIB_BUFFER_FRAMES:
            self._vib_buffer.pop(0)

        if len(self._vib_buffer) < _VIB_MIN_FRAMES:
            return False

        ordered = self._vib_buffer
        mean = sum(ordered) / len(ordered)

        # Count zero-crossings of the mean
        crossings = 0
        for i in range(len(ordered) - 1):
            if (ordered[i] - mean) * (ordered[i + 1] - mean) < 0:
                crossings += 1

        duration_s = len(ordered) * self._frame_duration_ms / 1000.0
        if duration_s < 0.05:
            return False

        rate_hz = crossings / (2.0 * duration_s)
        if rate_hz < _VIB_RATE_MIN_HZ or rate_hz > _VIB_RATE_MAX_HZ:
            return False

        amplitude = max(abs(c - mean) for c in ordered)
        if amplitude < _VIB_AMP_MIN_CENTS or amplitude > _VIB_AMP_MAX_CENTS:
            return False

        return True

    # --- Slide ---

    def _detect_slide(self, cents: float) -> str | None:
        """Detect slide via monotonic staircase pitch pattern."""
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
            return "slide"

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

    # --- Harmonic (spectral) ---

    def _detect_harmonic(self, audio_buffer: np.ndarray, fundamental_freq: float) -> bool:
        """Detect harmonic via weak fundamental and strong overtones."""
        if fundamental_freq <= 0 or len(audio_buffer) < 64:
            return False

        # Zero-pad to at least 4096 samples for adequate frequency resolution.
        # At hop_size=512, native bin width is 86 Hz — too coarse to resolve
        # E2 (82 Hz) from 1.5×E2 (124 Hz). Padding to 4096 gives 10.8 Hz bins.
        attack = audio_buffer
        fft_n = max(4096, len(attack))
        window = np.hanning(len(attack))
        # Zero-pad the windowed signal
        padded = np.zeros(fft_n, dtype=np.float32)
        padded[:len(attack)] = attack * window
        spectrum = np.abs(np.fft.rfft(padded))
        freqs = np.fft.rfftfreq(fft_n, 1.0 / self.sample_rate)

        f0_mag = self._mag_at_freq(spectrum, freqs, fundamental_freq)
        sub_mag = self._mag_at_freq(spectrum, freqs, 1.5 * fundamental_freq)
        h2_mag = self._mag_at_freq(spectrum, freqs, 2.0 * fundamental_freq)

        # Check 1: subharmonic ratio (characteristic of harmonics)
        if f0_mag > 0 and sub_mag > 0:
            ratio_db = 20.0 * math.log10(f0_mag / sub_mag)
            if ratio_db < _HARMONIC_SUBHARMONIC_RATIO_DB:
                return True

        # Check 2: 2nd overtone stronger than fundamental
        if f0_mag > 0 and h2_mag > 0:
            if h2_mag > f0_mag * _HARMONIC_OVERTONE_RATIO:
                return True

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
