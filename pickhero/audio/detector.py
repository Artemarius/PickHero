"""Pitch and onset detection using aubio.

Wraps aubio's YIN pitch detector and onset detector.
Processes audio buffers and returns detected notes.
"""

from dataclasses import dataclass

import aubio
import numpy as np

from pickhero.audio.note_utils import freq_to_midi, midi_to_name, is_in_guitar_range
from pickhero.audio.articulation import ArticulationDetector


@dataclass
class DetectedNote:
    """A single detected pitch event."""
    midi_note: int
    frequency: float
    confidence: float
    name: str
    is_onset: bool  # True if a new note strike was detected
    onset_sample: int | None = None  # absolute sample position of onset (from aubio get_last)
    articulation: str | None = None  # "hammer_on", "pull_off", "bend", "vibrato", "slide", "palm_mute", "harmonic"


class PitchDetector:
    """Real-time pitch and onset detection for guitar audio.

    Processes audio buffers (hop_size samples each) and returns
    detected notes with confidence values.
    """

    def __init__(
        self,
        buf_size: int = 2048,
        hop_size: int = 512,
        sample_rate: int = 44100,
        confidence_threshold: float = 0.8,
        onset_threshold: float = 0.3,
        noise_gate_db: float = -60.0,
        calibration: dict | None = None,
    ):
        self.buf_size = buf_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        self.onset_threshold = onset_threshold
        self.noise_gate_db = noise_gate_db
        self.last_signal_db: float = -120.0
        self.last_freq: float = 0.0
        self.last_confidence: float = 0.0

        # Octave jump protection
        self._prev_freq: float = 0.0
        self._freq_history: list[float] = []
        self._calibration = calibration

        # Pitch detector: YIN-fast is ~3x faster than yin with same accuracy.
        # Fall back to "yin" if the installed aubio doesn't have yinfast.
        try:
            self._pitch = aubio.pitch("yinfast", buf_size, hop_size, sample_rate)
        except (RuntimeError, ValueError):
            self._pitch = aubio.pitch("yin", buf_size, hop_size, sample_rate)
        self._pitch.set_unit("Hz")
        self._pitch.set_tolerance(confidence_threshold)

        # Onset detector
        self._onset = aubio.onset("default", buf_size, hop_size, sample_rate)
        self._onset.set_threshold(onset_threshold)

        # Articulation detector
        self._articulation = ArticulationDetector(sample_rate, hop_size, buf_size)
        self._articulation_frame: int = 0

    def process(self, audio_buffer: np.ndarray) -> DetectedNote | None:
        """Process a single audio buffer (hop_size float32 samples).

        Args:
            audio_buffer: 1D numpy array of float32 samples, length == hop_size.

        Returns:
            DetectedNote if a confident pitch was detected, None otherwise.
        """
        # Ensure correct format for aubio
        if audio_buffer.dtype != np.float32:
            audio_buffer = audio_buffer.astype(np.float32)

        # Check noise gate (RMS level)
        rms = np.sqrt(np.mean(audio_buffer ** 2))
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -120.0

        self.last_signal_db = db

        if db < self.noise_gate_db:
            return None

        # Detect pitch
        freq = float(self._pitch(audio_buffer)[0])
        confidence = float(self._pitch.get_confidence())

        # Correct octave jumps before exposing values
        if freq > 0:
            freq = self._correct_octave_jump(freq, confidence)

        # Store values for tuner (after octave correction)
        self.last_freq = freq
        self.last_confidence = confidence

        # Detect onset
        is_onset = bool(self._onset(audio_buffer))

        # Detect articulation (runs on every frame, using pitch + onset + audio)
        # Use a frame-relative timestamp (ms from detector start)
        articulation = self._articulation.process(
            freq, confidence, is_onset, audio_buffer,
            self._articulation_frame * self.hop_size / self.sample_rate * 1000.0,
        )
        self._articulation_frame += 1

        # Filter: need minimum confidence and valid frequency.
        # BUT: an onset is a timing event independent of pitch confidence.
        # If onset fires, return the note (with whatever pitch was detected) so
        # the Timing Judge can record the timing even if pitch is uncertain.
        if confidence < self.confidence_threshold or freq <= 0:
            if is_onset:
                # Low-confidence onset — return with best-effort pitch (may be 0)
                midi_note = freq_to_midi(freq) if freq > 0 else 0
                if midi_note == 0 or not is_in_guitar_range(midi_note):
                    midi_note = 0  # unknown pitch, but timing is still valid
                return DetectedNote(
                    midi_note=midi_note,
                    frequency=freq,
                    confidence=confidence,
                    name=midi_to_name(midi_note) if midi_note > 0 else "?",
                    is_onset=True,
                    onset_sample=self._onset.get_last(),
                    articulation=articulation,
                )
            return None

        midi_note = freq_to_midi(freq)
        if not is_in_guitar_range(midi_note):
            return None

        return DetectedNote(
            midi_note=midi_note,
            frequency=freq,
            confidence=confidence,
            name=midi_to_name(midi_note),
            is_onset=is_onset,
            onset_sample=self._onset.get_last() if is_onset else None,
            articulation=articulation,
        )

    def _correct_octave_jump(self, freq: float, confidence: float) -> float:
        """Suppress octave jumps caused by harmonic detection.

        If the new frequency is ~2x or ~0.5x the previous, and confidence
        isn't very high, prefer the previous frequency (likely the fundamental).
        When calibration data is available, also check if freq/2 matches a
        known open-string fundamental.
        """
        # Median filter: suppress single-frame octave jumps
        if len(self._freq_history) >= 3:
            valid = [f for f in self._freq_history if f > 0]
            if valid and freq > 0:
                median = sorted(valid)[len(valid) // 2]
                ratio = freq / median
                if ratio > 1.5:
                    freq = freq / 2.0
                elif ratio < 0.6:
                    freq = freq * 2.0

        # Update history (keep last 5)
        self._freq_history.append(freq)
        if len(self._freq_history) > 5:
            self._freq_history.pop(0)

        corrected = freq

        # Calibration-based correction: if freq/2 is near a calibrated string,
        # prefer freq/2 (the fundamental was likely the intended note)
        if self._calibration and freq > 0:
            cal_strings = self._calibration.get("strings", {})
            half_freq = freq / 2.0
            for cal in cal_strings.values():
                cal_freq = cal.get("frequency", 0)
                if cal_freq > 0:
                    ratio = half_freq / cal_freq
                    # Within ±1 semitone of a calibrated fundamental
                    if 0.944 < ratio < 1.06:
                        corrected = half_freq
                        self._prev_freq = corrected
                        return corrected

        # Generic ratio-based correction
        if self._prev_freq > 0 and confidence < 0.95:
            ratio = freq / self._prev_freq
            if 1.95 <= ratio <= 2.05:
                # One octave up — prefer previous (fundamental)
                corrected = self._prev_freq
            elif 0.48 <= ratio <= 0.52:
                # One octave down — prefer previous
                corrected = self._prev_freq

        self._prev_freq = corrected
        return corrected

    def set_noise_gate_db(self, db: float) -> None:
        """Update the noise gate threshold (dB). Takes effect on next process() call."""
        self.noise_gate_db = db

    def reset(self):
        """Reset detector state. Call when starting a new song/session."""
        self._prev_freq = 0.0
        self._freq_history = []
        self._articulation_frame = 0
        self._articulation.reset()

        # Re-create detectors to clear internal state
        try:
            self._pitch = aubio.pitch(
                "yinfast", self.buf_size, self.hop_size, self.sample_rate
            )
        except (RuntimeError, ValueError):
            self._pitch = aubio.pitch(
                "yin", self.buf_size, self.hop_size, self.sample_rate
            )
        self._pitch.set_unit("Hz")
        self._pitch.set_tolerance(self.confidence_threshold)

        self._onset = aubio.onset(
            "default", self.buf_size, self.hop_size, self.sample_rate
        )
        self._onset.set_threshold(self.onset_threshold)
