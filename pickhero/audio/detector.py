"""Pitch and onset detection using aubio.

Wraps aubio's YIN pitch detector and onset detector.
Processes audio buffers and returns detected notes.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import aubio
import numpy as np

from pickhero.audio.note_utils import freq_to_midi, midi_to_name, is_in_guitar_range
from pickhero.audio.articulation import ArticulationDetector

if TYPE_CHECKING:
    from pickhero.audio.event_types import EventKindSnapshot
    from pickhero.audio.performance import PerformanceEvent


@dataclass
class DetectedNote:
    """A single detected pitch event."""
    midi_note: int
    frequency: float
    confidence: float
    name: str
    is_onset: bool  # True if a new note strike was detected
    onset_sample: int | None = None  # absolute sample position of onset (from aubio get_last)
    performance: "PerformanceEvent | None" = None  # real-time per-note performance record (f0 curve, candidates)
    event_snapshot: "EventKindSnapshot | None" = None  # immutable snapshot from stabilizer


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

        # Pre-gate: noise transient rejection (spectral flatness + refractory)
        self._noise_rejected_at_ms: float | None = None

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

        # Detect onset FIRST — pick attacks are broadband (plectrum hit
        # produces a wideband burst) but are real musical events. The
        # spectral-flatness pre-gate must only reject non-onset noise, never
        # a pick attack.
        is_onset = bool(self._onset(audio_buffer))

        # Spectral-flatness pre-gate: reject broadband noise transients
        # (jack touch, cable bump, RF buzz) on NON-ONSET frames only.
        # Onset frames always pass — the stabilizer handles false-positive
        # onsets via confidence gating and consensus.
        #
        # Only apply to LOUD frames — silence has high flatness but isn't
        # a noise transient. Without this floor, silence starts a refractory
        # that blocks the next legitimate note's sustain frames.
        _SF_NOISE = 0.45
        _SF_RMS_FLOOR_DB = -40.0
        _REFRACTORY_MS = 60.0

        frame_ms = self._articulation_frame * self.hop_size / self.sample_rate * 1000.0

        if not is_onset:
            if self._noise_rejected_at_ms is not None:
                if frame_ms - self._noise_rejected_at_ms < _REFRACTORY_MS:
                    return None  # still in refractory after a noise transient
                self._noise_rejected_at_ms = None

            if db > _SF_RMS_FLOOR_DB:
                flatness = self._spectral_flatness(audio_buffer)
                if flatness > _SF_NOISE:
                    self._noise_rejected_at_ms = frame_ms
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

        # Detect articulation (runs on every frame, using pitch + onset + audio)
        # Use a frame-relative timestamp (ms from detector start)
        timestamp_ms = self._articulation_frame * self.hop_size / self.sample_rate * 1000.0
        self._articulation.process(
            freq, confidence, is_onset, audio_buffer, timestamp_ms,
        )
        self._articulation_frame += 1

        # The active PerformanceEvent is held by the articulation detector while
        # a note is sounding; it is closed and drained on the next onset. Attach
        # the in-progress event to the DetectedNote so the matcher can later
        # resolve which NoteEvent it corresponds to.
        active_event = self._articulation.active_event

        # Filter: need minimum confidence and valid frequency.
        # Onsets are timing events independent of pitch confidence — YIN often
        # hasn't locked during the attack transient, so confidence can be low.
        # But a pitchless onset (freq=0, midi=0) or one outside guitar range is
        # noise, not a real pick. Reject those; forward everything else.
        if confidence < self.confidence_threshold or freq <= 0:
            if is_onset:
                midi_note = freq_to_midi(freq) if freq > 0 else 0
                if midi_note == 0 or not is_in_guitar_range(midi_note):
                    return None
                return DetectedNote(
                    midi_note=midi_note,
                    frequency=freq,
                    confidence=confidence,
                    name=midi_to_name(midi_note),
                    is_onset=True,
                    onset_sample=self._onset.get_last(),
                    performance=active_event,
                )
            return None

        midi_note = freq_to_midi(freq)
        if not is_in_guitar_range(midi_note):
            return None

        # Keep the active event's midi_note / confidence in sync so the
        # matcher and analyzer see the final pitch decision.
        if active_event is not None:
            active_event.midi_note = midi_note
            active_event.confidence = confidence

        return DetectedNote(
            midi_note=midi_note,
            frequency=freq,
            confidence=confidence,
            name=midi_to_name(midi_note),
            is_onset=is_onset,
            onset_sample=self._onset.get_last() if is_onset else None,
            performance=active_event,
        )

    def _spectral_flatness(self, audio: np.ndarray) -> float:
        """Wiener entropy — geometric/arithmetic mean of spectrum.

        Tonal (guitar note): < 0.25. Broadband noise (jack touch): > 0.45.
        Used by the pre-gate to reject noise transients before onset/pitch
        detection, preventing phantom notes from electrical interference.
        """
        n = len(audio)
        windowed = audio * np.hanning(n)
        spec = np.abs(np.fft.rfft(windowed))
        spec = spec[1:]  # skip DC
        if len(spec) == 0 or np.all(spec < 1e-12):
            return 1.0
        eps = 1e-12
        geo = np.exp(np.mean(np.log(np.maximum(spec, eps))))
        ari = np.mean(spec)
        return float(geo / ari) if ari > eps else 1.0

    def drain_events(self) -> list["PerformanceEvent"]:
        """Drain PerformanceEvents completed since the last call.

        A PerformanceEvent is *completed* when the next onset fires (the
        previous note's release). The articulation detector owns the list;
        this is the thread-safe drain point for the worker / audio callback.
        """
        return self._articulation.drain_completed()

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
                valid_sorted = sorted(valid)
                n = len(valid_sorted)
                # True median: average of two middle values for even-length lists
                if n % 2 == 1:
                    median = valid_sorted[n // 2]
                else:
                    median = (valid_sorted[n // 2 - 1] + valid_sorted[n // 2]) / 2.0
                ratio = freq / median
                if ratio > 3.0:
                    freq = freq / 2.0
                elif ratio < 0.25:
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

        # Generic ratio-based correction. Only snap when the previous frequency
        # was stable (history agrees with it) AND confidence is not very high —
        # a strong 2nd harmonic can report high confidence, but a single-frame
        # _prev_freq from a transient sub-harmonic must not drag the next
        # confident reading down an octave.
        if self._prev_freq > 0 and len(self._freq_history) >= 3:
            recent = [f for f in self._freq_history[-3:] if f > 0]
            if recent and all(abs(f - self._prev_freq) / self._prev_freq < 0.05
                               for f in recent):
                ratio = freq / self._prev_freq
                if 1.95 <= ratio <= 2.05 and confidence < 0.95:
                    # One octave up — prefer the stable previous (fundamental)
                    corrected = self._prev_freq
                elif 0.48 <= ratio <= 0.52 and confidence < 0.95:
                    # One octave down — prefer previous
                    corrected = self._prev_freq

        self._prev_freq = corrected
        return corrected

    def set_noise_gate_db(self, db: float) -> None:
        """Update the noise gate threshold (dB). Takes effect on next process() call."""
        self.noise_gate_db = db

    def get_onset_delay(self) -> int:
        """Return the onset detector's algorithmic delay in samples.

        Exposed so callers (AudioCapture) don't reach into ``_onset`` directly.
        """
        return int(self._onset.get_delay())

    def reset(self):
        """Reset detector state. Call when starting a new song/session."""
        self._prev_freq = 0.0
        self._freq_history = []
        self._articulation_frame = 0
        self._articulation.reset()
        # Clear exposed tuner/signal state so stale readings don't persist
        # after a reset until the next process() call overwrites them.
        self.last_signal_db = -120.0
        self.last_freq = 0.0
        self.last_confidence = 0.0

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
