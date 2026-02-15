"""Pitch and onset detection using aubio.

Wraps aubio's YIN pitch detector and onset detector.
Processes audio buffers and returns detected notes.
"""

from dataclasses import dataclass

import aubio
import numpy as np

from pickhero.audio.note_utils import freq_to_midi, midi_to_name, is_in_guitar_range


@dataclass
class DetectedNote:
    """A single detected pitch event."""
    midi_note: int
    frequency: float
    confidence: float
    name: str
    is_onset: bool  # True if a new note strike was detected


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
    ):
        self.buf_size = buf_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        self.onset_threshold = onset_threshold
        self.noise_gate_db = noise_gate_db

        # Pitch detector (YIN algorithm)
        self._pitch = aubio.pitch("yin", buf_size, hop_size, sample_rate)
        self._pitch.set_unit("Hz")
        self._pitch.set_tolerance(confidence_threshold)

        # Onset detector
        self._onset = aubio.onset("default", buf_size, hop_size, sample_rate)
        self._onset.set_threshold(onset_threshold)

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

        if db < self.noise_gate_db:
            return None

        # Detect pitch
        freq = self._pitch(audio_buffer)[0]
        confidence = self._pitch.get_confidence()

        # Detect onset
        is_onset = bool(self._onset(audio_buffer))

        # Filter: need minimum confidence and valid frequency
        if confidence < self.confidence_threshold or freq <= 0:
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
        )

    def reset(self):
        """Reset detector state. Call when starting a new song/session."""
        # Re-create detectors to clear internal state
        self._pitch = aubio.pitch(
            "yin", self.buf_size, self.hop_size, self.sample_rate
        )
        self._pitch.set_unit("Hz")
        self._pitch.set_tolerance(self.confidence_threshold)

        self._onset = aubio.onset(
            "default", self.buf_size, self.hop_size, self.sample_rate
        )
        self._onset.set_threshold(self.onset_threshold)
