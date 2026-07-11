"""After-take analysis using Basic Pitch for polyphonic transcription.

Runs after song completion. Takes the raw recorded audio and produces:
- Polyphonic MIDI transcription (which notes were played)
- Pitch bend curves (for bend/vibrato analysis)
- Timing comparison against the tab

This replaces the real-time articulation detector for scoring.
Real-time articulation remains diagnostic-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OfflineResult:
    """Result of after-take offline analysis.

    Attributes:
        midi_notes: Transcribed MIDI note numbers.
        timing_errors_ms: Per-note timing deviations in milliseconds.
        pitch_bend_curves: Per-MIDI-note pitch bend contour (cents over time).
        technique_verdicts: Technique verdicts from offline classification.
    """

    midi_notes: list[int] = field(default_factory=list)
    timing_errors_ms: list[float] = field(default_factory=list)
    pitch_bend_curves: dict[int, list[float]] = field(default_factory=dict)
    technique_verdicts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize result to a plain dict."""
        return {
            "midi_notes": self.midi_notes,
            "timing_errors_ms": self.timing_errors_ms,
            "pitch_bend_curves": {k: list(v) for k, v in self.pitch_bend_curves.items()},
            "technique_verdicts": len(self.technique_verdicts),
        }


class OfflineAnalyzer:
    """After-take polyphonic analysis.

    Uses Basic Pitch (optional) for polyphonic transcription.
    Falls back to monophonic analysis if basic_pitch is not installed.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._basic_pitch_available = self._check_basic_pitch()

    @staticmethod
    def _check_basic_pitch() -> bool:
        try:
            import basic_pitch  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def is_full_analysis_available(self) -> bool:
        """Whether Basic Pitch is installed for polyphonic transcription."""
        return self._basic_pitch_available

    def analyze(
        self,
        raw_audio: np.ndarray,
        matched_pairs: list | None = None,
    ) -> list:
        """Run after-take analysis on recorded audio.

        Args:
            raw_audio: Mono float32 audio from take recording.
            matched_pairs: Optional list of (NoteEvent, PerformanceEvent) pairs.

        Returns:
            List of TechniqueVerdict from offline analysis.
        """
        if len(raw_audio) == 0:
            return []

        verdicts: list = []

        if self._basic_pitch_available:
            verdicts.extend(
                self._analyze_with_basic_pitch(raw_audio, matched_pairs or [])
            )
        else:
            # Fallback: use existing PerformanceAnalyzer results
            # (no additional verdicts without Basic Pitch).
            pass

        return verdicts

    def _analyze_with_basic_pitch(
        self,
        raw_audio: np.ndarray,
        matched_pairs: list,
    ) -> list:
        """Run Basic Pitch polyphonic transcription.

        Scaffold -- actual implementation requires the ``basic_pitch`` package
        and model inference. Returns empty until Basic Pitch is available
        and properly configured.
        """
        # TODO: implement polyphonic MIDI transcription via basic_pitch.infer()
        return []

    def transcribe(self, raw_audio: np.ndarray) -> OfflineResult:
        """Transcribe audio to MIDI notes + pitch bend curves.

        Returns a populated :class:`OfflineResult` with MIDI note numbers,
        timing errors, and pitch bend curves if Basic Pitch is available,
        otherwise an empty result.

        This is a scaffold -- the real implementation uses Basic Pitch for
        full polyphonic transcription.
        """
        result = OfflineResult()
        if len(raw_audio) == 0:
            return result

        # Basic transcription using onset detection + pitch tracking.
        # This is a scaffold -- real implementation uses Basic Pitch.
        result.midi_notes = []
        result.timing_errors_ms = []
        result.pitch_bend_curves = {}

        return result
