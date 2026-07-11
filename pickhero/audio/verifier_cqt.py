"""Multi-resolution log-frequency verifier for notes and chords.

The public class name remains ``CQTVerifier`` for compatibility, but the old
single-FFT interpolation has been replaced by a harmonic-sieve front end with
multiple analysis resolutions.  Low guitar/bass fundamentals receive long
windows, high notes retain shorter windows, and chord pitch classes are derived
from fundamental hypotheses instead of raw overtone bins.
"""

from __future__ import annotations

import numpy as np

from pickhero.audio.evidence import (
    ChordVerification,
    ExpectedNote,
    NoteVerification,
    PitchEvidence,
    TechniqueVerification,
)
from pickhero.audio.log_frequency import MultiResolutionLogSpectrum
from pickhero.audio.match_mode import MatchMode
from pickhero.audio.verifier import ExpectedEventVerifier


_THRESHOLDS = {
    MatchMode.ARCADE: 0.24,
    MatchMode.PRACTICE: 0.34,
    MatchMode.JUDGE: 0.44,
}

_EXPECTED_PC_THRESHOLDS = {
    MatchMode.ARCADE: 0.24,
    MatchMode.PRACTICE: 0.31,
    MatchMode.JUDGE: 0.38,
}

_EXTRA_PC_THRESHOLDS = {
    MatchMode.ARCADE: 0.70,
    MatchMode.PRACTICE: 0.76,
    MatchMode.JUDGE: 0.82,
}


class CQTVerifier(ExpectedEventVerifier):
    """Expected-event verifier backed by multi-resolution log-frequency evidence."""

    def __init__(self, sample_rate: int = 48000, fft_size: int = 8192):
        self.sample_rate = int(sample_rate)
        self.fft_size = max(4096, int(fft_size))
        sizes = tuple(sorted({4096, 8192, max(16384, self.fft_size)}))
        self._front_end = MultiResolutionLogSpectrum(
            sample_rate=self.sample_rate,
            min_midi=24,
            max_midi=108,
            fft_sizes=sizes,
        )

    def _detect_strum_onset(
        self,
        audio_window: np.ndarray,
        expected_onset_offset_ms: float | None = None,
        onset_tolerance_ms: float | None = None,
    ) -> tuple[bool, float | None]:
        """Return ``(is_onset, onset_ms)`` using robust short-term energy flux."""
        if len(audio_window) < int(self.sample_rate * 0.06):
            return False, None
        hop = max(1, int(self.sample_rate * 0.005))
        win = max(hop * 2, int(self.sample_rate * 0.015))
        rms_values: list[float] = []
        for start in range(0, len(audio_window) - win + 1, hop):
            frame = np.asarray(audio_window[start:start + win], dtype=np.float32)
            rms_values.append(float(np.sqrt(np.mean(frame * frame))))
        if len(rms_values) < 4:
            return False, None

        rms = np.asarray(rms_values, dtype=np.float32)
        # Compare against a short running baseline instead of the immediately
        # preceding frame; slow compressor release must not look like an onset.
        baseline = np.empty_like(rms)
        baseline[0] = rms[0]
        for i in range(1, len(rms)):
            baseline[i] = 0.82 * baseline[i - 1] + 0.18 * rms[i - 1]
        flux = np.maximum(0.0, rms - baseline)
        peak_idx = int(np.argmax(flux))
        peak_flux = float(flux[peak_idx])
        peak_rms = float(np.max(rms))
        noise = float(np.median(rms[: max(1, len(rms) // 4)]))
        relative = peak_flux / max(peak_rms, 1e-9)
        dynamic = peak_rms / max(noise, 1e-9)

        hop_ms = hop / self.sample_rate * 1000.0
        onset_ms = peak_idx * hop_ms
        strong = relative >= 0.075 and (dynamic >= 1.35 or relative >= 0.25)
        if expected_onset_offset_ms is not None:
            tolerance = onset_tolerance_ms or 50.0
            strong = strong and abs(onset_ms - expected_onset_offset_ms) <= tolerance
        else:
            strong = strong and peak_idx <= int(len(flux) * 0.65)
        return strong, onset_ms if strong else None

    def verify_single_note(
        self,
        audio_window: np.ndarray,
        expected_midi: int,
        mode: MatchMode,
        expected_onset_offset_ms: float | None = None,
        onset_tolerance_ms: float | None = None,
    ) -> NoteVerification:
        result = self.verify_chord(
            audio_window,
            [ExpectedNote(midi=expected_midi)],
            mode,
            expected_onset_offset_ms=expected_onset_offset_ms,
            onset_tolerance_ms=onset_tolerance_ms,
        )
        if result.notes:
            return result.notes[0]
        return NoteVerification(
            is_pitch_present=False,
            is_onset_present=False,
            pitch_evidence=None,
            onset_ms=None,
            harmonic_score=0.0,
            timing_error_ms=None,
            alias_risk=0.0,
        )

    def verify_chord(
        self,
        audio_window: np.ndarray,
        expected_notes: list[ExpectedNote],
        mode: MatchMode,
        expected_onset_offset_ms: float | None = None,
        onset_tolerance_ms: float | None = None,
    ) -> ChordVerification:
        if len(audio_window) == 0 or not expected_notes:
            return ChordVerification(
                notes=[], partial=False, total_harmonic_energy=0.0
            )

        expected_midis = {note.midi for note in expected_notes}
        frame = self._front_end.analyse(
            audio_window,
            top_k=max(6, len(expected_midis) * 2),
            prior_midis=expected_midis,
        )
        threshold = _THRESHOLDS.get(mode, _THRESHOLDS[MatchMode.PRACTICE])
        is_onset, onset_ms = self._detect_strum_onset(
            audio_window,
            expected_onset_offset_ms=expected_onset_offset_ms,
            onset_tolerance_ms=onset_tolerance_ms,
        )

        notes: list[NoteVerification] = []
        for expected in expected_notes:
            score = frame.score_for_midi(expected.midi)
            harmonic = float(frame.harmonic_support.get(expected.midi, 0.0))
            risk = frame.alias_risk_for_midi(expected.midi)
            # In Judge mode an octave-ambiguous candidate needs more direct
            # harmonic support.  Arcade remains forgiving of missing F0 energy.
            effective = score * (1.0 - (0.18 if mode == MatchMode.JUDGE else 0.08) * risk)
            if mode == MatchMode.ARCADE:
                present = effective >= threshold
            else:
                present = effective >= threshold and harmonic >= threshold * 0.58
            hypothesis = min(
                frame.hypotheses,
                key=lambda item: abs(item.midi - expected.midi),
                default=None,
            )
            cents = (
                hypothesis.cents_error
                if hypothesis is not None and hypothesis.midi == expected.midi
                else None
            )
            notes.append(NoteVerification(
                is_pitch_present=present,
                is_onset_present=is_onset,
                pitch_evidence=PitchEvidence(
                    midi_note=expected.midi,
                    cents_error=cents,
                    confidence=max(0.0, min(1.0, effective)),
                    source="multi_resolution_log_frequency",
                ),
                onset_ms=onset_ms,
                harmonic_score=harmonic,
                timing_error_ms=(
                    onset_ms - expected_onset_offset_ms
                    if onset_ms is not None and expected_onset_offset_ms is not None
                    else None
                ),
                alias_risk=risk,
            ))

        expected_pitch_classes = {note.midi % 12 for note in expected_notes}
        expected_threshold = _EXPECTED_PC_THRESHOLDS.get(
            mode, _EXPECTED_PC_THRESHOLDS[MatchMode.PRACTICE]
        )
        extra_threshold = _EXTRA_PC_THRESHOLDS.get(
            mode, _EXTRA_PC_THRESHOLDS[MatchMode.PRACTICE]
        )
        observed_pitch_classes = frozenset(
            pc for pc, energy in frame.pitch_class_energy.items()
            if (
                pc in expected_pitch_classes and energy >= expected_threshold
            ) or (
                pc not in expected_pitch_classes and energy >= extra_threshold
            )
        )

        partial = any(note.is_pitch_present for note in notes) and not all(
            note.is_pitch_present for note in notes
        )
        return ChordVerification(
            notes=notes,
            partial=partial,
            total_harmonic_energy=float(sum(frame.midi_scores.values())),
            observed_pitch_classes=observed_pitch_classes,
            pitch_class_energy=dict(frame.pitch_class_energy),
        )

    def verify_technique(
        self,
        audio_window: np.ndarray,
        expected: str,
        context: dict,
    ) -> TechniqueVerification:
        return TechniqueVerification(
            technique=expected,
            is_present=False,
            confidence=0.0,
            details={},
            uncertain=True,
        )

    def verify_silence(self, audio_window: np.ndarray, threshold_db: float) -> bool:
        if len(audio_window) == 0:
            return True
        audio = np.asarray(audio_window, dtype=np.float32)
        rms = float(np.sqrt(np.mean(audio * audio)))
        db = 20.0 * np.log10(rms + 1e-10)
        return db < threshold_db
