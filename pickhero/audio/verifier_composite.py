"""Composite verifier combining spectral/CQT and technique verification.

This is the verifier implementation wired into :class:`~pickhero.matcher.NoteMatcher`.
It dispatches single-note and chord verification to a spectral/CQT backend
and technique verification to :class:`~pickhero.audio.verifier_technique.TechniqueVerifier`.
"""

from __future__ import annotations

import numpy as np

from pickhero.audio.evidence import (
    ChordVerification,
    ExpectedNote,
    NoteVerification,
    TechniqueVerification,
)
from pickhero.audio.match_mode import MatchMode
from pickhero.audio.verifier import ExpectedEventVerifier
from pickhero.audio.verifier_cqt import CQTVerifier
from pickhero.audio.verifier_spectral import SpectralVerifier
from pickhero.audio.verifier_technique import TechniqueVerifier


class CompositeVerifier(ExpectedEventVerifier):
    """Expected-event verifier using spectral/CQT evidence + technique checks."""

    def __init__(
        self,
        sample_rate: int = 48000,
        use_cqt_for_chords: bool = True,
        fft_size: int = 8192,
    ):
        self.sample_rate = sample_rate
        self._spectral = SpectralVerifier(sample_rate=sample_rate)
        self._cqt = CQTVerifier(sample_rate=sample_rate, fft_size=fft_size)
        self._technique = TechniqueVerifier(sample_rate=sample_rate)
        self._use_cqt_for_chords = use_cqt_for_chords

    def verify_single_note(
        self,
        audio_window: np.ndarray,
        expected_midi: int,
        mode: MatchMode,
        expected_onset_offset_ms: float | None = None,
        onset_tolerance_ms: float | None = None,
    ) -> NoteVerification:
        """Fuse cents-aware spectral and multi-resolution evidence.

        The fixed spectral verifier is precise around a known fundamental; the
        log-frequency front end is better at low notes and octave ambiguity.
        Neither backend can unilaterally manufacture a Judge hit: disagreement
        requires strong, low-alias evidence from the recovering backend.
        """
        spectral = self._spectral.verify_single_note(
            audio_window, expected_midi, mode,
            expected_onset_offset_ms=expected_onset_offset_ms,
            onset_tolerance_ms=onset_tolerance_ms,
        )
        log_result = self._cqt.verify_single_note(
            audio_window, expected_midi, mode,
            expected_onset_offset_ms=expected_onset_offset_ms,
            onset_tolerance_ms=onset_tolerance_ms,
        )

        spectral_conf = spectral.confidence
        log_conf = log_result.confidence
        if spectral.is_pitch_present and log_result.is_pitch_present:
            present = True
            confidence = min(1.0, spectral_conf * 0.58 + log_conf * 0.42 + 0.08)
        elif spectral.is_pitch_present:
            present = (
                mode == MatchMode.ARCADE
                or spectral_conf >= 0.58
                or spectral.alias_risk <= 0.34
            )
            confidence = spectral_conf * 0.88
        elif log_result.is_pitch_present:
            recovery_threshold = {
                MatchMode.ARCADE: 0.50,
                MatchMode.PRACTICE: 0.64,
                MatchMode.JUDGE: 0.76,
            }.get(mode, 0.64)
            present = log_conf >= recovery_threshold and log_result.alias_risk <= 0.46
            confidence = log_conf * (0.90 if present else 0.62)
        else:
            present = False
            confidence = max(spectral_conf, log_conf) * 0.55

        pitch_evidence = spectral.pitch_evidence or log_result.pitch_evidence
        if pitch_evidence is not None:
            from pickhero.audio.evidence import PitchEvidence
            sources = []
            if spectral.pitch_evidence is not None:
                sources.append(spectral.pitch_evidence.source)
            if log_result.pitch_evidence is not None:
                sources.append(log_result.pitch_evidence.source)
            cents = (
                spectral.pitch_evidence.cents_error
                if spectral.pitch_evidence is not None
                and spectral.pitch_evidence.cents_error is not None
                else log_result.pitch_evidence.cents_error
                if log_result.pitch_evidence is not None
                else None
            )
            pitch_evidence = PitchEvidence(
                midi_note=expected_midi,
                cents_error=cents,
                confidence=max(0.0, min(1.0, confidence)),
                source="+".join(dict.fromkeys(sources)) or "composite",
            )

        onset_ms = spectral.onset_ms if spectral.onset_ms is not None else log_result.onset_ms
        timing_error = (
            spectral.timing_error_ms
            if spectral.timing_error_ms is not None
            else log_result.timing_error_ms
        )
        if spectral.pitch_evidence is not None and log_result.pitch_evidence is not None:
            alias_risk = min(spectral.alias_risk, log_result.alias_risk)
        elif spectral.pitch_evidence is not None:
            alias_risk = spectral.alias_risk
        else:
            alias_risk = log_result.alias_risk
        return NoteVerification(
            is_pitch_present=present,
            is_onset_present=spectral.is_onset_present or log_result.is_onset_present,
            pitch_evidence=pitch_evidence,
            onset_ms=onset_ms,
            harmonic_score=max(spectral.harmonic_score, log_result.harmonic_score),
            timing_error_ms=timing_error,
            alias_risk=alias_risk,
        )

    def verify_chord(
        self,
        audio_window: np.ndarray,
        expected_notes: list[ExpectedNote],
        mode: MatchMode,
        expected_onset_offset_ms: float | None = None,
        onset_tolerance_ms: float | None = None,
    ) -> ChordVerification:
        """Verify a chord using CQT or spectral backend."""
        if self._use_cqt_for_chords:
            return self._cqt.verify_chord(
                audio_window, expected_notes, mode,
                expected_onset_offset_ms=expected_onset_offset_ms,
                onset_tolerance_ms=onset_tolerance_ms,
            )
        return self._spectral.verify_chord(
            audio_window, expected_notes, mode,
            expected_onset_offset_ms=expected_onset_offset_ms,
            onset_tolerance_ms=onset_tolerance_ms,
        )

    def verify_technique(
        self,
        audio_window: np.ndarray,
        expected: str,
        context: dict,
    ) -> TechniqueVerification:
        """Verify a technique using the dedicated technique verifier."""
        return self._technique.verify(audio_window, expected, context)

    def verify_silence(self, audio_window: np.ndarray, threshold_db: float) -> bool:
        """True if the window is silent."""
        if len(audio_window) == 0:
            return True
        rms = float(np.sqrt(np.mean(audio_window**2)))
        db = 20.0 * np.log10(rms + 1e-10)
        return db < threshold_db
