"""Composite verifier combining spectral/CQT and technique verification.

This is the verifier implementation wired into :class:`~pickhero.matcher.NoteMatcher`.
It dispatches single-note and chord verification to a spectral/CQT backend
and technique verification to :class:`~pickhero.audio.verifier_technique.TechniqueVerifier`.
"""

from __future__ import annotations

import numpy as np

from pickhero.audio.evidence import (
    ChordVerification,
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
        """Verify a single note using only the spectral verifier.

        CQT is intentionally not used as a fallback because it lacks the
        cents, anti-alias, fundamental, and onset checks that the spectral
        verifier applies. Allowing it to run as a fallback re-accepts notes
        that spectral correctly rejected as aliases.
        """
        return self._spectral.verify_single_note(
            audio_window, expected_midi, mode,
            expected_onset_offset_ms=expected_onset_offset_ms,
            onset_tolerance_ms=onset_tolerance_ms,
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
