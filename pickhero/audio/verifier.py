"""Expected-event verifier protocol.

The verifier answers "did the expected note/technique happen in this audio
window?" rather than "what note did I hear?".  Concrete implementations
receive a raw audio window and an expected MIDI note or technique, then
return structured evidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from pickhero.audio.evidence import ExpectedNote

if TYPE_CHECKING:
    import numpy as np

    from pickhero.audio.evidence import (
        ChordVerification,
        ExpectedNote,
        NoteVerification,
        TechniqueVerification,
        VerificationResult,
    )
    from pickhero.audio.match_mode import MatchMode


class ExpectedEventVerifier(ABC):
    """Protocol for expected-event verification.

    Implementations must not mutate the supplied ``audio_window``.
    """

    @abstractmethod
    def verify_single_note(
        self,
        audio_window: np.ndarray,
        expected_midi: int,
        mode: MatchMode,
        expected_onset_offset_ms: float | None = None,
        onset_tolerance_ms: float | None = None,
    ) -> NoteVerification:
        """Verify whether a single expected note is present in the window."""
        ...

    @abstractmethod
    def verify_chord(
        self,
        audio_window: np.ndarray,
        expected_notes: list[ExpectedNote],
        mode: MatchMode,
        expected_onset_offset_ms: float | None = None,
        onset_tolerance_ms: float | None = None,
    ) -> ChordVerification:
        """Verify whether an expected chord is present in the window."""
        ...

    @abstractmethod
    def verify_technique(
        self,
        audio_window: np.ndarray,
        expected: str,
        context: dict,
    ) -> TechniqueVerification:
        """Verify whether an expected technique is present in the window."""
        ...

    @abstractmethod
    def verify_silence(self, audio_window: np.ndarray, threshold_db: float) -> bool:
        """Return True if the window is silent below ``threshold_db``."""
        ...

    def verify(
        self,
        audio_window: np.ndarray,
        expected_midi: set[int],
        expected_techniques: list[str],
        mode: MatchMode,
        timestamp_ms: float,
        technique_context: dict | None = None,
        expected_onset_offset_ms: float | None = None,
        onset_tolerance_ms: float | None = None,
    ) -> VerificationResult:
        """Convenience wrapper that verifies notes/chord + techniques.

        If ``expected_midi`` has one element, returns a ``NoteVerification``;
        otherwise returns a ``ChordVerification``.
        """
        # Copy before use — don't mutate the caller's set.
        expected = set(expected_midi)
        if len(expected) == 1:
            midi = next(iter(expected))
            verified: NoteVerification | ChordVerification = self.verify_single_note(
                audio_window, midi, mode,
                expected_onset_offset_ms=expected_onset_offset_ms,
                onset_tolerance_ms=onset_tolerance_ms,
            )
        else:
            # Convert to ExpectedNote list for ordered verification.
            expected_notes = [
                ExpectedNote(midi=m, event_id=f"verify:{i}")
                for i, m in enumerate(sorted(expected))
            ]
            verified = self.verify_chord(
                audio_window, expected_notes, mode,
                expected_onset_offset_ms=expected_onset_offset_ms,
                onset_tolerance_ms=onset_tolerance_ms,
            )

        techniques: list[TechniqueVerification] = []
        ctx = technique_context or {}
        for technique in expected_techniques:
            techniques.append(self.verify_technique(audio_window, technique, ctx))

        from pickhero.audio.evidence import VerificationResult

        return VerificationResult(
            expected_midi=expected,
            expected_techniques=expected_techniques,
            verified=verified,
            techniques=techniques,
            timestamp_ms=timestamp_ms,
        )
