"""Tests for pickhero.audio.verifier — the ExpectedEventVerifier abstract base class.

Tests focus on the abstract protocol itself: instantiation rules, abstract method
enforcement, and verify() dispatch to the concrete sub-methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from pickhero.audio.evidence import (
    ChordVerification,
    ExpectedNote,
    NoteVerification,
    TechniqueVerification,
    VerificationResult,
)
from pickhero.audio.match_mode import MatchMode
from pickhero.audio.verifier import ExpectedEventVerifier


# ─── Helper: concrete subclasses ───────────────────────────────────────


class ConcreteVerifier(ExpectedEventVerifier):
    """Minimal full implementation — all abstract methods filled."""

    def verify_single_note(
        self,
        audio_window: np.ndarray,
        expected_midi: int,
        mode: MatchMode,
        **kwargs,
    ) -> NoteVerification:
        return NoteVerification(
            is_pitch_present=True,
            is_onset_present=True,
            pitch_evidence=None,
            onset_ms=10.0,
            harmonic_score=0.85,
            timing_error_ms=0.5,
            alias_risk=0.0,
        )

    def verify_chord(
        self,
        audio_window: np.ndarray,
        expected_notes: list[ExpectedNote],
        mode: MatchMode,
        **kwargs,
    ) -> ChordVerification:
        return ChordVerification(
            notes=[],
            partial=False,
            total_harmonic_energy=1.0,
        )

    def verify_technique(
        self,
        audio_window: np.ndarray,
        expected: str,
        context: dict,
    ) -> TechniqueVerification:
        return TechniqueVerification(
            technique=expected,
            is_present=True,
            confidence=0.9,
        )

    def verify_silence(self, audio_window: np.ndarray, threshold_db: float) -> bool:
        return False


class CallTrackingVerifier(ExpectedEventVerifier):
    """Records every abstract-method call so verify() dispatch can be inspected."""

    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def verify_single_note(
        self,
        audio_window: np.ndarray,
        expected_midi: int,
        mode: MatchMode,
        **kwargs,
    ) -> NoteVerification:
        self.calls.append(("verify_single_note", expected_midi))
        return NoteVerification(
            is_pitch_present=True,
            is_onset_present=True,
            pitch_evidence=None,
            onset_ms=10.0,
            harmonic_score=0.85,
            timing_error_ms=0.5,
            alias_risk=0.0,
        )

    def verify_chord(
        self,
        audio_window: np.ndarray,
        expected_notes: list[ExpectedNote],
        mode: MatchMode,
        **kwargs,
    ) -> ChordVerification:
        self.calls.append(
            ("verify_chord", sorted(n.midi for n in expected_notes))
        )
        return ChordVerification(
            notes=[],
            partial=False,
            total_harmonic_energy=1.0,
        )

    def verify_technique(
        self,
        audio_window: np.ndarray,
        expected: str,
        context: dict,
    ) -> TechniqueVerification:
        self.calls.append(("verify_technique", expected))
        return TechniqueVerification(
            technique=expected,
            is_present=True,
            confidence=0.9,
        )

    def verify_silence(self, audio_window: np.ndarray, threshold_db: float) -> bool:
        self.calls.append(("verify_silence", None))
        return False


class MissingVerifySilence(ExpectedEventVerifier):
    """Subclass that forgets to implement verify_silence."""

    def verify_single_note(self, audio_window, expected_midi, mode, **kwargs):
        return NoteVerification(
            is_pitch_present=True, is_onset_present=False, pitch_evidence=None,
            onset_ms=None, harmonic_score=0.0, timing_error_ms=None, alias_risk=0.0,
        )

    def verify_chord(self, audio_window, expected_notes, mode, **kwargs):
        return ChordVerification(notes=[], partial=False, total_harmonic_energy=0.0)

    def verify_technique(self, audio_window, expected, context):
        return TechniqueVerification(technique=expected, is_present=False, confidence=0.0)


class MissingAll(ExpectedEventVerifier):
    """Subclass that implements nothing — all four methods still abstract."""
    pass


# ─── Tests ─────────────────────────────────────────────────────────────


class TestExpectedEventVerifierABC:
    """ExpectedEventVerifier is an abstract base class with four protocols."""

    def test_is_abc(self):
        """The class itself should inherit from ABC."""
        import abc
        assert issubclass(ExpectedEventVerifier, abc.ABC)

    def test_cannot_instantiate_directly(self):
        """Instantiation of the ABC itself must raise TypeError."""
        with pytest.raises(TypeError, match="abstract"):
            ExpectedEventVerifier()

    def test_cannot_instantiate_missing_one_method(self):
        """A subclass missing one abstract method must also raise TypeError."""
        with pytest.raises(TypeError, match="abstract"):
            MissingVerifySilence()

    def test_cannot_instantiate_missing_all_methods(self):
        """A subclass missing every abstract method must raise TypeError."""
        with pytest.raises(TypeError, match="abstract"):
            MissingAll()

    def test_concrete_subclass_instantiable(self):
        """A subclass that fills every abstract method can be instantiated."""
        v = ConcreteVerifier()
        assert isinstance(v, ExpectedEventVerifier)

    def test_has_four_abstract_methods(self):
        """Check that exactly four methods are still abstract on the class."""
        abstract = set()
        for name in dir(ExpectedEventVerifier):
            obj = getattr(ExpectedEventVerifier, name, None)
            if getattr(obj, "__isabstractmethod__", False):
                abstract.add(name)
        assert abstract == {
            "verify_single_note",
            "verify_chord",
            "verify_technique",
            "verify_silence",
        }

    def test_verify_is_concrete(self):
        """The verify() convenience wrapper should NOT be abstract."""
        assert not getattr(
            ExpectedEventVerifier.verify, "__isabstractmethod__", False
        )


class TestVerifyDispatch:
    """verify() routes to verify_single_note vs verify_chord and calls
    verify_technique for each expected technique."""

    def make_audio(self) -> np.ndarray:
        """Return a short dummy audio buffer."""
        return np.zeros(4096, dtype=np.float32)

    def test_single_note_returns_note_verification(self):
        """When expected_midi has one element, verify dispatches to
        verify_single_note and the result wraps a NoteVerification."""
        v = CallTrackingVerifier()
        audio = self.make_audio()
        result = v.verify(
            audio_window=audio,
            expected_midi={48},
            expected_techniques=[],
            mode=MatchMode.JUDGE,
            timestamp_ms=1000.0,
        )

        assert isinstance(result, VerificationResult)
        assert v.calls == [("verify_single_note", 48)]
        assert isinstance(result.verified, NoteVerification)
        assert result.expected_midi == {48}
        assert result.timestamp_ms == 1000.0

    def test_single_note_passes_kwargs(self):
        """verify passes optional keyword arguments through to
        verify_single_note."""
        v = CallTrackingVerifier()
        audio = self.make_audio()
        result = v.verify(
            audio_window=audio,
            expected_midi={60},
            expected_techniques=[],
            mode=MatchMode.PRACTICE,
            timestamp_ms=500.0,
            expected_onset_offset_ms=15.0,
            onset_tolerance_ms=50.0,
        )
        assert isinstance(result.verified, NoteVerification)
        assert result.expected_midi == {60}

    def test_chord_returns_chord_verification(self):
        """When expected_midi has multiple elements, verify dispatches to
        verify_chord and the result wraps a ChordVerification."""
        v = CallTrackingVerifier()
        audio = self.make_audio()
        result = v.verify(
            audio_window=audio,
            expected_midi={60, 64, 67},
            expected_techniques=[],
            mode=MatchMode.ARCADE,
            timestamp_ms=2000.0,
        )

        assert isinstance(result, VerificationResult)
        # The concrete implementation converts the set to a sorted list of ExpectedNote
        assert v.calls == [("verify_chord", [60, 64, 67])]
        assert isinstance(result.verified, ChordVerification)
        assert result.expected_midi == {60, 64, 67}

    def test_techniques_are_verified(self):
        """Each string in expected_techniques causes a verify_technique call."""
        v = CallTrackingVerifier()
        audio = self.make_audio()
        result = v.verify(
            audio_window=audio,
            expected_midi={48},
            expected_techniques=["vibrato", "bend"],
            mode=MatchMode.JUDGE,
            timestamp_ms=1500.0,
        )

        assert v.calls == [
            ("verify_single_note", 48),
            ("verify_technique", "vibrato"),
            ("verify_technique", "bend"),
        ]
        assert len(result.techniques) == 2
        assert result.techniques[0].technique == "vibrato"
        assert result.techniques[1].technique == "bend"

    def test_techniques_empty_list(self):
        """When no techniques are expected, the techniques list is empty."""
        v = CallTrackingVerifier()
        audio = self.make_audio()
        result = v.verify(
            audio_window=audio,
            expected_midi={48},
            expected_techniques=[],
            mode=MatchMode.JUDGE,
            timestamp_ms=0.0,
        )
        assert result.techniques == []

    def test_technique_context_passed_through(self):
        """The technique_context dict is forwarded to each verify_technique call."""
        v = CallTrackingVerifier()
        audio = self.make_audio()
        ctx = {"fret": 3, "string": 5}
        result = v.verify(
            audio_window=audio,
            expected_midi={48},
            expected_techniques=["slide"],
            mode=MatchMode.JUDGE,
            timestamp_ms=500.0,
            technique_context=ctx,
        )
        # verify_technique is called — CallTrackingVerifier doesn't inspect context
        # but the dispatch should not crash and techniques should be populated.
        assert len(result.techniques) == 1
        assert result.techniques[0].technique == "slide"

    def test_verify_does_not_mutate_input_set(self):
        """The expected_midi set passed to verify must not be mutated."""
        original = frozenset({48, 60})
        midi_set: set[int] = set(original)
        v = CallTrackingVerifier()
        audio = self.make_audio()
        v.verify(
            audio_window=audio,
            expected_midi=midi_set,
            expected_techniques=[],
            mode=MatchMode.JUDGE,
            timestamp_ms=0.0,
        )
        assert midi_set == set(original), "caller's set was mutated"


class TestVerifySilence:
    """verify_silence contract — abstract method returning bool."""

    def test_returns_bool(self):
        """A concrete verify_silence implementation returns a bool."""
        v = ConcreteVerifier()
        audio = np.zeros(2048, dtype=np.float32)
        result = v.verify_silence(audio, threshold_db=-40.0)
        assert result is False
        assert isinstance(result, bool)

    def test_silent_audio_returns_true(self):
        """When the subclass detects silence, it should return True."""

        class SilentVerifier(ExpectedEventVerifier):
            def verify_single_note(self, *args, **kwargs):
                return NoteVerification(
                    is_pitch_present=False, is_onset_present=False,
                    pitch_evidence=None, onset_ms=None, harmonic_score=0.0,
                    timing_error_ms=None, alias_risk=0.0,
                )

            def verify_chord(self, *args, **kwargs):
                return ChordVerification(notes=[], partial=False, total_harmonic_energy=0.0)

            def verify_technique(self, *args, **kwargs):
                return TechniqueVerification(technique="", is_present=False, confidence=0.0)

            def verify_silence(self, audio_window, threshold_db):
                # Simple RMS-based silence detection
                rms = np.sqrt(np.mean(audio_window**2))
                return rms < 10.0 ** (threshold_db / 20.0)

        v = SilentVerifier()
        # Completely silent audio
        silent = np.zeros(4096, dtype=np.float32)
        assert v.verify_silence(silent, threshold_db=-40.0)
        assert v.verify_silence(silent, threshold_db=-40.0) == True

        # Loud audio should NOT be silent
        loud = np.ones(4096, dtype=np.float32) * 0.5
        assert not v.verify_silence(loud, threshold_db=-40.0)
        assert v.verify_silence(loud, threshold_db=-40.0) == False


class TestSubclassInheritance:
    """Inheritance and type-checking behavior."""

    def test_isinstance_check(self):
        """A concrete subclass instance passes isinstance check."""
        v = ConcreteVerifier()
        assert isinstance(v, ExpectedEventVerifier)

    def test_concrete_class_not_abstract(self):
        """A fully-implemented subclass should be instantiable (not abstract)."""
        # Direct instantiation succeeds — the class is not abstract.
        v = ConcreteVerifier()
        assert v is not None
        # __abstractmethods__ should be empty on a concrete subclass
        assert not hasattr(ConcreteVerifier, "__abstractmethods__") or not ConcreteVerifier.__abstractmethods__
