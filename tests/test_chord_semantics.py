"""Tests for pickhero.audio.chord_semantics — musically weighted chord scoring."""

import pytest

from pickhero.audio.chord_semantics import score_chord
from pickhero.audio.evidence import ChordVerification, ExpectedNote, NoteVerification, PitchEvidence


def _note_verification(midi: int, confidence: float = 0.9, is_pitch_present: bool = True) -> NoteVerification:
    """Build a NoteVerification with sensible defaults."""
    pitch_evidence = PitchEvidence(
        midi_note=midi,
        cents_error=0.0,
        confidence=confidence,
        source="test",
    )
    return NoteVerification(
        is_pitch_present=is_pitch_present,
        is_onset_present=True,
        pitch_evidence=pitch_evidence,
        onset_ms=0.0,
        harmonic_score=confidence,
        timing_error_ms=0.0,
        alias_risk=0.0,
    )


def _default_chord_verification(
    observed_pcs: frozenset[int] | None = None,
    pc_energy: dict[int, float] | None = None,
    notes: list[NoteVerification] | None = None,
) -> ChordVerification:
    """Build a ChordVerification with sensible defaults."""
    if observed_pcs is None:
        observed_pcs = frozenset()
    if pc_energy is None:
        pc_energy = {}
    return ChordVerification(
        notes=notes or [],
        partial=False,
        total_harmonic_energy=1.0,
        observed_pitch_classes=observed_pcs,
        pitch_class_energy=pc_energy,
        quality_score=0.0,
    )


C_MAJOR_NOTES = [ExpectedNote(midi=60), ExpectedNote(midi=64), ExpectedNote(midi=67)]
"""C major: C(60), E(64), G(67). Pitch classes: {0, 4, 7}."""


class TestScoreChord:
    """score_chord() produces verdicts with correct musical semantics."""

    def test_all_notes_present_returns_hit(self):
        """All chord tones present with high confidence returns 'hit'."""
        notes = [
            _note_verification(midi=60, confidence=0.9),
            _note_verification(midi=64, confidence=0.85),
            _note_verification(midi=67, confidence=0.8),
        ]
        observed_pcs = frozenset({0, 4, 7})
        pc_energy = {0: 0.9, 4: 0.85, 7: 0.8}
        verification = _default_chord_verification(observed_pcs, pc_energy, notes)

        result = score_chord(
            C_MAJOR_NOTES,
            verification,
            hit_threshold=0.75,
            partial_threshold=0.50,
            max_extra_for_hit=0,
        )

        assert result.verdict == "hit"
        assert result.score >= 0.75
        assert result.root_pitch_class == 0  # C
        assert result.expected_pitch_classes == (0, 4, 7)

    def test_missing_third_returns_partial(self):
        """Missing third (E) with root and fifth present returns 'partial'."""
        notes = [
            _note_verification(midi=60, confidence=0.9, is_pitch_present=True),   # C present
            _note_verification(midi=64, confidence=0.0, is_pitch_present=False),  # E absent
            _note_verification(midi=67, confidence=0.8, is_pitch_present=True),   # G present
        ]
        observed_pcs = frozenset({0, 7})
        pc_energy = {0: 0.9, 7: 0.8}
        verification = _default_chord_verification(observed_pcs, pc_energy, notes)

        result = score_chord(
            C_MAJOR_NOTES,
            verification,
            hit_threshold=0.75,
            partial_threshold=0.50,
            max_extra_for_hit=0,
        )

        assert result.verdict == "partial"
        assert "third" in result.missing_critical_roles
        assert result.score < 0.75

    def test_extra_pitch_classes_reduce_score(self):
        """Extra pitch classes (foreign notes) reduce the chord score."""
        notes = [
            _note_verification(midi=60, confidence=0.9),
            _note_verification(midi=64, confidence=0.85),
            _note_verification(midi=67, confidence=0.8),
        ]
        # C, E, G plus F# (pitch class 6) as an extra
        observed_pcs = frozenset({0, 4, 6, 7})
        pc_energy = {0: 0.9, 4: 0.85, 6: 0.7, 7: 0.8}
        verification = _default_chord_verification(observed_pcs, pc_energy, notes)

        result = score_chord(
            C_MAJOR_NOTES,
            verification,
            hit_threshold=0.75,
            partial_threshold=0.50,
            max_extra_for_hit=0,
        )

        assert result.score < 0.80  # penalized for the extra
        assert 6 in result.extra_pitch_classes

    def test_strum_spread_affects_verdict(self):
        """Excessive strum spread reduces the chord score."""
        notes = [
            _note_verification(midi=60, confidence=0.9),
            _note_verification(midi=64, confidence=0.85),
            _note_verification(midi=67, confidence=0.8),
        ]
        observed_pcs = frozenset({0, 4, 7})
        pc_energy = {0: 0.9, 4: 0.85, 7: 0.8}
        verification = _default_chord_verification(observed_pcs, pc_energy, notes)

        # Very wide strum — well over 90 ms max
        result = score_chord(
            C_MAJOR_NOTES,
            verification,
            hit_threshold=0.75,
            partial_threshold=0.50,
            max_extra_for_hit=0,
            strum_spread_ms=200.0,
            max_strum_spread_ms=90.0,
        )

        assert result.score < 0.75
        assert result.strum_spread_ms == 200.0

    def test_empty_expected_notes_returns_miss(self):
        """Empty expected_notes list returns a 'miss' verdict."""
        verification = _default_chord_verification()
        result = score_chord(
            [],
            verification,
            hit_threshold=0.75,
            partial_threshold=0.50,
            max_extra_for_hit=0,
        )

        assert result.verdict == "miss"
        assert result.score == 0.0
        assert result.missing_critical_roles == ("chord",)

    def test_chord_score_has_all_fields(self):
        """ChordScore dataclass exposes all required fields."""
        notes = [
            _note_verification(midi=60, confidence=0.9),
            _note_verification(midi=64, confidence=0.85),
            _note_verification(midi=67, confidence=0.8),
        ]
        observed_pcs = frozenset({0, 4, 7})
        pc_energy = {0: 0.9, 4: 0.85, 7: 0.8}
        verification = _default_chord_verification(observed_pcs, pc_energy, notes)

        result = score_chord(
            C_MAJOR_NOTES,
            verification,
            hit_threshold=0.75,
            partial_threshold=0.50,
            max_extra_for_hit=0,
        )

        assert hasattr(result, "verdict")
        assert hasattr(result, "score")
        assert hasattr(result, "root_pitch_class")
        assert hasattr(result, "bass_pitch_class")
        assert hasattr(result, "expected_pitch_classes")
        assert hasattr(result, "observed_pitch_classes")
        assert hasattr(result, "missing_critical_roles")
        assert hasattr(result, "extra_pitch_classes")
        assert hasattr(result, "role_quality")
        assert hasattr(result, "strum_spread_ms")

    def test_miss_when_all_notes_low_confidence(self):
        """Very low confidence on all notes results in 'miss'."""
        notes = [
            _note_verification(midi=60, confidence=0.1, is_pitch_present=False),
            _note_verification(midi=64, confidence=0.15, is_pitch_present=False),
            _note_verification(midi=67, confidence=0.05, is_pitch_present=False),
        ]
        # No pitch class observations either
        observed_pcs = frozenset()
        pc_energy = {}
        verification = _default_chord_verification(observed_pcs, pc_energy, notes)

        result = score_chord(
            C_MAJOR_NOTES,
            verification,
            hit_threshold=0.75,
            partial_threshold=0.50,
            max_extra_for_hit=0,
        )

        assert result.verdict == "miss"
