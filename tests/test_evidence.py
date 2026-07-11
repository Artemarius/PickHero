"""Tests for pickhero.audio.evidence — data model for verification results."""

import copy
import pickle

import pytest
from dataclasses import replace

from pickhero.audio.evidence import (
    ExpectedNote,
    EvidenceType,
    PitchEvidence,
    NoteVerification,
    ChordVerification,
    TechniqueVerification,
    VerificationResult,
)


# ─── Helper factories ────────────────────────────────────────────────


def make_pitch_evidence(**overrides) -> PitchEvidence:
    """Build a PitchEvidence with sensible defaults."""
    kwargs = dict(
        midi_note=60,
        cents_error=2.5,
        confidence=0.85,
        source="YIN",
    )
    kwargs.update(overrides)
    return PitchEvidence(**kwargs)


def make_note_verification(**overrides) -> NoteVerification:
    """Build a NoteVerification with sensible defaults."""
    kwargs = dict(
        is_pitch_present=True,
        is_onset_present=True,
        pitch_evidence=make_pitch_evidence(),
        onset_ms=12.0,
        harmonic_score=0.9,
        timing_error_ms=1.0,
        alias_risk=0.0,
    )
    kwargs.update(overrides)
    return NoteVerification(**kwargs)


def make_chord_verification(**overrides) -> ChordVerification:
    """Build a ChordVerification with sensible defaults."""
    kwargs = dict(
        notes=[
            make_note_verification(pitch_evidence=make_pitch_evidence(midi_note=60)),
            make_note_verification(pitch_evidence=make_pitch_evidence(midi_note=64)),
            make_note_verification(pitch_evidence=make_pitch_evidence(midi_note=67)),
        ],
        partial=False,
        total_harmonic_energy=2.7,
        observed_pitch_classes=frozenset({0, 4, 7}),
        pitch_class_energy={0: 0.9, 4: 0.85, 7: 0.95},
        quality_score=1.0,
        missing_roles=(),
        extra_pitch_classes=(),
    )
    kwargs.update(overrides)
    return ChordVerification(**kwargs)


def make_technique_verification(**overrides) -> TechniqueVerification:
    """Build a TechniqueVerification with sensible defaults."""
    kwargs = dict(
        technique="bend",
        is_present=True,
        confidence=0.78,
        details={"cents": 100},
        uncertain=False,
        quality=0.8,
    )
    kwargs.update(overrides)
    return TechniqueVerification(**kwargs)


# ═══════════════════════════════════════════════════════════════════════
# ExpectedNote (frozen dataclass)
# ═══════════════════════════════════════════════════════════════════════


class TestExpectedNote:
    def test_construct_minimal(self):
        """Only midi is required; others default to None."""
        n = ExpectedNote(midi=60)
        assert n.midi == 60
        assert n.string is None
        assert n.fret is None
        assert n.event_id is None

    def test_construct_full(self):
        """All positional + optional fields set correctly."""
        n = ExpectedNote(midi=64, string=2, fret=5, event_id="12345:1")
        assert n.midi == 64
        assert n.string == 2
        assert n.fret == 5
        assert n.event_id == "12345:1"

    def test_frozen_prevents_attribute_mutation(self):
        """Direct setattr on a frozen dataclass raises FrozenInstanceError."""
        n = ExpectedNote(midi=60, string=1)
        with pytest.raises(Exception):
            n.midi = 62

    def test_frozen_prevents_replace_bypass(self):
        """dataclasses.replace creates a *new* instance; original is unchanged."""
        n = ExpectedNote(midi=60, string=1, fret=3)
        n2 = replace(n, midi=62)
        assert n2.midi == 62
        assert n2.string == 1
        assert n2.fret == 3
        # original untouched
        assert n.midi == 60

    def test_frozen_is_hashable(self):
        """Frozen dataclasses are hashable by default (can be used in sets)."""
        n1 = ExpectedNote(midi=60, string=1)
        n2 = ExpectedNote(midi=60, string=1)
        s = {n1, n2}
        assert len(s) == 1  # same value → same hash → deduplicated

    def test_repr(self):
        """repr is the standard dataclass repr (not the default object repr)."""
        n = ExpectedNote(midi=60, string=1, fret=3, event_id="ev1")
        r = repr(n)
        assert r.startswith("ExpectedNote(")
        assert "midi=60" in r
        assert "string=1" in r
        assert "fret=3" in r
        assert "event_id=" in r

    def test_equality(self):
        """Two ExpectedNotes with identical fields are equal."""
        a = ExpectedNote(midi=60, string=1, fret=3, event_id="ev1")
        b = ExpectedNote(midi=60, string=1, fret=3, event_id="ev1")
        assert a == b

    def test_inequality(self):
        """Difference in any field breaks equality."""
        a = ExpectedNote(midi=60, string=1)
        b = ExpectedNote(midi=62, string=1)
        assert a != b

    def test_copy_roundtrip(self):
        """copy.copy/deepcopy produce equal (not same) objects for frozen dataclass."""
        n = ExpectedNote(midi=60)
        assert copy.copy(n) == n
        assert copy.deepcopy(n) == n

    def test_pickle_roundtrip(self):
        """Frozen dataclass survives pickle roundtrip."""
        n = ExpectedNote(midi=72, string=3, fret=12, event_id="pickle_test")
        restored = pickle.loads(pickle.dumps(n))
        assert restored == n
        assert restored.midi == 72
        assert restored.string == 3


# ═══════════════════════════════════════════════════════════════════════
# EvidenceType (enum)
# ═══════════════════════════════════════════════════════════════════════


class TestEvidenceType:
    def test_all_members_present(self):
        """All 9 expected enum values exist."""
        assert EvidenceType.SINGLE_NOTE.value == "single_note"
        assert EvidenceType.CHORD.value == "chord"
        assert EvidenceType.BEND.value == "bend"
        assert EvidenceType.SLIDE.value == "slide"
        assert EvidenceType.VIBRATO.value == "vibrato"
        assert EvidenceType.HARMONIC.value == "harmonic"
        assert EvidenceType.DEAD_NOTE.value == "dead_note"
        assert EvidenceType.PALM_MUTE.value == "palm_mute"
        assert EvidenceType.NOISE_GESTURE.value == "noise_gesture"

    def test_member_count(self):
        """Exactly 9 evidence types."""
        assert len(EvidenceType) == 9

    def test_member_names(self):
        """All expected names match the exact casing."""
        names = {m.name for m in EvidenceType}
        expected = {
            "SINGLE_NOTE",
            "CHORD",
            "BEND",
            "SLIDE",
            "VIBRATO",
            "HARMONIC",
            "DEAD_NOTE",
            "PALM_MUTE",
            "NOISE_GESTURE",
        }
        assert names == expected

    def test_value_uniqueness(self):
        """Every member has a unique string value."""
        values = [m.value for m in EvidenceType]
        assert len(values) == len(set(values))

    def test_is_enum(self):
        """EvidenceType is a proper enum subclass."""
        from enum import Enum
        assert issubclass(EvidenceType, Enum)
    def test_from_value(self):
        """Can look up by value (e.g. from JSON/serialized data)."""
        assert EvidenceType("single_note") is EvidenceType.SINGLE_NOTE
        assert EvidenceType("chord") is EvidenceType.CHORD
        assert EvidenceType("bend") is EvidenceType.BEND
        assert EvidenceType("noise_gesture") is EvidenceType.NOISE_GESTURE

    def test_invalid_value_raises(self):
        """Unknown string raises ValueError."""
        with pytest.raises(ValueError):
            EvidenceType("invalid_type")

    def test_hashable(self):
        """Enums are hashable so can be used in sets/dicts."""
        s = {EvidenceType.SINGLE_NOTE, EvidenceType.CHORD, EvidenceType.SINGLE_NOTE}
        assert len(s) == 2


# ═══════════════════════════════════════════════════════════════════════
# PitchEvidence
# ═══════════════════════════════════════════════════════════════════════


class TestPitchEvidence:
    def test_construct_all_fields(self):
        """Constructor sets midi_note, cents_error, confidence, source."""
        pe = PitchEvidence(midi_note=72, cents_error=-5.0, confidence=0.92, source="CREPE")
        assert pe.midi_note == 72
        assert pe.cents_error == -5.0
        assert pe.confidence == 0.92
        assert pe.source == "CREPE"

    def test_cents_error_none(self):
        """cents_error can be None (e.g. when pitch not reliably tracked)."""
        pe = PitchEvidence(midi_note=48, cents_error=None, confidence=0.3, source="chroma")
        assert pe.cents_error is None

    def test_confidence_edge_values(self):
        """confidence can be 0.0 or 1.0 (boundary)."""
        low = PitchEvidence(midi_note=36, cents_error=0.0, confidence=0.0, source="YIN")
        high = PitchEvidence(midi_note=36, cents_error=0.0, confidence=1.0, source="YIN")
        assert low.confidence == 0.0
        assert high.confidence == 1.0

    def test_confidence_range(self):
        """confidence is not validated to be 0-1 (but typically is)."""
        # The dataclass imposes no range constraint; this is acceptable.
        pe = PitchEvidence(midi_note=60, cents_error=0.0, confidence=-0.1, source="YIN")
        assert pe.confidence == -0.1

    def test_source_various(self):
        """source accepts any string identifier."""
        for src in ("YIN", "CREPE", "spectral", "chroma", "pitch_shifter"):
            pe = PitchEvidence(midi_note=60, cents_error=0.0, confidence=0.5, source=src)
            assert pe.source == src

    def test_mutable_can_be_modified(self):
        """PitchEvidence is a regular dataclass — fields can be reassigned."""
        pe = PitchEvidence(midi_note=60, cents_error=0.0, confidence=0.5, source="YIN")
        pe.midi_note = 64
        assert pe.midi_note == 64

    def test_replace(self):
        """dataclasses.replace returns a new instance with overrides."""
        pe = PitchEvidence(midi_note=60, cents_error=0.0, confidence=0.5, source="YIN")
        pe2 = replace(pe, confidence=0.95)
        assert pe2.confidence == 0.95
        assert pe2.midi_note == 60
        # original unchanged
        assert pe.confidence == 0.5

    def test_equality(self):
        """Two PitchEvidence with matching fields are equal."""
        a = PitchEvidence(midi_note=60, cents_error=0.0, confidence=0.5, source="YIN")
        b = PitchEvidence(midi_note=60, cents_error=0.0, confidence=0.5, source="YIN")
        assert a == b

    def test_repr(self):
        """repr includes all fields."""
        pe = PitchEvidence(midi_note=72, cents_error=1.5, confidence=0.9, source="CREPE")
        r = repr(pe)
        assert r.startswith("PitchEvidence(")
        assert "midi_note=72" in r
        assert "cents_error=1.5" in r
        assert "confidence=0.9" in r
        assert "source=" in r


# ═══════════════════════════════════════════════════════════════════════
# NoteVerification
# ═══════════════════════════════════════════════════════════════════════


class TestNoteVerification:
    def test_construct_all_fields(self):
        """All fields assignable via constructor."""
        pe = make_pitch_evidence()
        nv = NoteVerification(
            is_pitch_present=True,
            is_onset_present=False,
            pitch_evidence=pe,
            onset_ms=15.0,
            harmonic_score=0.7,
            timing_error_ms=-2.0,
            alias_risk=0.15,
        )
        assert nv.is_pitch_present is True
        assert nv.is_onset_present is False
        assert nv.pitch_evidence is pe
        assert nv.onset_ms == 15.0
        assert nv.harmonic_score == 0.7
        assert nv.timing_error_ms == -2.0
        assert nv.alias_risk == 0.15

    def test_default_alias_risk(self):
        """alias_risk defaults to 0.0."""
        nv = NoteVerification(
            is_pitch_present=True,
            is_onset_present=True,
            pitch_evidence=None,
            onset_ms=None,
            harmonic_score=0.5,
            timing_error_ms=None,
        )
        assert nv.alias_risk == 0.0

    def test_is_pitch_present_false(self):
        """Test with is_pitch_present=False."""
        nv = make_note_verification(is_pitch_present=False)
        assert nv.is_pitch_present is False

    def test_is_onset_present_false(self):
        """Test with is_onset_present=False."""
        nv = make_note_verification(is_onset_present=False)
        assert nv.is_onset_present is False

    def test_confidence_with_pitch_evidence(self):
        """confidence property returns pitch_evidence.confidence when present."""
        pe = make_pitch_evidence(confidence=0.88)
        nv = make_note_verification(pitch_evidence=pe, harmonic_score=0.3)
        # Should use pitch_evidence.confidence (0.88), not harmonic_score (0.3)
        assert nv.confidence == 0.88

    def test_confidence_without_pitch_evidence(self):
        """confidence property falls back to harmonic_score when pitch_evidence is None."""
        nv = make_note_verification(
            pitch_evidence=None,
            harmonic_score=0.65,
        )
        assert nv.confidence == 0.65

    def test_confidence_pitch_none_and_zero_harmonic(self):
        """Edge case: no pitch evidence and harmonic_score is 0.0."""
        nv = make_note_verification(pitch_evidence=None, harmonic_score=0.0)
        assert nv.confidence == 0.0

    def test_confidence_with_negative_harmonic(self):
        """Edge case: harmonic_score negative (confidence = that value)."""
        nv = make_note_verification(pitch_evidence=None, harmonic_score=-1.0)
        assert nv.confidence == -1.0

    def test_onset_ms_none(self):
        """onset_ms can be None when onset not detected."""
        nv = make_note_verification(onset_ms=None)
        assert nv.onset_ms is None

    def test_timing_error_ms_none(self):
        """timing_error_ms can be None."""
        nv = make_note_verification(timing_error_ms=None)
        assert nv.timing_error_ms is None

    def test_pitch_evidence_none(self):
        """pitch_evidence can be None."""
        nv = make_note_verification(pitch_evidence=None)
        assert nv.pitch_evidence is None

    def test_replace_mutable_fields(self):
        """dataclasses.replace works (NoteVerification is not frozen)."""
        nv = make_note_verification(is_pitch_present=True, harmonic_score=0.9)
        nv2 = replace(nv, is_pitch_present=False, harmonic_score=0.2)
        assert nv2.is_pitch_present is False
        assert nv2.harmonic_score == 0.2
        # original unchanged
        assert nv.is_pitch_present is True

    def test_alias_risk_out_of_range(self):
        """alias_risk is not ranged-validated (0.0-1.0 is convention)."""
        nv = make_note_verification(alias_risk=1.5)
        assert nv.alias_risk == 1.5

    def test_repr(self):
        """repr includes key fields."""
        nv = make_note_verification(is_pitch_present=True, harmonic_score=0.75)
        r = repr(nv)
        assert r.startswith("NoteVerification(")
        assert "is_pitch_present=True" in r
        assert "harmonic_score=0.75" in r


# ═══════════════════════════════════════════════════════════════════════
# ChordVerification
# ═══════════════════════════════════════════════════════════════════════


class TestChordVerification:
    def test_construct_minimal(self):
        """Only required fields: notes, partial, total_harmonic_energy."""
        nv1 = make_note_verification()
        nv2 = make_note_verification(pitch_evidence=make_pitch_evidence(midi_note=64))
        cv = ChordVerification(notes=[nv1, nv2], partial=False, total_harmonic_energy=1.5)
        assert cv.notes == [nv1, nv2]
        assert cv.partial is False
        assert cv.total_harmonic_energy == 1.5
        # defaults
        assert cv.observed_pitch_classes == frozenset()
        assert cv.pitch_class_energy == {}
        assert cv.quality_score == 0.0
        assert cv.missing_roles == ()
        assert cv.extra_pitch_classes == ()

    def test_construct_full(self):
        """All fields set via constructor."""
        nv_list = [make_note_verification()]
        cv = ChordVerification(
            notes=nv_list,
            partial=True,
            total_harmonic_energy=3.2,
            observed_pitch_classes=frozenset({0, 4}),
            pitch_class_energy={0: 0.9, 4: 0.8},
            quality_score=0.75,
            missing_roles=("root", "fifth"),
            extra_pitch_classes=(6, 11),
        )
        assert cv.notes is nv_list
        assert cv.partial is True
        assert cv.total_harmonic_energy == 3.2
        assert cv.observed_pitch_classes == frozenset({0, 4})
        assert cv.pitch_class_energy == {0: 0.9, 4: 0.8}
        assert cv.quality_score == 0.75
        assert cv.missing_roles == ("root", "fifth")
        assert cv.extra_pitch_classes == (6, 11)

    def test_partial_flag(self):
        """Test partial=True / False explicitly."""
        cv_partial = make_chord_verification(partial=True)
        cv_full = make_chord_verification(partial=False)
        assert cv_partial.partial is True
        assert cv_full.partial is False

    def test_notes_list_mutable(self):
        """notes is a mutable list (regular dataclass)."""
        cv = make_chord_verification()
        original_len = len(cv.notes)
        cv.notes.append(make_note_verification())
        assert len(cv.notes) == original_len + 1

    def test_notes_empty(self):
        """Edge case: empty notes list."""
        cv = ChordVerification(notes=[], partial=False, total_harmonic_energy=0.0)
        assert cv.notes == []
        assert cv.partial is False

    def test_observed_pitch_classes_immutable(self):
        """observed_pitch_classes is a frozenset — cannot be mutated."""
        cv = make_chord_verification()
        assert isinstance(cv.observed_pitch_classes, frozenset)

    def test_pitch_class_energy_dict_mutable(self):
        """pitch_class_energy is a regular dict (mutable)."""
        cv = make_chord_verification()
        cv.pitch_class_energy[3] = 0.5
        assert cv.pitch_class_energy[3] == 0.5

    def test_quality_score_zero(self):
        """quality_score defaults to 0.0."""
        cv = ChordVerification(notes=[], partial=False, total_harmonic_energy=0.0)
        assert cv.quality_score == 0.0

    def test_quality_score_one(self):
        """quality_score can be 1.0 (perfect chord)."""
        cv = make_chord_verification(quality_score=1.0)
        assert cv.quality_score == 1.0

    def test_missing_roles_tuple(self):
        """missing_roles is a tuple of strings."""
        cv = make_chord_verification(missing_roles=("third",))
        assert isinstance(cv.missing_roles, tuple)
        assert cv.missing_roles == ("third",)

    def test_extra_pitch_classes_tuple(self):
        """extra_pitch_classes is a tuple of ints."""
        cv = make_chord_verification(extra_pitch_classes=(5, 8))
        assert isinstance(cv.extra_pitch_classes, tuple)
        assert cv.extra_pitch_classes == (5, 8)

    def test_extra_pitch_classes_empty(self):
        """extra_pitch_classes defaults to empty tuple."""
        cv = make_chord_verification(extra_pitch_classes=())
        assert cv.extra_pitch_classes == ()

    def test_replace(self):
        """dataclasses.replace works."""
        cv = make_chord_verification(partial=False, total_harmonic_energy=2.0)
        cv2 = replace(cv, partial=True, total_harmonic_energy=3.0)
        assert cv2.partial is True
        assert cv2.total_harmonic_energy == 3.0
        assert cv.partial is False  # original unchanged

    def test_repr_with_defaults(self):
        """repr includes list-of-notes and partial."""
        cv = make_chord_verification(partial=True)
        r = repr(cv)
        assert r.startswith("ChordVerification(")
        assert "partial=True" in r
        assert "total_harmonic_energy=" in r


# ═══════════════════════════════════════════════════════════════════════
# TechniqueVerification
# ═══════════════════════════════════════════════════════════════════════


class TestTechniqueVerification:
    def test_construct_minimal(self):
        """Required fields only: technique, is_present, confidence."""
        tv = TechniqueVerification(technique="bend", is_present=True, confidence=0.9)
        assert tv.technique == "bend"
        assert tv.is_present is True
        assert tv.confidence == 0.9
        # defaults
        assert tv.details == {}
        assert tv.uncertain is False
        assert tv.quality is None

    def test_construct_full(self):
        """All fields set."""
        tv = TechniqueVerification(
            technique="slide",
            is_present=False,
            confidence=0.2,
            details={"start": 60, "end": 62},
            uncertain=True,
            quality=0.3,
        )
        assert tv.technique == "slide"
        assert tv.is_present is False
        assert tv.confidence == 0.2
        assert tv.details == {"start": 60, "end": 62}
        assert tv.uncertain is True
        assert tv.quality == 0.3

    def test_tri_state_uncertain(self):
        """Uncertain flag makes it tri-state — not present, not absent."""
        tv = TechniqueVerification(technique="vibrato", is_present=True, confidence=0.5, uncertain=True)
        assert tv.uncertain is True
        assert tv.is_present is True  # still set, but uncertain overrides semantics

    def test_uncertain_with_is_present_false(self):
        """Edge case: uncertain=True with is_present=False."""
        tv = TechniqueVerification(technique="harmonic", is_present=False, confidence=0.1, uncertain=True)
        assert tv.uncertain is True
        assert tv.is_present is False

    def test_is_present_true(self):
        """is_present=True indicates the technique was detected."""
        tv = make_technique_verification(is_present=True)
        assert tv.is_present is True

    def test_is_present_false(self):
        """is_present=False indicates the technique was not detected."""
        tv = make_technique_verification(is_present=False)
        assert tv.is_present is False

    def test_confidence_boundary_values(self):
        """confidence can be 0.0 or 1.0."""
        low = TechniqueVerification(technique="bend", is_present=False, confidence=0.0)
        high = TechniqueVerification(technique="bend", is_present=True, confidence=1.0)
        assert low.confidence == 0.0
        assert high.confidence == 1.0

    def test_quality_none_to_float(self):
        """quality transitions from None (default) to a float."""
        tv = make_technique_verification(quality=None)
        assert tv.quality is None
        tv2 = replace(tv, quality=0.5)
        assert tv2.quality == 0.5

    def test_details_dict_mutable(self):
        """details is a regular dict — can be mutated."""
        tv = TechniqueVerification(technique="bend", is_present=True, confidence=0.9)
        tv.details["cents"] = 50
        assert tv.details["cents"] == 50

    def test_technique_various(self):
        """technique accepts any string label."""
        for name in ("bend", "slide", "vibrato", "harmonic", "tap", "palm_mute", "custom_artifact"):
            tv = TechniqueVerification(technique=name, is_present=True, confidence=0.5)
            assert tv.technique == name

    def test_replace(self):
        """dataclasses.replace works."""
        tv = make_technique_verification(technique="bend", is_present=True)
        tv2 = replace(tv, technique="slide", is_present=False)
        assert tv2.technique == "slide"
        assert tv2.is_present is False
        assert tv.technique == "bend"  # original untouched

    def test_repr(self):
        """repr includes technique and is_present."""
        tv = make_technique_verification(technique="tremolo", is_present=True)
        r = repr(tv)
        assert r.startswith("TechniqueVerification(")
        assert "technique=" in r
        assert "is_present=True" in r


# ═══════════════════════════════════════════════════════════════════════
# VerificationResult
# ═══════════════════════════════════════════════════════════════════════


class TestVerificationResult:
    def test_construct_with_note_verification(self):
        """Construct with a NoteVerification as the verified field."""
        nv = make_note_verification()
        tv = make_technique_verification()
        vr = VerificationResult(
            expected_midi={60},
            expected_techniques=["bend"],
            verified=nv,
            techniques=[tv],
            timestamp_ms=1234.5,
        )
        assert vr.expected_midi == {60}
        assert vr.expected_techniques == ["bend"]
        assert vr.verified is nv
        assert vr.techniques == [tv]
        assert vr.timestamp_ms == 1234.5

    def test_construct_with_chord_verification(self):
        """Construct with a ChordVerification as the verified field."""
        cv = make_chord_verification()
        vr = VerificationResult(
            expected_midi={60, 64, 67},
            expected_techniques=[],
            verified=cv,
            techniques=[],
            timestamp_ms=5678.9,
        )
        assert vr.expected_midi == {60, 64, 67}
        assert vr.verified is cv
        assert vr.techniques == []

    def test_expected_midi_mutable_set(self):
        """expected_midi is a set — mutable."""
        vr = VerificationResult(
            expected_midi={60},
            expected_techniques=[],
            verified=make_note_verification(),
            techniques=[],
            timestamp_ms=0.0,
        )
        vr.expected_midi.add(64)
        assert 64 in vr.expected_midi

    def test_expected_techniques_mutable_list(self):
        """expected_techniques is a list — mutable."""
        vr = VerificationResult(
            expected_midi=set(),
            expected_techniques=["bend"],
            verified=make_note_verification(),
            techniques=[],
            timestamp_ms=0.0,
        )
        vr.expected_techniques.append("slide")
        assert len(vr.expected_techniques) == 2

    def test_techniques_empty_list(self):
        """techniques can be an empty list."""
        vr = VerificationResult(
            expected_midi=set(),
            expected_techniques=[],
            verified=make_note_verification(),
            techniques=[],
            timestamp_ms=0.0,
        )
        assert vr.techniques == []

    def test_timestamp_ms_zero(self):
        """timestamp_ms can be 0.0."""
        vr = VerificationResult(
            expected_midi=set(),
            expected_techniques=[],
            verified=make_note_verification(),
            techniques=[],
            timestamp_ms=0.0,
        )
        assert vr.timestamp_ms == 0.0

    def test_timestamp_ms_negative(self):
        """timestamp_ms can be negative (e.g. pre-roll)."""
        vr = VerificationResult(
            expected_midi=set(),
            expected_techniques=[],
            verified=make_note_verification(),
            techniques=[],
            timestamp_ms=-100.0,
        )
        assert vr.timestamp_ms == -100.0

    def test_replace(self):
        """dataclasses.replace works."""
        nv = make_note_verification()
        vr = VerificationResult(
            expected_midi={60},
            expected_techniques=["bend"],
            verified=nv,
            techniques=[],
            timestamp_ms=100.0,
        )
        vr2 = replace(vr, timestamp_ms=200.0, expected_midi={60, 64})
        assert vr2.timestamp_ms == 200.0
        assert vr2.expected_midi == {60, 64}
        assert vr.timestamp_ms == 100.0  # original unchanged

    def test_isinstance_verified_note(self):
        """verified is an instance of NoteVerification."""
        vr = VerificationResult(
            expected_midi={60},
            expected_techniques=[],
            verified=make_note_verification(),
            techniques=[],
            timestamp_ms=0.0,
        )
        assert isinstance(vr.verified, NoteVerification)
        assert not isinstance(vr.verified, ChordVerification)

    def test_isinstance_verified_chord(self):
        """verified is an instance of ChordVerification."""
        vr = VerificationResult(
            expected_midi={60, 64, 67},
            expected_techniques=[],
            verified=make_chord_verification(),
            techniques=[],
            timestamp_ms=0.0,
        )
        assert isinstance(vr.verified, ChordVerification)

    def test_repr(self):
        """repr includes expected_midi and timestamp_ms."""
        vr = VerificationResult(
            expected_midi={60, 64},
            expected_techniques=["bend"],
            verified=make_note_verification(),
            techniques=[],
            timestamp_ms=42.0,
        )
        r = repr(vr)
        assert r.startswith("VerificationResult(")
        assert "expected_midi=" in r
        assert "timestamp_ms=42" in r
