"""Tests for pickhero.evaluation.conversion — dataset-to-case conversion."""

from __future__ import annotations

import pytest

from pickhero.datasets.schema import ClipEvent, ClipExpectedNote
from pickhero.evaluation.conversion import (
    _counterfactuals,
    _split_for_group,
    cases_from_clip_events,
)
from pickhero.evaluation.manifest import (
    CorpusCase,
    CorpusExpectedNote,
    CorpusSplit,
    EventKind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(**overrides: object) -> ClipEvent:
    """Minimal single-note ClipEvent with overridable fields."""
    defaults: dict[str, object] = dict(
        clip_id="test/case/001",
        source="TestDataset",
        start_s=1.0,
        end_s=1.5,
        midi=40,
        technique="none",
        confidence=1.0,
        audio_path="/data/test/sample.wav",
        string=2,
        fret=3,
        notes=(),
        metadata={},
    )
    defaults.update(overrides)
    return ClipEvent(**defaults)  # type: ignore[arg-type]


def _case(**overrides: object) -> CorpusCase:
    """Minimal valid CorpusCase with overridable fields."""
    defaults: dict[str, object] = dict(
        case_id="test/case",
        audio_path="/data/sample.wav",
        source="Test",
        split=CorpusSplit.TEST,
        event_kind=EventKind.SINGLE_NOTE,
        start_s=0.0,
        end_s=1.0,
        expected_present=True,
        notes=(CorpusExpectedNote(midi=40),),
    )
    defaults.update(overrides)
    return CorpusCase(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# cases_from_clip_events
# ---------------------------------------------------------------------------


class TestCasesFromClipEvents:
    def test_empty_events(self):
        """Empty events list yields empty cases list."""
        assert cases_from_clip_events([]) == []

    def test_single_note_required_fields(self):
        """Each produced case has all required fields populated."""
        event = _event()
        cases = cases_from_clip_events([event])
        assert len(cases) == 1
        case = cases[0]
        assert case.case_id == "test/case/001"
        assert case.audio_path == "/data/test/sample.wav"
        assert case.source == "TestDataset"
        assert case.split in (CorpusSplit.CALIBRATION, CorpusSplit.TEST)
        assert case.event_kind == EventKind.SINGLE_NOTE
        assert case.start_s == 1.0
        assert case.end_s == 1.5
        assert case.expected_present is True
        assert len(case.notes) == 1
        assert case.notes[0].midi == 40
        assert case.notes[0].string == 2
        assert case.notes[0].fret == 3
        assert case.notes[0].role is None
        assert case.technique is None
        assert case.technique_present is None
        assert case.expected_onset_s == 1.0
        assert case.annotation_confidence == 1.0
        assert case.negative_reason is None
        assert "provenance" in case.metadata
        assert case.metadata["provenance"] == "dataset_annotation"

    def test_end_s_floor(self):
        """end_s is floored to start_s + 0.05 when event.end_s <= start_s."""
        event = _event(start_s=2.0, end_s=1.9)
        cases = cases_from_clip_events([event])
        assert cases[0].end_s == pytest.approx(2.05)

    def test_end_s_zero_length(self):
        """end_s == start_s gets bumped to +0.05."""
        event = _event(start_s=3.0, end_s=3.0)
        cases = cases_from_clip_events([event])
        assert cases[0].end_s == pytest.approx(3.05)

    def test_chord_event(self):
        """Multiple notes in event.notes produce CHORD case kind."""
        event = _event(
            clip_id="test/chord/001",
            midi=None,
            notes=(
                ClipExpectedNote(midi=48, string=1, fret=5, role="root"),
                ClipExpectedNote(midi=52, string=2, fret=5, role="third"),
                ClipExpectedNote(midi=55, string=3, fret=5, role="fifth"),
            ),
        )
        cases = cases_from_clip_events([event])
        assert len(cases) == 1
        case = cases[0]
        assert case.event_kind == EventKind.CHORD
        assert len(case.notes) == 3
        assert case.notes[0].midi == 48
        assert case.notes[0].role == "root"
        assert case.notes[1].midi == 52
        assert case.notes[1].role == "third"
        assert case.notes[2].midi == 55
        assert case.notes[2].role == "fifth"
        assert case.technique is None
        assert case.technique_present is None

    def test_technique_event(self):
        """Non-normal technique produces TECHNIQUE case kind."""
        event = _event(clip_id="test/tech/001", technique="vibrato")
        cases = cases_from_clip_events([event])
        assert len(cases) == 1
        case = cases[0]
        assert case.event_kind == EventKind.TECHNIQUE
        assert case.technique == "vibrato"
        assert case.technique_present is True

    def test_technique_normalization(self):
        """Technique label is stripped and lowercased before comparison."""
        cases = cases_from_clip_events([
            _event(clip_id="a", technique="  VIBRATO  "),
        ])
        assert cases[0].technique == "vibrato"

    def test_normal_techniques_yield_single_note(self):
        """'normal', 'sustain', 'none', and blank technique -> SINGLE_NOTE."""
        for label in ("normal", "sustain", "none", "", "unknown"):
            cases = cases_from_clip_events([
                _event(clip_id=f"test/{label}/001", technique=label),
            ])
            assert cases[0].event_kind == EventKind.SINGLE_NOTE, \
                f"technique {label!r} should yield SINGLE_NOTE"
            assert cases[0].technique is None

    def test_split_determinism(self):
        """Same audio_path yields same split across calls."""
        e1 = _event(clip_id="test/det/001")
        e2 = _event(clip_id="test/det/002")
        cases = cases_from_clip_events(
            [e1, e2],
            calibration_fraction=0.50,
        )
        assert cases[0].split == cases[1].split

    def test_metadata_provenance(self):
        """metadata gets provenance and preserves original values."""
        event = _event(clip_id="test/meta/001", metadata={"key": "val"})
        cases = cases_from_clip_events([event])
        assert cases[0].metadata["key"] == "val"
        assert cases[0].metadata["provenance"] == "dataset_annotation"

    def test_annotation_confidence_range(self):
        """annotation_confidence is carried through."""
        event = _event(clip_id="test/conf/001", confidence=0.85)
        cases = cases_from_clip_events([event])
        assert cases[0].annotation_confidence == 0.85

    def test_metadata_group_key_fallback(self):
        """When metadata lacks the split_group key, fallback to audio_path."""
        event = _event(
            clip_id="test/group/001",
            audio_path="/fallback/path.wav",
            metadata={},
        )
        cases = cases_from_clip_events(
            [event],
            split_group="custom_group",
        )
        assert len(cases) == 1

    def test_metadata_group_key_used(self):
        """When metadata has the split_group key, it is used for grouping."""
        event = _event(
            clip_id="test/mgroup/001",
            audio_path="/audio/path.wav",
            metadata={"group_key": "shared_group"},
        )
        cases = cases_from_clip_events(
            [event],
            split_group="group_key",
        )
        assert len(cases) == 1

    def test_invalid_calibration_fraction_negative(self):
        """Negative calibration_fraction raises ValueError."""
        with pytest.raises(ValueError, match="calibration_fraction"):
            cases_from_clip_events(
                [_event()],
                calibration_fraction=-0.1,
            )

    def test_invalid_calibration_fraction_too_high(self):
        """calibration_fraction > 1 raises ValueError."""
        with pytest.raises(ValueError, match="calibration_fraction"):
            cases_from_clip_events(
                [_event()],
                calibration_fraction=1.5,
            )

    def test_add_counterfactual_negatives_flag(self):
        """With add_counterfactual_negatives=True, three counterfactuals per case."""
        cases = cases_from_clip_events(
            [
                _event(clip_id="test/cf/001"),
                _event(clip_id="test/cf/002", technique="hammer_on"),
            ],
            add_counterfactual_negatives=True,
        )
        # 2 original + 3 + 3 counterfactuals = 8
        assert len(cases) == 8

        cfs = [c for c in cases if "counterfactual" in c.case_id]
        assert len(cfs) == 6
        assert all(c.expected_present is False for c in cfs)
        assert all(c.negative_reason == "counterfactual_expected_label" for c in cfs)

    def test_multiple_events(self):
        """Multiple events produce one case each."""
        events = [
            _event(clip_id="test/multi/001"),
            _event(clip_id="test/multi/002", midi=45),
            _event(clip_id="test/multi/003", midi=50),
        ]
        cases = cases_from_clip_events(events)
        assert len(cases) == 3
        assert cases[0].case_id == "test/multi/001"
        assert cases[1].case_id == "test/multi/002"
        assert cases[2].case_id == "test/multi/003"


# ---------------------------------------------------------------------------
# _split_for_group
# ---------------------------------------------------------------------------


class TestSplitForGroup:
    def test_deterministic(self):
        """Same group string yields same split."""
        assert _split_for_group("hello", 0.30) == _split_for_group("hello", 0.30)

    def test_different_groups_can_differ(self):
        """Different group strings may produce different splits (probabilistic)."""
        splits = {_split_for_group(f"group_{i}", 0.30) for i in range(100)}
        assert len(splits) > 0

    def test_calibration_threshold_zero(self):
        """calibration_fraction=0.0 => all groups TEST."""
        assert _split_for_group("any_group", 0.0) == CorpusSplit.TEST

    def test_calibration_threshold_one(self):
        """calibration_fraction=1.0 => all groups CALIBRATION."""
        assert _split_for_group("any_group", 1.0) == CorpusSplit.CALIBRATION

    def test_hash_distribution(self):
        """Groups distribute roughly according to calibration_fraction."""
        n = 500
        cal_count = sum(
            1 for i in range(n)
            if _split_for_group(f"group_{i}", 0.30) == CorpusSplit.CALIBRATION
        )
        # Roughly 30 % (allow margin for n = 500)
        assert 0.15 * n <= cal_count <= 0.45 * n, f"cal_count={cal_count} out of range"

    def test_empty_string(self):
        """Empty group string does not crash."""
        result = _split_for_group("", 0.30)
        assert result in (CorpusSplit.CALIBRATION, CorpusSplit.TEST)

    def test_unicode_string(self):
        """Unicode group string works."""
        result = _split_for_group("cafe/etudiant/100", 0.30)
        assert result in (CorpusSplit.CALIBRATION, CorpusSplit.TEST)

    def test_long_string(self):
        """Very long group strings work without issue."""
        result = _split_for_group("a" * 10000, 0.30)
        assert result in (CorpusSplit.CALIBRATION, CorpusSplit.TEST)


# ---------------------------------------------------------------------------
# _counterfactuals
# ---------------------------------------------------------------------------


class TestCounterfactuals:
    def test_silence_returns_empty(self):
        """Silence cases produce no counterfactuals."""
        case = _case(
            case_id="test/silence",
            event_kind=EventKind.SILENCE,
            expected_present=False,
            notes=(),
            negative_reason="silent_clip",
        )
        assert _counterfactuals(case) == []

    def test_single_note_three_offsets(self):
        """Single-note case produces 3 counterfactuals at offsets 1, 2, 12."""
        case = _case(case_id="test/note")
        cfs = _counterfactuals(case)
        assert len(cfs) == 3
        expected_offsets = (1, 2, 12)
        for i, offset in enumerate(expected_offsets):
            assert cfs[i].case_id == f"test/note:counterfactual:+{offset}"
            assert cfs[i].expected_present is False
            assert cfs[i].negative_reason == "counterfactual_expected_label"
            assert cfs[i].notes[0].midi == 40 + offset
            assert cfs[i].event_kind == EventKind.SINGLE_NOTE
            assert cfs[i].technique is None
            assert cfs[i].technique_present is None

    def test_chord_shifts_all_notes(self):
        """Chord case shifts all note MIDIs in counterfactuals."""
        case = _case(
            case_id="test/chord",
            event_kind=EventKind.CHORD,
            notes=(
                CorpusExpectedNote(midi=48, role="root"),
                CorpusExpectedNote(midi=52, role="third"),
                CorpusExpectedNote(midi=55, role="fifth"),
            ),
        )
        cfs = _counterfactuals(case)
        assert len(cfs) == 3
        for cf in cfs:
            assert cf.event_kind == EventKind.CHORD
            assert len(cf.notes) == 3
        # Offset 1: 49, 53, 56
        assert cfs[0].notes[0].midi == 49
        assert cfs[0].notes[1].midi == 53
        assert cfs[0].notes[2].midi == 56
        # Offset 2: 50, 54, 57
        assert cfs[1].notes[0].midi == 50
        assert cfs[1].notes[1].midi == 54
        assert cfs[1].notes[2].midi == 57
        # Offset 12: 60, 64, 67
        assert cfs[2].notes[0].midi == 60
        assert cfs[2].notes[1].midi == 64
        assert cfs[2].notes[2].midi == 67

    def test_counterfactual_metadata(self):
        """Counterfactuals have updated provenance and counterfactual_offset."""
        case = _case(
            case_id="test/meta",
            metadata={"original_key": "value"},
        )
        cfs = _counterfactuals(case)
        assert len(cfs) == 3
        for i, cf in enumerate(cfs):
            assert cf.metadata["provenance"] == "counterfactual"
            assert cf.metadata["counterfactual_offset"] == str((1, 2, 12)[i])
            assert cf.metadata["original_key"] == "value"

    def test_minimal_case_does_not_crash(self):
        """Counterfactuals on a minimal valid case does not crash."""
        case = CorpusCase(
            case_id="minimal",
            audio_path="/data/min.wav",
            source="Test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=0.5,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=36),),
        )
        cfs = _counterfactuals(case)
        assert len(cfs) == 3

    def test_preserves_other_fields(self):
        """Counterfactuals preserve fields not related to pitch or expectation."""
        case = _case(
            case_id="test/preserve",
            audio_path="/keep/path.wav",
            source="KeepSource",
            split=CorpusSplit.CALIBRATION,
            start_s=1.5,
            end_s=2.0,
            annotation_confidence=0.75,
        )
        cfs = _counterfactuals(case)
        for cf in cfs:
            assert cf.audio_path == "/keep/path.wav"
            assert cf.source == "KeepSource"
            assert cf.split == CorpusSplit.CALIBRATION
            assert cf.start_s == 1.5
            assert cf.end_s == 2.0
            assert cf.annotation_confidence == 0.75


# ---------------------------------------------------------------------------
# Integration: full round-trip with counterfactuals
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_round_trip(self):
        """End-to-end: events -> cases with counterfactuals, verify structure."""
        events = [
            _event(clip_id="int/a", midi=40, audio_path="/a.wav"),
            _event(
                clip_id="int/b",
                midi=None,
                notes=(
                    ClipExpectedNote(midi=48, role="root"),
                    ClipExpectedNote(midi=52, role="third"),
                ),
                audio_path="/b.wav",
            ),
            _event(clip_id="int/c", midi=45, technique="tap", audio_path="/c.wav"),
        ]
        cases = cases_from_clip_events(events, add_counterfactual_negatives=True)
        # 3 original + 9 counterfactuals = 12
        assert len(cases) == 12

        # Check original cases
        originals = [c for c in cases if "counterfactual" not in c.case_id]
        assert len(originals) == 3
        assert originals[0].event_kind == EventKind.SINGLE_NOTE
        assert originals[1].event_kind == EventKind.CHORD
        assert originals[2].event_kind == EventKind.TECHNIQUE

        # Check all counterfactuals: expected_present=False, negative_reason set
        cfs = [c for c in cases if "counterfactual" in c.case_id]
        assert len(cfs) == 9
        for cf in cfs:
            assert cf.expected_present is False
            assert cf.negative_reason == "counterfactual_expected_label"
            assert "provenance" in cf.metadata
