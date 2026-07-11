"""Tests for pickhero.evaluation.records — EvaluationRecord dataclass."""

import numpy as np
import pytest

from pickhero.evaluation.records import EvaluationRecord


class TestEvaluationRecord:
    """EvaluationRecord is a serializable dataclass for per-case observations."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _minimal_kwargs() -> dict:
        """Return keyword arguments that satisfy all required fields."""
        return dict(
            case_id="case-001",
            source="test_source",
            split="train",
            event_kind="note",
            mode="judge",
            audio_path="/audio/test.wav",
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            predicted_present=False,
            score=0.85,
            expected_midis=(40, 42),
            annotation_confidence=1.0,
        )

    @staticmethod
    def _full_kwargs() -> dict:
        """Return keyword arguments covering every field."""
        return dict(
            case_id="case-002",
            source="test_source_2",
            split="test",
            event_kind="chord",
            mode="cqt",
            audio_path="/audio/test2.wav",
            start_s=0.5,
            end_s=2.0,
            expected_present=False,
            predicted_present=True,
            score=0.25,
            expected_midis=(36, 40, 43),
            annotation_confidence=0.9,
            metadata={"transcriber": "jose"},
            onset_expected=True,
            onset_detected=True,
            onset_error_ms=3.2,
            cents_error=12.5,
            alias_risk=0.1,
            chord_verdict="correct",
            chord_score=0.95,
            missing_roles=("bass",),
            extra_pitch_classes=(4,),
            technique="palm_mute",
            technique_expected=True,
            technique_detected=True,
            technique_uncertain=False,
            technique_quality=0.88,
            peak_dbfs=-3.0,
            rms_dbfs=-18.5,
            dc_offset=0.001,
            clipped_fraction=0.0,
            failure_reasons=(),
            details={"confidence_baseline": 0.7},
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_construction_required_fields_only(self):
        """EvaluationRecord can be created with only required fields."""
        kwargs = self._minimal_kwargs()
        rec = EvaluationRecord(**kwargs)
        assert rec.case_id == "case-001"
        assert rec.source == "test_source"
        assert rec.split == "train"
        assert rec.event_kind == "note"
        assert rec.mode == "judge"
        assert rec.audio_path == "/audio/test.wav"
        assert rec.start_s == 0.0
        assert rec.end_s == 1.0
        assert rec.expected_present is True
        assert rec.predicted_present is False
        assert rec.score == 0.85
        assert rec.expected_midis == (40, 42)
        assert rec.annotation_confidence == 1.0

    def test_construction_defaults(self):
        """Optional fields get expected default values."""
        rec = EvaluationRecord(**self._minimal_kwargs())
        assert rec.metadata == {}
        assert rec.onset_expected is False
        assert rec.onset_detected is False
        assert rec.onset_error_ms is None
        assert rec.cents_error is None
        assert rec.alias_risk is None
        assert rec.chord_verdict is None
        assert rec.chord_score is None
        assert rec.missing_roles == ()
        assert rec.extra_pitch_classes == ()
        assert rec.technique is None
        assert rec.technique_expected is None
        assert rec.technique_detected is None
        assert rec.technique_uncertain is None
        assert rec.technique_quality is None
        assert rec.peak_dbfs == -120.0
        assert rec.rms_dbfs == -120.0
        assert rec.dc_offset == 0.0
        assert rec.clipped_fraction == 0.0
        assert rec.failure_reasons == ()
        assert rec.details == {}

    def test_construction_all_fields(self):
        """EvaluationRecord can be created with every field populated."""
        kwargs = self._full_kwargs()
        rec = EvaluationRecord(**kwargs)
        for key, value in kwargs.items():
            assert getattr(rec, key) == value, f"Mismatch for field {key!r}"

    # ------------------------------------------------------------------
    # to_dict
    # ------------------------------------------------------------------

    def test_to_dict_returns_all_fields(self):
        """to_dict() returns a dict with all dataclass fields present."""
        rec = EvaluationRecord(**self._full_kwargs())
        d = rec.to_dict()
        fields = {
            "case_id",
            "source",
            "split",
            "event_kind",
            "mode",
            "audio_path",
            "start_s",
            "end_s",
            "expected_present",
            "predicted_present",
            "score",
            "expected_midis",
            "annotation_confidence",
            "metadata",
            "onset_expected",
            "onset_detected",
            "onset_error_ms",
            "cents_error",
            "alias_risk",
            "chord_verdict",
            "chord_score",
            "missing_roles",
            "extra_pitch_classes",
            "technique",
            "technique_expected",
            "technique_detected",
            "technique_uncertain",
            "technique_quality",
            "peak_dbfs",
            "rms_dbfs",
            "dc_offset",
            "clipped_fraction",
            "failure_reasons",
            "details",
        }
        assert set(d.keys()) == fields

    def test_to_dict_values_are_json_serializable(self):
        """All values returned by to_dict() are JSON-serializable types (str, int, float, bool, None, list, dict)."""
        rec = EvaluationRecord(**self._full_kwargs())
        d = rec.to_dict()
        self._check_json_serializable(d)

    @staticmethod
    def _check_json_serializable(obj: object) -> None:
        """Recursively verify obj contains only JSON-safe types."""
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return
        if isinstance(obj, list):
            for item in obj:
                TestEvaluationRecord._check_json_serializable(item)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                assert isinstance(key, str), f"dict key {key!r} is not str"
                TestEvaluationRecord._check_json_serializable(value)
            return
        pytest.fail(f"non-JSON-serializable type: {type(obj).__name__}: {obj!r}")

    # ------------------------------------------------------------------
    # from_dict
    # ------------------------------------------------------------------

    def test_from_dict(self):
        """from_dict() reconstructs EvaluationRecord from a dict."""
        d = {
            "case_id": "case-003",
            "source": "from_dict_test",
            "split": "val",
            "event_kind": "note",
            "mode": "detect",
            "audio_path": "/audio/a.wav",
            "start_s": 0.1,
            "end_s": 0.9,
            "expected_present": True,
            "predicted_present": True,
            "score": 0.99,
            "expected_midis": (45, 47),
            "annotation_confidence": 0.8,
            "metadata": {"env": "test"},
        }
        rec = EvaluationRecord.from_dict(d)
        assert rec.case_id == "case-003"
        assert rec.expected_midis == (45, 47)
        assert rec.metadata == {"env": "test"}
        assert rec.score == 0.99

    def test_from_dict_converts_lists_to_tuples(self):
        """from_dict() converts list-typed fields to tuples for the dataclass."""
        d = {
            **self._minimal_kwargs(),
            "expected_midis": [40, 41, 42],
            "missing_roles": ["bass"],
            "extra_pitch_classes": [5, 6],
            "failure_reasons": ["low_confidence"],
        }
        rec = EvaluationRecord.from_dict(d)
        assert rec.expected_midis == (40, 41, 42)
        assert rec.missing_roles == ("bass",)
        assert rec.extra_pitch_classes == (5, 6)
        assert rec.failure_reasons == ("low_confidence",)

    def test_from_dict_handles_existing_tuples(self):
        """from_dict() does NOT double-wrap fields that are already tuples."""
        d = {
            **self._minimal_kwargs(),
            "expected_midis": (40,),
            "failure_reasons": ("err",),
        }
        rec = EvaluationRecord.from_dict(d)
        assert rec.expected_midis == (40,)
        assert rec.failure_reasons == ("err",)

    def test_from_dict_missing_fields_use_defaults(self):
        """from_dict() applies dataclass defaults for omitted fields."""
        d = {
            "case_id": "partial",
            "source": "src",
            "split": "train",
            "event_kind": "note",
            "mode": "judge",
            "audio_path": "/p.wav",
            "start_s": 0.0,
            "end_s": 1.0,
            "expected_present": False,
            "predicted_present": False,
            "score": 0.0,
            "expected_midis": (),
            "annotation_confidence": 0.0,
        }
        rec = EvaluationRecord.from_dict(d)
        assert rec.peak_dbfs == -120.0  # default
        assert rec.onset_expected is False
        assert rec.technique is None
        assert rec.metadata == {}
        assert rec.details == {}


    def test_from_dict_extra_fields_cause_error(self):
        """from_dict() passes through all keys — unrecognized fields raise TypeError.

        Note: from_dict does NOT filter unknown keys. Extra fields in the input
        dict will be forwarded directly to EvaluationRecord.__init__(), which
        rejects them. This test documents that contract.
        """
        d = {
            **self._minimal_kwargs(),
            "imaginary_flag": True,
        }
        with pytest.raises(TypeError, match="imaginary_flag"):
            EvaluationRecord.from_dict(d)

    def test_round_trip_minimal(self):
        """Minimal record survives to_dict → from_dict with identical content."""
        original = EvaluationRecord(**self._minimal_kwargs())
        d = original.to_dict()
        reconstructed = EvaluationRecord.from_dict(d)
        assert reconstructed == original

    def test_round_trip_full(self):
        """Full record survives to_dict → from_dict with identical content."""
        original = EvaluationRecord(**self._full_kwargs())
        d = original.to_dict()
        reconstructed = EvaluationRecord.from_dict(d)
        assert reconstructed == original

    def test_round_trip_with_metadata(self):
        """Metadata dict survives the round-trip unchanged."""
        original = EvaluationRecord(
            **{**self._minimal_kwargs(), "metadata": {"key": "value", "nested": "ok"}}
        )
        d = original.to_dict()
        reconstructed = EvaluationRecord.from_dict(d)
        assert reconstructed.metadata == {"key": "value", "nested": "ok"}
        assert reconstructed == original

    def test_round_trip_with_details(self):
        """Details dict (with heterogeneous values) survives the round-trip."""
        original = EvaluationRecord(
            **{
                **self._minimal_kwargs(),
                "details": {"config": {"threshold": 0.5}, "tags": ["a", 1]},
            }
        )
        d = original.to_dict()
        reconstructed = EvaluationRecord.from_dict(d)
        assert reconstructed.details == {"config": {"threshold": 0.5}, "tags": ["a", 1]}
        assert reconstructed == original

    # ------------------------------------------------------------------
    # _jsonable helper (numpy → JSON-safe)
    # ------------------------------------------------------------------

    def test_jsonable_numpy_int64(self):
        """numpy int64 converts to Python int via _jsonable."""
        record = EvaluationRecord(
            **{**self._minimal_kwargs(), "score": np.int64(42)}
        )
        d = record.to_dict()
        assert d["score"] == 42
        assert isinstance(d["score"], int)

    def test_jsonable_numpy_float64(self):
        """numpy float64 converts to Python float via _jsonable."""
        record = EvaluationRecord(
            **{
                **self._minimal_kwargs(),
                "score": np.float64(0.875),
                "annotation_confidence": np.float64(1.0),
            }
        )
        d = record.to_dict()
        assert d["score"] == 0.875
        assert d["annotation_confidence"] == 1.0
        assert isinstance(d["score"], float)
        assert isinstance(d["annotation_confidence"], float)


    def test_jsonable_numpy_bool(self):
        """numpy bool_ converts to Python bool via _jsonable."""
        record = EvaluationRecord(
            **{
                **self._minimal_kwargs(),
                "expected_present": np.bool_(True),
                "predicted_present": np.bool_(False),
            }
        )
        d = record.to_dict()
        assert d["expected_present"] is True
        assert d["predicted_present"] is False
        assert isinstance(d["expected_present"], bool)
        assert isinstance(d["predicted_present"], bool)

    def test_jsonable_numpy_ndarray_single_element(self):
        """A single-element numpy ndarray is converted to a Python scalar via .item()."""
        record = EvaluationRecord(
            **{
                **self._minimal_kwargs(),
                "score": np.array([0.99]),
            }
        )
        d = record.to_dict()
        assert d["score"] == 0.99
        assert isinstance(d["score"], float)

    def test_jsonable_numpy_scalar_list_converted(self):
        """numpy scalars inside a plain Python list in details are converted recursively."""
        record = EvaluationRecord(
            **{
                **self._minimal_kwargs(),
                "details": {"vals": [np.float32(0.1), np.float64(0.2)]},
            }
        )
        d = record.to_dict()
        vals = d["details"]["vals"]
        assert isinstance(vals, list)
        assert vals[0] == pytest.approx(0.1, abs=1e-6)
        assert vals[1] == pytest.approx(0.2)
    def test_jsonable_numpy_scalar_in_metadata(self):
        """numpy scalars nested in metadata are converted."""
        record = EvaluationRecord(
            **{
                **self._minimal_kwargs(),
                "metadata": {"gain": np.float32(0.5)},
            }
        )
        d = record.to_dict()
        assert d["metadata"]["gain"] == 0.5
        assert isinstance(d["metadata"]["gain"], float)


    def test_jsonable_empty_tuple_preserved(self):
        """Empty tuples remain as empty lists in JSON output."""
        record = EvaluationRecord(**self._minimal_kwargs())
        d = record.to_dict()
        assert d["expected_midis"] == [40, 42]

    # ------------------------------------------------------------------
    # Equality & identity
    # ------------------------------------------------------------------

    def test_equality_equal_records(self):
        """Two records with identical fields are equal."""
        a = EvaluationRecord(**self._minimal_kwargs())
        b = EvaluationRecord(**self._minimal_kwargs())
        assert a == b
        assert not (a != b)

    def test_equality_different_records(self):
        """Two records with differing fields are not equal."""
        a = EvaluationRecord(**self._minimal_kwargs())
        b = EvaluationRecord(**{**self._minimal_kwargs(), "score": 0.5})
        assert a != b

    def test_immutable_sequence_types(self):
        """Tuple fields are immutable — confirms expected_midis, missing_roles etc are tuples."""
        rec = EvaluationRecord(**self._minimal_kwargs())
        assert isinstance(rec.expected_midis, tuple)
        assert isinstance(rec.missing_roles, tuple)
        assert isinstance(rec.extra_pitch_classes, tuple)
        assert isinstance(rec.failure_reasons, tuple)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_expected_midis(self):
        """Empty tuple for expected_midis is valid."""
        rec = EvaluationRecord(**{**self._minimal_kwargs(), "expected_midis": ()})
        assert rec.expected_midis == ()
        assert rec.score == 0.85

    def test_all_none_optionals(self):
        """Optional fields can all be None simultaneously."""
        rec = EvaluationRecord(**self._minimal_kwargs())
        for field_name in (
            "onset_error_ms",
            "cents_error",
            "alias_risk",
            "chord_verdict",
            "chord_score",
            "technique",
            "technique_expected",
            "technique_detected",
            "technique_uncertain",
            "technique_quality",
        ):
            assert getattr(rec, field_name) is None, f"{field_name} should be None"

    def test_from_dict_preserves_exact_types(self):
        """from_dict keeps ints as ints, floats as floats after tuple conversion."""
        d = {
            **self._minimal_kwargs(),
            "expected_midis": [40, 41],
            "start_s": 0.0,
            "score": 1,
        }
        rec = EvaluationRecord.from_dict(d)
        assert rec.expected_midis == (40, 41)
        assert isinstance(rec.expected_midis[0], int)
        assert rec.start_s == 0.0
        assert isinstance(rec.start_s, float)

    def test_path_value_in_details(self):
        """A pathlib.Path or datetime value stored in details gets stringified by _jsonable."""
        import pathlib

        rec = EvaluationRecord(
            **{
                **self._minimal_kwargs(),
                "details": {"config_path": pathlib.Path("/tmp/test.cfg")},
            }
        )
        d = rec.to_dict()
        assert d["details"]["config_path"] == "/tmp/test.cfg"
        assert isinstance(d["details"]["config_path"], str)

    def test_from_dict_mutates_nothing_shared(self):
        """from_dict copies the input dict and does not mutate it."""
        source = {**self._minimal_kwargs(), "expected_midis": [60, 61, 62]}
        expected_midis_copy = list(source["expected_midis"])
        rec = EvaluationRecord.from_dict(source)
        assert rec.expected_midis == (60, 61, 62)
        # original list should be untouched
        assert source["expected_midis"] == expected_midis_copy
        assert isinstance(source["expected_midis"], list)
