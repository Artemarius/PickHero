"""Tests for pickhero.evaluation.calibration — threshold calibration."""

import pytest

from pickhero.evaluation.calibration import (
    ThresholdResult,
    optimize_threshold,
    calibrate_records,
    _rank,
)
from pickhero.evaluation.records import EvaluationRecord


class TestThresholdResult:
    def test_construction(self):
        r = ThresholdResult(
            threshold=0.5, f1=0.8, precision=0.9, recall=0.7,
            false_accept_rate=0.02, count=100,
        )
        assert r.threshold == 0.5
        assert r.f1 == 0.8
        assert r.precision == 0.9
        assert r.recall == 0.7
        assert r.false_accept_rate == 0.02
        assert r.count == 100

    def test_frozen(self):
        import dataclasses
        r = ThresholdResult(0.5, 0.8, 0.9, 0.7, 0.02, 100)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.threshold = 0.6

    def test_to_dict(self):
        r = ThresholdResult(0.5, 0.8, 0.9, 0.7, 0.02, 100)
        d = r.to_dict()
        assert d["threshold"] == 0.5
        assert d["f1"] == 0.8
        assert d["precision"] == 0.9
        assert d["recall"] == 0.7
        assert d["false_accept_rate"] == 0.02
        assert d["count"] == 100


class TestOptimizeThreshold:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no calibration examples"):
            optimize_threshold([])

    def test_perfect_separation(self):
        # Scores: positive=0.9, negative=0.1; threshold at 0.9 gives perfect F1
        examples = [
            (0.9, True, 1.0),
            (0.1, False, 1.0),
        ]
        result = optimize_threshold(examples)
        assert result.f1 == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.count == 2

    def test_threshold_at_boundary(self):
        # All positive at 0.5, all negative at 0.3
        examples = [
            (0.5, True, 1.0),
            (0.3, False, 1.0),
        ]
        result = optimize_threshold(examples)
        assert result.threshold <= 0.5
        assert result.recall == pytest.approx(1.0)

    def test_clamps_scores(self):
        # Scores outside [0,1] are clamped
        examples = [
            (1.5, True, 1.0),  # clamped to 1.0
            (-0.5, False, 1.0),  # clamped to 0.0
        ]
        result = optimize_threshold(examples)
        assert result.f1 == pytest.approx(1.0)

    def test_max_false_accept_rate_constraint(self):
        # Very strict false accept rate
        examples = [
            (0.9, True, 1.0),
            (0.8, False, 1.0),  # this negative will be falsely accepted at threshold 0.8
        ]
        # With max_false_accept_rate=0.0, threshold 0.8 should be rejected
        result = optimize_threshold(examples, max_false_accept_rate=0.0)
        assert result.false_accept_rate == pytest.approx(0.0)

    def test_minimum_recall_constraint(self):
        examples = [
            (0.9, True, 1.0),
            (0.1, False, 1.0),
        ]
        result = optimize_threshold(examples, minimum_recall=1.0)
        assert result.recall >= 1.0

    def test_no_threshold_satisfies_constraints(self):
        examples = [
            (0.9, True, 1.0),
            (0.9, False, 1.0),  # same score, different labels
        ]
        with pytest.raises(ValueError, match="no threshold satisfies"):
            optimize_threshold(examples, max_false_accept_rate=0.0, minimum_recall=1.0)

    def test_weight_awareness(self):
        # Higher weight on false accept should push threshold higher
        examples = [
            (0.6, True, 1.0),
            (0.5, False, 10.0),  # heavily weighted negative
        ]
        result = optimize_threshold(examples, max_false_accept_rate=0.05)
        # Should pick a threshold above 0.5 to avoid the weighted false accept
        assert result.threshold > 0.5

    def test_includes_boundary_values(self):
        # 0.0 and 1.0 are always candidate thresholds
        examples = [
            (0.5, True, 1.0),
            (0.5, False, 1.0),
        ]
        result = optimize_threshold(examples)
        # At threshold=1.0, nothing is predicted positive → recall=0, but also no false accepts
        # At threshold=0.0, everything is predicted positive → recall=1, but false accepts
        # Best F1 should be at 0.0 (catches all true positives)
        assert result.threshold in (0.0, 0.5, 1.0)


class TestCalibrateRecords:
    def _make_record(
        self, *, split="calibration", event_kind="single_note",
        score=0.8, expected_present=True, technique=None,
        technique_quality=None, technique_expected=None,
        chord_score=None,
    ):
        return EvaluationRecord(
            case_id="test",
            source="test",
            split=split,
            event_kind=event_kind,
            mode="judge",
            audio_path="/dev/null",
            start_s=0.0,
            end_s=1.0,
            expected_present=expected_present,
            predicted_present=score >= 0.5,
            score=score,
            expected_midis=(40,),
            annotation_confidence=1.0,
            metadata={},
            technique=technique,
            technique_quality=technique_quality,
            technique_expected=technique_expected,
            chord_score=chord_score,
            details={},
        )

    def test_no_calibration_split_raises(self):
        records = [self._make_record(split="test")]
        with pytest.raises(ValueError, match="no calibration split"):
            calibrate_records(records)

    def test_single_note_calibration(self):
        records = [
            self._make_record(score=0.9, expected_present=True),
            self._make_record(score=0.1, expected_present=False),
        ]
        result = calibrate_records(records)
        assert "single_note" in result
        assert result["single_note"]["count"] == 2

    def test_chord_calibration(self):
        records = [
            self._make_record(
                event_kind="chord", chord_score=0.9, expected_present=True,
                score=0.5,
            ),
            self._make_record(
                event_kind="chord", chord_score=0.1, expected_present=False,
                score=0.5,
            ),
        ]
        result = calibrate_records(records)
        assert "chord" in result

    def test_technique_calibration(self):
        records = [
            self._make_record(
                event_kind="technique", technique="bend",
                technique_quality=0.9, technique_expected=True,
                score=0.5,
            ),
            self._make_record(
                event_kind="technique", technique="bend",
                technique_quality=0.1, technique_expected=False,
                score=0.5,
            ),
        ]
        result = calibrate_records(records)
        assert "technique:bend" in result

    def test_technique_skipped_when_quality_none(self):
        records = [
            self._make_record(
                event_kind="technique", technique="bend",
                technique_quality=None, technique_expected=True,
                score=0.5,
            ),
        ]
        result = calibrate_records(records)
        # The technique with None quality is skipped, but single_note fallback
        # depends on chord_score being None (which it is)
        assert "technique:bend" not in result

    def test_multiple_groups(self):
        records = [
            self._make_record(score=0.9, expected_present=True),
            self._make_record(score=0.1, expected_present=False),
            self._make_record(
                event_kind="chord", chord_score=0.8, expected_present=True,
                score=0.5,
            ),
            self._make_record(
                event_kind="chord", chord_score=0.2, expected_present=False,
                score=0.5,
            ),
        ]
        result = calibrate_records(records)
        assert "single_note" in result
        assert "chord" in result

    def test_max_false_accept_rate_param(self):
        records = [
            self._make_record(score=0.9, expected_present=True),
            self._make_record(score=0.8, expected_present=False),
        ]
        result = calibrate_records(records, max_false_accept_rate=0.0)
        assert result["single_note"]["false_accept_rate"] == pytest.approx(0.0)


class TestRank:
    def test_higher_f1_wins(self):
        a = ThresholdResult(0.3, 0.9, 0.9, 0.9, 0.02, 100)
        b = ThresholdResult(0.5, 0.7, 0.8, 0.6, 0.01, 100)
        assert _rank(a) > _rank(b)

    def test_lower_false_accept_wins_on_tie(self):
        a = ThresholdResult(0.3, 0.8, 0.9, 0.7, 0.01, 100)
        b = ThresholdResult(0.5, 0.8, 0.9, 0.7, 0.05, 100)
        assert _rank(a) > _rank(b)

    def test_higher_recall_wins_on_tie(self):
        a = ThresholdResult(0.3, 0.8, 0.9, 0.9, 0.01, 100)
        b = ThresholdResult(0.5, 0.8, 0.9, 0.7, 0.01, 100)
        assert _rank(a) > _rank(b)

    def test_stricter_threshold_wins_on_tie(self):
        a = ThresholdResult(0.5, 0.8, 0.9, 0.7, 0.01, 100)
        b = ThresholdResult(0.3, 0.8, 0.9, 0.7, 0.01, 100)
        assert _rank(a) > _rank(b)
