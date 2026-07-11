"""Tests for pickhero.evaluation.metrics — BinaryMetrics, Distribution, MetricBundle, etc."""

from __future__ import annotations

import math
from collections import defaultdict

import pytest

from pickhero.evaluation.metrics import (
    BinaryMetrics,
    Distribution,
    MetricBundle,
    _percentile,
    failing_records,
    summarize_records,
)
from pickhero.evaluation.records import EvaluationRecord


# ---------------------------------------------------------------------------
# BinaryMetrics
# ---------------------------------------------------------------------------


class TestBinaryMetrics:
    def test_precision_recall_f1(self):
        """add(10 tp, 2 fp, 8 tn, 1 fn) via direct field access."""
        m = BinaryMetrics(tp=10.0, fp=2.0, tn=8.0, fn=1.0)
        assert m.count == 21.0
        assert m.precision == pytest.approx(10.0 / 12.0)  # 10/12
        assert m.recall == pytest.approx(10.0 / 11.0)  # 10/11
        # specificity = tn / (tn + fp) = 8 / 10 = 0.8
        assert m.specificity == pytest.approx(8.0 / 10.0)
        p = 10.0 / 12.0
        r = 10.0 / 11.0
        expected_f1 = 2.0 * p * r / (p + r)
        assert m.f1 == pytest.approx(expected_f1)

    def test_empty_all_zeros(self):
        """All-zero fields yield zero metrics (no division-by-zero crashes)."""
        m = BinaryMetrics()
        assert m.count == 0.0
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.specificity == 0.0
        assert m.false_accept_rate == 0.0
        assert m.f1 == 0.0
        assert m.balanced_accuracy == 0.0

    def test_false_accept_rate(self):
        """FAR = fp / (fp + tn)."""
        m = BinaryMetrics(tp=5, fp=3, tn=7, fn=2)
        assert m.false_accept_rate == pytest.approx(3.0 / 10.0)

    def test_false_accept_rate_no_negatives(self):
        """FAR is 0 when fp+tn is 0."""
        m = BinaryMetrics(tp=5, fp=0, tn=0, fn=0)
        assert m.false_accept_rate == 0.0

    def test_balanced_accuracy(self):
        """(recall + specificity) / 2."""
        m = BinaryMetrics(tp=8, fp=2, tn=6, fn=1)
        expected = ((8 / 9) + (6 / 8)) / 2.0
        assert m.balanced_accuracy == pytest.approx(expected)

    def test_f1_edge_no_precision(self):
        """When precision is 0 (tp=0, fp>0), f1 is 0."""
        m = BinaryMetrics(tp=0, fp=5, fn=3, tn=2)
        assert m.f1 == 0.0

    def test_f1_edge_no_recall(self):
        """When recall is 0 (tp=0, fn>0), f1 is 0."""
        m = BinaryMetrics(tp=0, fp=2, fn=4, tn=1)
        assert m.f1 == 0.0

    def test_perfect_metrics(self):
        """Perfect classification: all correct, no errors."""
        m = BinaryMetrics(tp=100, fp=0, fn=0, tn=200)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.specificity == 1.0
        assert m.false_accept_rate == 0.0
        assert m.f1 == 1.0
        assert m.balanced_accuracy == 1.0

    def test_to_dict_keys(self):
        """to_dict() contains all expected keys."""
        m = BinaryMetrics(tp=1, fp=1, fn=1, tn=1)
        d = m.to_dict()
        expected_keys = {
            "count",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "specificity",
            "false_accept_rate",
            "f1",
            "balanced_accuracy",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        """to_dict() values match the live properties."""
        m = BinaryMetrics(tp=3, fp=1, tn=5, fn=1)
        d = m.to_dict()
        assert d["count"] == m.count
        assert d["precision"] == m.precision
        assert d["recall"] == m.recall
        assert d["f1"] == m.f1

    def test_add_method(self):
        """add(expected, predicted) increments the correct counter."""
        m = BinaryMetrics()
        # True positive
        m.add(True, True)
        assert m.tp == 1.0
        assert m.count == 1.0
        # False positive
        m.add(False, True)
        assert m.fp == 1.0
        assert m.count == 2.0
        # False negative
        m.add(True, False)
        assert m.fn == 1.0
        assert m.count == 3.0
        # True negative
        m.add(False, False)
        assert m.tn == 1.0
        assert m.count == 4.0

    def test_add_with_weight(self):
        """add(expected, predicted, weight) applies the weight instead of 1."""
        m = BinaryMetrics()
        m.add(True, True, weight=2.5)
        assert m.tp == 2.5
        assert m.count == 2.5

    def test_add_multiple_accumulates(self):
        """Multiple add() calls accumulate."""
        m = BinaryMetrics()
        for _ in range(5):
            m.add(True, True)
        for _ in range(3):
            m.add(False, False)
        assert m.tp == 5.0
        assert m.tn == 3.0
        assert m.count == 8.0
        assert m.precision == 1.0
        assert m.recall == 1.0


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------


class TestDistribution:
    def test_sequence(self):
        """add([1,2,3,4,5]) yields correct stats."""
        d = Distribution()
        for v in [1, 2, 3, 4, 5]:
            d.add(v)
        stats = d.to_dict()
        assert stats["count"] == 5.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        # median = _percentile(ordered, 0.5) = 3
        assert stats["median"] == 3.0
        # p95 = _percentile(ordered, 0.95)
        # position = 0.95 * 4 = 3.8  →  ordered[3]*0.2 + ordered[4]*0.8  =  4*0.2 + 5*0.8 = 4.8
        assert stats["p95"] == pytest.approx(4.8)

    def test_to_dict_keys(self):
        """to_dict() contains all expected keys (no min/std in current impl)."""
        d = Distribution()
        d.add(42)
        keys = set(d.to_dict().keys())
        expected = {"count", "max", "mean", "median", "p95"}
        assert keys == expected

    def test_empty(self):
        """Empty distribution returns safe defaults (no min/std keys)."""
        d = Distribution()
        stats = d.to_dict()
        assert stats["count"] == 0.0
        assert stats["max"] == 0.0
        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0
        assert stats["p95"] == 0.0

    def test_single_value(self):
        """Single value distribution."""
        d = Distribution()
        d.add(7.0)
        stats = d.to_dict()
        assert stats["count"] == 1.0
        assert stats["max"] == 7.0
        assert stats["mean"] == 7.0
        assert stats["median"] == 7.0

    def test_ignores_none_and_nan(self):
        """None and NaN/Inf values are filtered out."""
        d = Distribution()
        d.add(None)
        d.add(float("nan"))
        d.add(float("inf"))
        d.add(5.0)
        assert d.to_dict()["count"] == 1.0
        assert d.to_dict()["mean"] == 5.0

    def test_negative_values(self):
        """Distribution handles negative values."""
        d = Distribution()
        for v in [-5, -3, -1, -2, -4]:
            d.add(v)
        stats = d.to_dict()
        assert stats["max"] == -1.0
        assert stats["mean"] == -3.0



# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_median_odd(self):
        """P50 of [1,2,3,4,5] is 3."""
        assert _percentile([1, 2, 3, 4, 5], 0.5) == 3.0

    def test_median_even(self):
        """P50 of [1,2,3,4] is 2.5 (interpolation between 2 and 3)."""
        result = _percentile([1, 2, 3, 4], 0.5)
        # position = 0.5 * 3 = 1.5, lower=1, upper=2, alpha=0.5
        # ordered[1]*0.5 + ordered[2]*0.5 = 2*0.5 + 3*0.5 = 2.5
        assert result == 2.5

    def test_single_element(self):
        """Single-element list returns that element."""
        assert _percentile([42.0], 0.0) == 42.0
        assert _percentile([42.0], 0.5) == 42.0
        assert _percentile([42.0], 1.0) == 42.0

    def test_percentile_0(self):
        """P0 is always the first element."""
        assert _percentile([1, 2, 3, 4, 5], 0.0) == 1.0

    def test_percentile_100(self):
        """P100 is always the last element."""
        assert _percentile([1, 2, 3, 4, 5], 1.0) == 5.0

    def test_percentile_25(self):
        """P25 of [1,2,3,4,5] is 2."""
        assert _percentile([1, 2, 3, 4, 5], 0.25) == 2.0

    def test_percentile_75(self):
        """P75 of [1,2,3,4,5] is 4."""
        assert _percentile([1, 2, 3, 4, 5], 0.75) == 4.0

    def test_interpolation_fractional(self):
        """P99 of [1,2,3,4,5] interpolates near the end."""
        result = _percentile([1, 2, 3, 4, 5], 0.99)
        # position = 0.99 * 4 = 3.96, lower=3, upper=4, alpha=0.96
        # ordered[3]*0.04 + ordered[4]*0.96 = 4*0.04 + 5*0.96 = 4.96
        assert result == pytest.approx(4.96)


# ---------------------------------------------------------------------------
# MetricBundle
# ---------------------------------------------------------------------------


def _make_record(
    *,
    expected_present: bool = True,
    predicted_present: bool = True,
    score: float = 0.8,
    onset_expected: bool = True,
    onset_detected: bool = True,
    onset_error_ms: float | None = 10.0,
    cents_error: float | None = 5.0,
    technique_expected: bool | None = True,
    technique_detected: bool | None = True,
    technique_uncertain: bool | None = False,
    annotation_confidence: float = 1.0,
    clipped_fraction: float = 0.0,
    dc_offset: float = 0.0,
    source: str = "test_source",
    event_kind: str = "single",
    technique: str | None = "pick",
    **extra: object,
) -> EvaluationRecord:
    """Helper to build EvaluationRecord with sensible defaults."""
    return EvaluationRecord(
        case_id=extra.get("case_id", "test_case"),
        source=source,
        split="test",
        event_kind=event_kind,
        mode="judge",
        audio_path="test.wav",
        start_s=0.0,
        end_s=1.0,
        expected_present=expected_present,
        predicted_present=predicted_present,
        score=score,
        expected_midis=(60,),
        annotation_confidence=annotation_confidence,
        onset_expected=onset_expected,
        onset_detected=onset_detected,
        onset_error_ms=onset_error_ms,
        cents_error=cents_error,
        technique=technique,
        technique_expected=technique_expected,
        technique_detected=technique_detected,
        technique_uncertain=technique_uncertain,
        clipped_fraction=clipped_fraction,
        dc_offset=dc_offset,
        metadata={},
    )


class TestMetricBundle:
    def test_add_record_event_metrics(self):
        """add() populates event BinaryMetrics based on expected/predicted."""
        bundle = MetricBundle()
        bundle.add(_make_record(expected_present=True, predicted_present=True))
        bundle.add(_make_record(expected_present=True, predicted_present=False))
        bundle.add(_make_record(expected_present=False, predicted_present=True))
        bundle.add(_make_record(expected_present=False, predicted_present=False))
        assert bundle.event.tp == 1.0
        assert bundle.event.fn == 1.0
        assert bundle.event.fp == 1.0
        assert bundle.event.tn == 1.0

    def test_add_record_onset_metrics(self):
        """add() populates onset BinaryMetrics."""
        bundle = MetricBundle()
        # onset_expected=True, onset_detected=True
        bundle.add(_make_record(onset_expected=True, onset_detected=True))
        # onset_expected=True, onset_detected=False
        bundle.add(_make_record(onset_expected=True, onset_detected=False))
        assert bundle.onset.tp == 1.0
        assert bundle.onset.fn == 1.0

    def test_add_record_no_onset(self):
        """When onset_expected is False, onset metrics are not updated."""
        bundle = MetricBundle()
        bundle.add(_make_record(onset_expected=False, onset_detected=False))
        assert bundle.onset.tp == 0.0
        assert bundle.onset.fn == 0.0

    def test_add_record_technique_metrics(self):
        """add() populates technique BinaryMetrics."""
        bundle = MetricBundle()
        bundle.add(
            _make_record(
                technique_expected=True,
                technique_detected=True,
                technique_uncertain=False,
            )
        )
        bundle.add(
            _make_record(
                technique_expected=True,
                technique_detected=False,
                technique_uncertain=False,
            )
        )
        assert bundle.technique.tp == 1.0
        assert bundle.technique.fn == 1.0

    def test_add_record_technique_uncertain_skipped(self):
        """Uncertain technique records don't add to technique metrics."""
        bundle = MetricBundle()
        bundle.add(
            _make_record(
                technique_expected=True,
                technique_detected=False,
                technique_uncertain=True,
            )
        )
        assert bundle.technique.tp == 0.0
        assert bundle.technique.fn == 0.0
        assert bundle.uncertain_techniques == 1.0

    def test_add_record_score_distribution(self):
        """add() routes score to positive or negative distribution."""
        bundle = MetricBundle()
        bundle.add(_make_record(expected_present=True, score=0.9))
        bundle.add(_make_record(expected_present=False, score=0.1))
        assert bundle.score_positive.to_dict()["count"] == 1.0
        assert bundle.score_positive.to_dict()["mean"] == 0.9
        assert bundle.score_negative.to_dict()["count"] == 1.0
        assert bundle.score_negative.to_dict()["mean"] == 0.1

    def test_add_record_onset_error(self):
        """Absolute onset error accumulates."""
        bundle = MetricBundle()
        bundle.add(_make_record(onset_error_ms=15.0))
        bundle.add(_make_record(onset_error_ms=-8.0))
        d = bundle.absolute_onset_error_ms.to_dict()
        assert d["count"] == 2.0
        assert d["mean"] == pytest.approx((15.0 + 8.0) / 2.0)  # abs of -8

    def test_add_record_cents_error(self):
        """Absolute cents error accumulates."""
        bundle = MetricBundle()
        bundle.add(_make_record(cents_error=12.0))
        bundle.add(_make_record(cents_error=-5.0))
        d = bundle.absolute_cents_error.to_dict()
        assert d["count"] == 2.0
        assert d["mean"] == pytest.approx((12.0 + 5.0) / 2.0)

    def test_add_annotation_confidence_weights(self):
        """annotation_confidence caps weight to [0,1]."""
        bundle = MetricBundle()
        bundle.add(_make_record(annotation_confidence=0.5, predicted_present=False))
        assert bundle.event.fn == 0.5

    def test_add_clipped_dc_offset_flags(self):
        """Clipped and DC-offset cases are counted."""
        bundle = MetricBundle()
        bundle.add(_make_record(clipped_fraction=0.01))  # >= 0.0005
        bundle.add(_make_record(dc_offset=0.05))  # >= 0.02
        assert bundle.clipped_cases == 1
        assert bundle.dc_offset_cases == 1

    def test_to_dict_structure(self):
        """to_dict() returns nested dict with all metric groups."""
        bundle = MetricBundle()
        bundle.add(_make_record())
        d = bundle.to_dict()
        assert "event" in d
        assert "onset" in d
        assert "technique" in d
        assert "absolute_onset_error_ms" in d
        assert "absolute_cents_error" in d
        assert "positive_score" in d
        assert "negative_score" in d
        assert "technique_uncertain_rate" in d
        assert "clipped_cases" in d
        assert "dc_offset_cases" in d
        # Confirm nested BinaryMetrics
        for key in ("event", "onset", "technique"):
            assert "precision" in d[key]
        # Confirm nested Distribution
        for key in ("absolute_onset_error_ms", "absolute_cents_error"):
            assert "mean" in d[key]

    def test_technique_uncertain_rate(self):
        """technique_uncertain_rate = uncertain / total technique cases."""
        bundle = MetricBundle()
        bundle.add(
            _make_record(
                technique_expected=True,
                technique_detected=True,
                technique_uncertain=False,
            )
        )
        bundle.add(
            _make_record(
                technique_expected=True,
                technique_detected=False,
                technique_uncertain=False,
            )
        )
        bundle.add(
            _make_record(
                technique_expected=True,
                technique_detected=False,
                technique_uncertain=True,
            )
        )
        assert bundle.technique_cases == 3.0
        assert bundle.uncertain_techniques == 1.0
        assert bundle.to_dict()["technique_uncertain_rate"] == pytest.approx(1.0 / 3.0)

    def test_technique_uncertain_rate_zero_cases(self):
        """When no technique cases, uncertain rate is 0."""
        bundle = MetricBundle()
        assert bundle.to_dict()["technique_uncertain_rate"] == 0.0


# ---------------------------------------------------------------------------
# summarize_records
# ---------------------------------------------------------------------------


class TestSummarizeRecords:
    def test_empty_records(self):
        """Empty record list returns reasonable defaults."""
        result = summarize_records([])
        assert result["case_count"] == 0
        overall = result["overall"]
        assert overall["event"]["count"] == 0.0
        assert overall["event"]["precision"] == 0.0
        assert overall["onset"]["count"] == 0.0

    def test_single_record(self):
        """Single record produces correct aggregated metrics."""
        records = [_make_record()]
        result = summarize_records(records)
        assert result["case_count"] == 1
        assert result["overall"]["event"]["tp"] == 1.0

    def test_multiple_records(self):
        """Multiple records aggregate correctly."""
        records = [
            _make_record(
                case_id="c1",
                expected_present=True,
                predicted_present=True,
                onset_expected=True,
                onset_detected=True,
            ),
            _make_record(
                case_id="c2",
                expected_present=True,
                predicted_present=False,
                onset_expected=True,
                onset_detected=False,
            ),
            _make_record(
                case_id="c3",
                expected_present=False,
                predicted_present=True,
                onset_expected=False,
            ),
            _make_record(
                case_id="c4",
                expected_present=False,
                predicted_present=False,
                onset_expected=False,
            ),
        ]
        result = summarize_records(records)
        assert result["case_count"] == 4
        ev = result["overall"]["event"]
        assert ev["tp"] == 1.0
        assert ev["fn"] == 1.0
        assert ev["fp"] == 1.0
        assert ev["tn"] == 1.0

    def test_slicing_by_source(self):
        """Records are sliced by source key."""
        records = [
            _make_record(case_id="c1", source="a", expected_present=True, predicted_present=True),
            _make_record(case_id="c2", source="b", expected_present=True, predicted_present=False),
        ]
        result = summarize_records(records, slice_keys=("source",))
        slices = result["slices"]
        assert "source" in slices
        assert "a" in slices["source"]
        assert "b" in slices["source"]
        assert slices["source"]["a"]["event"]["tp"] == 1.0
        assert slices["source"]["b"]["event"]["fn"] == 1.0

    def test_no_slices_when_not_requested(self):
        """Without slice_keys, no slices in result."""
        records = [_make_record()]
        result = summarize_records(records, slice_keys=())
        assert result["slices"] == {}

    def test_slicing_skips_missing_values(self):
        """Slices with None/missing values are skipped."""
        records = [
            _make_record(case_id="c1", source="test", metadata={"missing_key": "val"}),
        ]
        result = summarize_records(records, slice_keys=("source", "nonexistent_key"))
        slices = result.get("slices", {})
        assert "source" in slices
        # nonexistent_key has no slice values since _slice_value returns None
        assert "nonexistent_key" not in slices


# ---------------------------------------------------------------------------
# failing_records
# ---------------------------------------------------------------------------


class TestFailingRecords:
    def test_no_failures(self):
        """All perfect records return empty list."""
        records = [
            _make_record(expected_present=True, predicted_present=True),
            _make_record(expected_present=False, predicted_present=False),
        ]
        assert failing_records(records) == []

    def test_event_mismatch(self):
        """Event prediction mismatch marks a record as failing."""
        records = [_make_record(expected_present=True, predicted_present=False)]
        failed = failing_records(records)
        assert len(failed) == 1
        assert failed[0].case_id == "test_case"

    def test_technique_mismatch(self):
        """Technique mismatch marks a record as failing."""
        records = [
            _make_record(
                technique_expected=True,
                technique_detected=False,
                technique_uncertain=False,
            )
        ]
        failed = failing_records(records)
        assert len(failed) == 1

    def test_technique_uncertain_not_failing(self):
        """Uncertain technique (no clear answer) is not a failure."""
        records = [
            _make_record(
                technique_expected=True,
                technique_detected=False,
                technique_uncertain=True,
            )
        ]
        failed = failing_records(records)
        assert len(failed) == 0

    def test_missing_onset(self):
        """Expected but undetected onset marks as failing."""
        records = [
            _make_record(onset_expected=True, onset_detected=False),
        ]
        failed = failing_records(records)
        assert len(failed) == 1

    def test_multiple_failures(self):
        """Multiple failing records returned."""
        records = [
            _make_record(case_id="c1", expected_present=True, predicted_present=True),
            _make_record(case_id="c2", expected_present=True, predicted_present=False),
            _make_record(case_id="c3", expected_present=False, predicted_present=True),
        ]
        failed = failing_records(records)
        assert len(failed) == 2
        assert {r.case_id for r in failed} == {"c2", "c3"}
