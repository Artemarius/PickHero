"""Leakage-safe scalar-threshold calibration from evaluation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pickhero.evaluation.metrics import BinaryMetrics
from pickhero.evaluation.records import EvaluationRecord


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    f1: float
    precision: float
    recall: float
    false_accept_rate: float
    count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "false_accept_rate": self.false_accept_rate,
            "count": self.count,
        }


def optimize_threshold(
    examples: Iterable[tuple[float, bool, float]],
    *,
    max_false_accept_rate: float | None = None,
    minimum_recall: float | None = None,
) -> ThresholdResult:
    materialized = [
        (max(0.0, min(1.0, float(score))), bool(label), max(0.0, float(weight)))
        for score, label, weight in examples
    ]
    if not materialized:
        raise ValueError("no calibration examples")
    candidates = sorted({0.0, 1.0, *(score for score, _, _ in materialized)})
    best: ThresholdResult | None = None
    for threshold in candidates:
        metrics = BinaryMetrics()
        for score, expected, weight in materialized:
            metrics.add(expected, score >= threshold, weight)
        if (
            max_false_accept_rate is not None
            and metrics.false_accept_rate > max_false_accept_rate
        ):
            continue
        if minimum_recall is not None and metrics.recall < minimum_recall:
            continue
        candidate = ThresholdResult(
            threshold=threshold,
            f1=metrics.f1,
            precision=metrics.precision,
            recall=metrics.recall,
            false_accept_rate=metrics.false_accept_rate,
            count=len(materialized),
        )
        if best is None or _rank(candidate) > _rank(best):
            best = candidate
    if best is None:
        raise ValueError("no threshold satisfies the requested constraints")
    return best


def calibrate_records(
    records: Iterable[EvaluationRecord],
    *,
    max_false_accept_rate: float = 0.01,
) -> dict[str, dict[str, float | int]]:
    calibration = [record for record in records if record.split == "calibration"]
    if not calibration:
        raise ValueError("records contain no calibration split")
    grouped: dict[str, list[tuple[float, bool, float]]] = {}
    for record in calibration:
        key = record.event_kind
        if record.event_kind == "technique" and record.technique:
            key = f"technique:{record.technique}"
            if record.technique_quality is None or record.technique_expected is None:
                continue
            score = record.technique_quality
            expected = record.technique_expected
        elif record.chord_score is not None:
            key = "chord"
            score = record.chord_score
            expected = record.expected_present
        else:
            key = "single_note"
            score = record.score
            expected = record.expected_present
        grouped.setdefault(key, []).append(
            (score, expected, record.annotation_confidence)
        )
    return {
        key: optimize_threshold(
            examples,
            max_false_accept_rate=max_false_accept_rate,
        ).to_dict()
        for key, examples in sorted(grouped.items())
        if examples
    }


def _rank(result: ThresholdResult) -> tuple[float, float, float, float]:
    # Prefer F1, then lower false accepts, then higher recall, then the stricter
    # threshold.  Deterministic tie-breaking makes calibration reproducible.
    return (
        result.f1,
        -result.false_accept_rate,
        result.recall,
        result.threshold,
    )
