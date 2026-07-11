"""Metrics and slice reporting for corpus evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

from pickhero.evaluation.records import EvaluationRecord


@dataclass
class BinaryMetrics:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0

    def add(self, expected: bool, predicted: bool, weight: float = 1.0) -> None:
        if expected and predicted:
            self.tp += weight
        elif not expected and predicted:
            self.fp += weight
        elif expected and not predicted:
            self.fn += weight
        else:
            self.tn += weight

    @property
    def count(self) -> float:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def specificity(self) -> float:
        denominator = self.tn + self.fp
        return self.tn / denominator if denominator else 0.0

    @property
    def false_accept_rate(self) -> float:
        denominator = self.fp + self.tn
        return self.fp / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2.0 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.recall + self.specificity) / 2.0

    def to_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "false_accept_rate": self.false_accept_rate,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
        }


@dataclass
class Distribution:
    values: list[float] = field(default_factory=list)

    def add(self, value: float | None) -> None:
        if value is not None and math.isfinite(value):
            self.values.append(float(value))

    def to_dict(self) -> dict[str, float]:
        ordered = sorted(self.values)
        if not ordered:
            return {"count": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "count": float(len(ordered)),
            "mean": sum(ordered) / len(ordered),
            "median": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
            "max": ordered[-1],
        }


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


@dataclass
class MetricBundle:
    event: BinaryMetrics = field(default_factory=BinaryMetrics)
    onset: BinaryMetrics = field(default_factory=BinaryMetrics)
    technique: BinaryMetrics = field(default_factory=BinaryMetrics)
    absolute_onset_error_ms: Distribution = field(default_factory=Distribution)
    absolute_cents_error: Distribution = field(default_factory=Distribution)
    score_positive: Distribution = field(default_factory=Distribution)
    score_negative: Distribution = field(default_factory=Distribution)
    uncertain_techniques: float = 0.0
    technique_cases: float = 0.0
    clipped_cases: int = 0
    dc_offset_cases: int = 0

    def add(self, record: EvaluationRecord) -> None:
        weight = max(0.0, min(1.0, record.annotation_confidence))
        self.event.add(record.expected_present, record.predicted_present, weight)
        if record.expected_present:
            self.score_positive.add(record.score)
        else:
            self.score_negative.add(record.score)
        if record.onset_expected:
            self.onset.add(True, record.onset_detected, weight)
            self.absolute_onset_error_ms.add(
                abs(record.onset_error_ms) if record.onset_error_ms is not None else None
            )
        if record.cents_error is not None and record.expected_present:
            self.absolute_cents_error.add(abs(record.cents_error))
        if record.technique_expected is not None:
            self.technique_cases += weight
            if record.technique_uncertain:
                self.uncertain_techniques += weight
            elif record.technique_detected is not None:
                self.technique.add(
                    record.technique_expected,
                    record.technique_detected,
                    weight,
                )
        if record.clipped_fraction >= 0.0005:
            self.clipped_cases += 1
        if abs(record.dc_offset) >= 0.02:
            self.dc_offset_cases += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "event": self.event.to_dict(),
            "onset": self.onset.to_dict(),
            "technique": self.technique.to_dict(),
            "absolute_onset_error_ms": self.absolute_onset_error_ms.to_dict(),
            "absolute_cents_error": self.absolute_cents_error.to_dict(),
            "positive_score": self.score_positive.to_dict(),
            "negative_score": self.score_negative.to_dict(),
            "technique_uncertain_rate": (
                self.uncertain_techniques / self.technique_cases
                if self.technique_cases
                else 0.0
            ),
            "clipped_cases": self.clipped_cases,
            "dc_offset_cases": self.dc_offset_cases,
        }


def summarize_records(
    records: Iterable[EvaluationRecord],
    *,
    slice_keys: tuple[str, ...] = (
        "source",
        "event_kind",
        "technique",
        "guitar",
        "pickup",
        "interface",
        "tone",
        "tuning",
    ),
) -> dict[str, object]:
    materialized = list(records)
    overall = MetricBundle()
    for record in materialized:
        overall.add(record)

    slices: dict[str, dict[str, MetricBundle]] = {
        key: defaultdict(MetricBundle) for key in slice_keys
    }
    for record in materialized:
        for key in slice_keys:
            value = _slice_value(record, key)
            if value:
                slices[key][value].add(record)

    return {
        "case_count": len(materialized),
        "overall": overall.to_dict(),
        "slices": {
            key: {
                value: bundle.to_dict()
                for value, bundle in sorted(groups.items())
            }
            for key, groups in slices.items()
            if groups
        },
    }


def _slice_value(record: EvaluationRecord, key: str) -> str | None:
    if key == "source":
        return record.source
    if key == "event_kind":
        return record.event_kind
    if key == "technique":
        return record.technique
    return record.metadata.get(key)


def failing_records(records: Iterable[EvaluationRecord]) -> list[EvaluationRecord]:
    return [
        record
        for record in records
        if record.expected_present != record.predicted_present
        or (
            record.technique_expected is not None
            and not record.technique_uncertain
            and record.technique_detected != record.technique_expected
        )
        or (record.onset_expected and not record.onset_detected)
    ]
