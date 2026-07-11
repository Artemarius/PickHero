"""Compare two evaluation runs and fail on material regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict[str, Any], path: str) -> float:
    value: Any = payload
    for segment in path.split("."):
        value = value[segment]
    return float(value)


def _compare_bundle(
    name: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_f1_drop: float,
    max_false_accept_increase: float,
    max_onset_p95_increase_ms: float,
    minimum_count: float,
) -> list[str]:
    failures: list[str] = []
    base_count = _metric(baseline, "event.count")
    candidate_count = _metric(candidate, "event.count")
    if base_count < minimum_count or candidate_count < minimum_count:
        return failures
    base_f1 = _metric(baseline, "event.f1")
    candidate_f1 = _metric(candidate, "event.f1")
    if candidate_f1 < base_f1 - max_f1_drop:
        failures.append(
            f"{name}: event F1 {base_f1:.4f} -> {candidate_f1:.4f}"
        )
    base_far = _metric(baseline, "event.false_accept_rate")
    candidate_far = _metric(candidate, "event.false_accept_rate")
    if candidate_far > base_far + max_false_accept_increase:
        failures.append(
            f"{name}: false-accept rate {base_far:.3%} -> {candidate_far:.3%}"
        )
    base_p95 = _metric(baseline, "absolute_onset_error_ms.p95")
    candidate_p95 = _metric(candidate, "absolute_onset_error_ms.p95")
    if candidate_p95 > base_p95 + max_onset_p95_increase_ms:
        failures.append(
            f"{name}: onset p95 {base_p95:.1f}ms -> {candidate_p95:.1f}ms"
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evaluation summaries")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-f1-drop", type=float, default=0.005)
    parser.add_argument("--max-false-accept-increase", type=float, default=0.002)
    parser.add_argument("--max-onset-p95-increase-ms", type=float, default=3.0)
    parser.add_argument("--minimum-slice-count", type=float, default=20.0)
    parser.add_argument(
        "--allow-missing-slices",
        action="store_true",
        help="Do not fail if a baseline slice disappears from the candidate",
    )
    args = parser.parse_args()

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    failures = _compare_bundle(
        "overall",
        baseline["overall"],
        candidate["overall"],
        max_f1_drop=args.max_f1_drop,
        max_false_accept_increase=args.max_false_accept_increase,
        max_onset_p95_increase_ms=args.max_onset_p95_increase_ms,
        minimum_count=0.0,
    )

    baseline_slices = baseline.get("slices", {})
    candidate_slices = candidate.get("slices", {})
    for dimension, values in baseline_slices.items():
        candidate_values = candidate_slices.get(dimension, {})
        for value, base_bundle in values.items():
            candidate_bundle = candidate_values.get(value)
            if candidate_bundle is None:
                if not args.allow_missing_slices:
                    failures.append(f"{dimension}={value}: slice missing in candidate")
                continue
            failures.extend(
                _compare_bundle(
                    f"{dimension}={value}",
                    base_bundle,
                    candidate_bundle,
                    max_f1_drop=args.max_f1_drop,
                    max_false_accept_increase=args.max_false_accept_increase,
                    max_onset_p95_increase_ms=args.max_onset_p95_increase_ms,
                    minimum_count=args.minimum_slice_count,
                )
            )

    if failures:
        print("Evaluation regressions detected:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("No material overall or slice regression detected.")


if __name__ == "__main__":
    main()
