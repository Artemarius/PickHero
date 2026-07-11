"""Compatibility evaluator for the normalized public-dataset cache.

For production acceptance use ``tools/evaluate_corpus.py`` with a versioned
manifest containing recorded negatives. This command remains convenient for
quickly inspecting imported public annotations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_repo_root() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_repo_root()

from pickhero.audio.match_mode import MatchMode
from pickhero.datasets import DatasetRegistry
from pickhero.evaluation.conversion import cases_from_clip_events
from pickhero.evaluation.runner import CorpusEvaluator, EvaluationConfig


def evaluate(
    registry: DatasetRegistry,
    mode: MatchMode,
    sample_rate: int = 48000,
    limit: int | None = None,
    source_filter: str | None = None,
    negatives: bool = True,
) -> dict[str, dict[str, float]]:
    """Return legacy per-source event metrics.

    ``negatives=True`` adds explicitly tagged counterfactual expected labels.
    Those values exercise alias rejection but are not evidence of real-world
    false-accept performance.
    """
    events = registry.load_events()
    if source_filter:
        events = [event for event in events if event.source == source_filter]
    if limit is not None:
        events = events[: max(0, limit)]
    cases = cases_from_clip_events(
        events,
        calibration_fraction=0.0,
        add_counterfactual_negatives=negatives,
    )
    if not cases:
        return {}
    evaluator = CorpusEvaluator(
        EvaluationConfig(mode=mode, sample_rate=sample_rate)
    )
    run = evaluator.evaluate(cases, Path("dataset-cache.jsonl"))
    source_slices = run.summary.get("slices", {}).get("source", {})
    return {
        source: dict(metrics["event"])
        for source, metrics in source_slices.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate verifier against dataset cache")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".pickhero" / "datasets",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in MatchMode],
        default=MatchMode.ARCADE.value,
    )
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--source")
    parser.add_argument(
        "--negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    metrics = evaluate(
        DatasetRegistry(cache_dir=args.cache_dir),
        MatchMode(args.mode),
        sample_rate=args.sample_rate,
        limit=args.limit,
        source_filter=args.source,
        negatives=args.negatives,
    )
    if not metrics:
        print("No cached events found.")
        return
    print(
        f"{'Source':<18} {'Count':>7} {'TP':>7} {'FP':>7} {'FN':>7} "
        f"{'TN':>7} {'P':>7} {'R':>7} {'F1':>7} {'FA':>7}"
    )
    print("-" * 92)
    for source, values in sorted(metrics.items()):
        print(
            f"{source:<18} {values['count']:>7.0f} {values['tp']:>7.1f} "
            f"{values['fp']:>7.1f} {values['fn']:>7.1f} {values['tn']:>7.1f} "
            f"{values['precision']:>7.3f} {values['recall']:>7.3f} "
            f"{values['f1']:>7.3f} {values['false_accept_rate']:>7.2%}"
        )
    if args.negatives:
        print("\nCounterfactual negatives are smoke tests, not recorded player mistakes.")


if __name__ == "__main__":
    main()
