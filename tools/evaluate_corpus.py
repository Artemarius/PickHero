"""Evaluate the production verifier against a versioned JSONL corpus."""

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
from pickhero.evaluation.manifest import CorpusSplit, load_manifest
from pickhero.evaluation.runner import CorpusEvaluator, EvaluationConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PickHero against a held-out guitar corpus"
    )
    parser.add_argument("manifest", type=Path, help="Corpus JSONL manifest")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation-results"),
        help="Directory for records, failures, summary and Markdown report",
    )
    parser.add_argument(
        "--split",
        choices=[split.value for split in CorpusSplit],
        default=CorpusSplit.TEST.value,
        help="Manifest split to evaluate",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in MatchMode],
        default=MatchMode.JUDGE.value,
    )
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--onset-tolerance-ms", type=float, default=45.0)
    parser.add_argument("--silence-threshold-db", type=float, default=-55.0)
    parser.add_argument(
        "--require-onset",
        action="store_true",
        help="Require onset evidence for the event-level pitch/chord verdict",
    )
    parser.add_argument("--source", help="Only score one source")
    parser.add_argument("--limit", type=int, help="Limit cases after filtering")
    args = parser.parse_args()

    split = CorpusSplit(args.split)
    cases = load_manifest(args.manifest, split=split)
    if args.source:
        cases = [case for case in cases if case.source == args.source]
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    if not cases:
        parser.error("no corpus cases match the requested split and filters")

    evaluator = CorpusEvaluator(
        EvaluationConfig(
            mode=MatchMode(args.mode),
            sample_rate=args.sample_rate,
            onset_tolerance_ms=args.onset_tolerance_ms,
            silence_threshold_db=args.silence_threshold_db,
            require_onset_for_event=args.require_onset,
        )
    )
    run = evaluator.evaluate(cases, args.manifest)
    run.write(args.output)

    overall = run.summary["overall"]
    event = overall["event"]
    onset = overall["absolute_onset_error_ms"]
    technique = overall["technique"]
    print(f"Cases: {run.summary['case_count']}")
    print(
        "Event: "
        f"precision={event['precision']:.3f} "
        f"recall={event['recall']:.3f} "
        f"f1={event['f1']:.3f} "
        f"false_accept={event['false_accept_rate']:.2%}"
    )
    print(
        "Onset absolute error: "
        f"median={onset['median']:.1f}ms p95={onset['p95']:.1f}ms"
    )
    if technique["count"]:
        print(
            "Technique: "
            f"precision={technique['precision']:.3f} "
            f"recall={technique['recall']:.3f} "
            f"f1={technique['f1']:.3f}"
        )
    print(f"Report: {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
