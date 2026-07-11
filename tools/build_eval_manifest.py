"""Build a versioned evaluator manifest from the normalized dataset cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_repo_root() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_repo_root()

from pickhero.datasets import DatasetRegistry
from pickhero.evaluation.conversion import cases_from_clip_events
from pickhero.evaluation.manifest import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PickHero's dataset cache to an evaluation manifest"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".pickhero" / "datasets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/evaluation/corpus.jsonl"),
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.30,
        help="Fraction of source recordings reserved for threshold calibration",
    )
    parser.add_argument(
        "--split-group",
        choices=("audio_path", "player"),
        default="audio_path",
        help="Keep each recording or each player entirely in one split",
    )
    parser.add_argument(
        "--counterfactual-negatives",
        action="store_true",
        help=(
            "Add transposed expected labels as smoke-test negatives. "
            "These are tagged and do not replace recorded mistakes."
        ),
    )
    args = parser.parse_args()

    registry = DatasetRegistry(cache_dir=args.cache_dir)
    events = registry.load_events()
    if not events:
        parser.error("dataset cache is empty; import datasets first")
    cases = cases_from_clip_events(
        events,
        calibration_fraction=args.calibration_fraction,
        add_counterfactual_negatives=args.counterfactual_negatives,
        split_group=args.split_group,
    )
    write_manifest(args.output, cases)
    calibration = sum(case.split.value == "calibration" for case in cases)
    test = sum(case.split.value == "test" for case in cases)
    counterfactual = sum(
        case.metadata.get("provenance") == "counterfactual" for case in cases
    )
    print(f"Wrote {len(cases)} cases to {args.output}")
    print(f"Calibration: {calibration}; test: {test}; counterfactual: {counterfactual}")


if __name__ == "__main__":
    main()
