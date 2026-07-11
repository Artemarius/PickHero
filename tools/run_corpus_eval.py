"""Scan configured datasets, update the cache, and run the corpus evaluation.

Usage:
    python tools/run_corpus_eval.py --goat /path/to/GOAT --guitarset /path/to/GuitarSet
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan datasets and evaluate verifier")
    parser.add_argument("--cache-dir", type=Path, help="Cache directory")
    parser.add_argument("--goat", type=Path, help="Path to GOAT dataset")
    parser.add_argument("--guitarset", type=Path, help="Path to GuitarSet dataset")
    parser.add_argument("--guitar-techs", type=Path, help="Path to Guitar-TECHS dataset")
    parser.add_argument("--idmt", type=Path, help="Path to IDMT dataset")
    parser.add_argument(
        "--mode",
        choices=["arcade", "practice", "judge"],
        default="arcade",
        help="Match mode / verification policy",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Sample rate used by the verifier",
    )
    args = parser.parse_args()

    kwargs: dict = {"cache_dir": args.cache_dir} if args.cache_dir else {}
    registry = DatasetRegistry(**kwargs)

    paths: dict[str, Path] = {}
    if args.goat:
        paths["GOAT"] = args.goat
    if args.guitarset:
        paths["GuitarSet"] = args.guitarset
    if args.guitar_techs:
        paths["Guitar-TECHS"] = args.guitar_techs
    if args.idmt:
        paths["IDMT"] = args.idmt

    if paths:
        for source, path in paths.items():
            registry.set_path(source, path)
        print(f"Scanning {len(paths)} dataset(s)...")
        count = len(registry.scan_datasets())
        print(f"Cached {count} event(s)")
    else:
        print("No dataset paths provided; using existing cache.")

    # Import after sys.path is set so the tool works when invoked directly.
    from tools.eval_detector import evaluate

    mode = MatchMode(args.mode)
    metrics = evaluate(registry, mode, sample_rate=args.sample_rate)

    if not metrics:
        print("No events to evaluate.")
        sys.exit(1)

    print(f"\nMode: {args.mode}")
    print(f"{'Source':<18} {'Count':>7} {'TP':>7} {'FP':>7} {'FN':>7} {'P':>7} {'R':>7} {'F1':>7} {'FA':>7}")
    print("-" * 88)
    for source, m in sorted(metrics.items()):
        print(
            f"{source:<18} {m['count']:>7.0f} {m['tp']:>7.1f} {m['fp']:>7.1f} {m['fn']:>7.1f} "
            f"{m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
            f"{m['false_accept_rate']:>7.2%}"
        )


if __name__ == "__main__":
    main()
