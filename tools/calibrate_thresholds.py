"""Find leakage-safe score thresholds using only the calibration split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_root() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_repo_root()

from pickhero.evaluation.calibration import calibrate_records
from pickhero.evaluation.records import EvaluationRecord


def _read_records(path: Path) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(EvaluationRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate detector thresholds without touching the held-out split"
    )
    parser.add_argument("records", type=Path, help="records.jsonl from a calibration run")
    parser.add_argument(
        "--max-false-accept-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation-results/calibrated-thresholds.json"),
    )
    args = parser.parse_args()

    records = _read_records(args.records)
    thresholds = calibrate_records(
        records,
        max_false_accept_rate=args.max_false_accept_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(thresholds, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for task, result in thresholds.items():
        print(
            f"{task:<24} threshold={result['threshold']:.3f} "
            f"f1={result['f1']:.3f} "
            f"recall={result['recall']:.3f} "
            f"false_accept={result['false_accept_rate']:.2%}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
