"""Fail a corpus run when declared production-quality gates are not met."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _lookup(payload: dict[str, Any], path: str) -> float:
    value: Any = payload
    for segment in path.split("."):
        if not isinstance(value, dict) or segment not in value:
            raise KeyError(path)
        value = value[segment]
    if not isinstance(value, (int, float)):
        raise TypeError(f"gate value {path!r} is not numeric")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check evaluation summary quality gates")
    parser.add_argument("summary", type=Path)
    parser.add_argument(
        "--gates",
        type=Path,
        default=Path("research/evaluation/quality-gates.json"),
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    gate_config = json.loads(args.gates.read_text(encoding="utf-8"))
    failures: list[str] = []
    for gate in gate_config.get("gates", []):
        path = str(gate["path"])
        actual = _lookup(summary, path)
        if "min" in gate and actual < float(gate["min"]):
            failures.append(f"{path}: {actual:.6g} < minimum {gate['min']}")
        if "max" in gate and actual > float(gate["max"]):
            failures.append(f"{path}: {actual:.6g} > maximum {gate['max']}")

    if failures:
        print("Quality gates failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print(f"All {len(gate_config.get('gates', []))} quality gates passed.")


if __name__ == "__main__":
    main()
