"""Validate corpus coverage, labels, files, and split isolation."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path


def _ensure_repo_root() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_repo_root()

from pickhero.evaluation.manifest import EventKind, load_manifest


_REQUIRED_METADATA = ("guitar", "pickup", "interface", "tone", "tuning", "player")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an evaluation corpus")
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--allow-missing-audio",
        action="store_true",
        help="Validate a capture plan whose WAV files do not exist yet",
    )
    parser.add_argument(
        "--require-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--group-key",
        action="append",
        default=["audio_path"],
        help=(
            "Field that must not cross calibration/test splits. Use a top-level "
            "field such as audio_path or a metadata key such as player."
        ),
    )
    parser.add_argument("--minimum-real-negatives", type=int, default=1)
    args = parser.parse_args()

    cases = load_manifest(args.manifest)
    errors: list[str] = []
    warnings: list[str] = []
    kind_counts = collections.Counter(case.event_kind.value for case in cases)
    split_counts = collections.Counter(case.split.value for case in cases)
    real_negatives = 0
    technique_labels: dict[str, set[bool]] = collections.defaultdict(set)

    for case in cases:
        path = case.resolve_audio_path(args.manifest)
        if not args.allow_missing_audio and not path.exists():
            errors.append(f"{case.case_id}: missing audio {path}")
        if args.require_metadata:
            missing = [key for key in _REQUIRED_METADATA if not case.metadata.get(key)]
            if missing:
                errors.append(
                    f"{case.case_id}: missing metadata {', '.join(missing)}"
                )
        if not case.expected_present:
            if case.metadata.get("provenance") != "counterfactual":
                real_negatives += 1
        if case.technique and case.technique_present is not None:
            technique_labels[case.technique].add(case.technique_present)

    for group_key in dict.fromkeys(args.group_key):
        groups: dict[str, set[str]] = collections.defaultdict(set)
        for case in cases:
            if hasattr(case, group_key):
                value = str(getattr(case, group_key))
            else:
                value = case.metadata.get(group_key, "")
            if value:
                groups[value].add(case.split.value)
        for value, splits in groups.items():
            if "calibration" in splits and "test" in splits:
                errors.append(
                    f"split leakage for {group_key}={value!r}: {sorted(splits)}"
                )

    if real_negatives < args.minimum_real_negatives:
        errors.append(
            f"only {real_negatives} recorded negative cases; "
            f"minimum is {args.minimum_real_negatives}"
        )
    for technique, labels in sorted(technique_labels.items()):
        if labels != {False, True}:
            warnings.append(
                f"technique {technique!r} lacks both positive and negative examples"
            )
    if EventKind.SILENCE.value not in kind_counts:
        warnings.append("no silence/noise cases")

    print(f"Cases: {len(cases)}")
    print("Splits: " + ", ".join(f"{key}={value}" for key, value in sorted(split_counts.items())))
    print("Kinds: " + ", ".join(f"{key}={value}" for key, value in sorted(kind_counts.items())))
    print(f"Recorded negatives: {real_negatives}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print("Corpus validation failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("Corpus validation passed.")


if __name__ == "__main__":
    main()
