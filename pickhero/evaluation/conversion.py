"""Conversion from downloaded dataset annotations to evaluation cases."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Iterable

from pickhero.datasets.schema import ClipEvent
from pickhero.evaluation.manifest import (
    CorpusCase,
    CorpusExpectedNote,
    CorpusSplit,
    EventKind,
)


_NORMAL_TECHNIQUES = {"", "none", "normal", "sustain", "unknown"}


def cases_from_clip_events(
    events: Iterable[ClipEvent],
    *,
    calibration_fraction: float = 0.30,
    add_counterfactual_negatives: bool = False,
    split_group: str = "audio_path",
) -> list[CorpusCase]:
    """Convert normalized dataset events without leaking clips across splits.

    Every event from the same source audio file is assigned to the same split.
    Counterfactual negatives are useful for smoke-testing alias rejection but
    are tagged explicitly and must not replace recordings of actual mistakes.
    """
    if not 0.0 <= calibration_fraction <= 1.0:
        raise ValueError("calibration_fraction must be in [0, 1]")
    cases: list[CorpusCase] = []
    for event in events:
        group_value = (
            event.audio_path
            if split_group == "audio_path"
            else event.metadata.get(split_group) or event.audio_path
        )
        split = _split_for_group(
            f"{event.source}\0{split_group}={group_value}", calibration_fraction
        )
        technique = event.technique.strip().lower()
        has_technique = technique not in _NORMAL_TECHNIQUES
        notes = (
            tuple(
                CorpusExpectedNote(
                    midi=note.midi,
                    string=note.string,
                    fret=note.fret,
                    role=note.role,
                )
                for note in event.notes
            )
            if event.notes
            else (
                CorpusExpectedNote(
                    midi=int(event.midi),
                    string=event.string,
                    fret=event.fret,
                ),
            )
        )
        if len(notes) > 1:
            event_kind = EventKind.CHORD
        elif has_technique:
            event_kind = EventKind.TECHNIQUE
        else:
            event_kind = EventKind.SINGLE_NOTE
        case = CorpusCase(
            case_id=event.clip_id,
            audio_path=event.audio_path,
            source=event.source,
            split=split,
            event_kind=event_kind,
            start_s=event.start_s,
            end_s=max(event.end_s, event.start_s + 0.05),
            expected_present=True,
            notes=notes,
            technique=technique if has_technique else None,
            technique_present=True if has_technique else None,
            expected_onset_s=event.start_s,
            annotation_confidence=event.confidence,
            metadata={**event.metadata, "provenance": "dataset_annotation"},
        )
        cases.append(case)
        if add_counterfactual_negatives:
            cases.extend(_counterfactuals(case))
    return cases


def _split_for_group(group: str, calibration_fraction: float) -> CorpusSplit:
    digest = hashlib.sha256(group.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return (
        CorpusSplit.CALIBRATION
        if bucket < calibration_fraction
        else CorpusSplit.TEST
    )


def _counterfactuals(case: CorpusCase) -> list[CorpusCase]:
    if case.event_kind == EventKind.SILENCE:
        return []
    offsets = (1, 2, 12)
    generated: list[CorpusCase] = []
    for offset in offsets:
        shifted = tuple(replace(note, midi=note.midi + offset) for note in case.notes)
        generated.append(
            replace(
                case,
                case_id=f"{case.case_id}:counterfactual:+{offset}",
                event_kind=(
                    EventKind.CHORD if len(shifted) > 1 else EventKind.SINGLE_NOTE
                ),
                expected_present=False,
                notes=shifted,
                technique=None,
                technique_present=None,
                negative_reason="counterfactual_expected_label",
                metadata={
                    **case.metadata,
                    "provenance": "counterfactual",
                    "counterfactual_offset": str(offset),
                },
            )
        )
    return generated
