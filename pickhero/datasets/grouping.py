"""Helpers for preserving simultaneous annotated notes as chord events."""

from __future__ import annotations

from pickhero.datasets.schema import ClipEvent, ClipExpectedNote


def group_simultaneous_notes(
    events: list[ClipEvent],
    *,
    tolerance_s: float = 0.012,
) -> list[ClipEvent]:
    """Collapse same-recording note onsets into chord annotations.

    The former importers emitted one positive single-note case per string even
    when the recording contained a full chord. That measures whether a pitch is
    somewhere in a polyphonic mixture, not whether single-note recognition or
    chord judgment works. Existing chord events are retained unchanged.
    """
    if not events:
        return []
    singles = sorted(
        (event for event in events if event.midi is not None),
        key=lambda event: (event.audio_path, event.start_s, event.midi or 0),
    )
    grouped: list[ClipEvent] = [event for event in events if event.midi is None]
    index = 0
    while index < len(singles):
        first = singles[index]
        bucket = [first]
        cursor = index + 1
        while cursor < len(singles):
            candidate = singles[cursor]
            if candidate.audio_path != first.audio_path:
                break
            if candidate.source != first.source or candidate.technique != first.technique:
                break
            if abs(candidate.start_s - first.start_s) > tolerance_s:
                break
            bucket.append(candidate)
            cursor += 1

        if len(bucket) == 1:
            grouped.append(first)
        else:
            notes = tuple(
                ClipExpectedNote(
                    midi=int(event.midi),
                    string=event.string,
                    fret=event.fret,
                )
                for event in sorted(
                    bucket,
                    key=lambda event: (
                        event.string if event.string is not None else 99,
                        event.midi or 0,
                    ),
                )
            )
            grouped.append(
                ClipEvent(
                    clip_id=f"{first.clip_id}:chord",
                    source=first.source,
                    start_s=min(event.start_s for event in bucket),
                    end_s=max(event.end_s for event in bucket),
                    midi=None,
                    notes=notes,
                    technique=first.technique,
                    confidence=min(event.confidence for event in bucket),
                    audio_path=first.audio_path,
                    metadata=dict(first.metadata),
                )
            )
        index = cursor
    return sorted(grouped, key=lambda event: (event.audio_path, event.start_s, event.clip_id))
