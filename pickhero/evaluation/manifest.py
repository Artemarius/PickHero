"""Versioned corpus manifest for detector and scorer evaluation.

The runtime dataset cache only describes annotated positive events.  A useful
acceptance corpus also needs real negative performances, technique-negative
examples, held-out splits, and hardware/tone metadata.  This module keeps that
measurement schema independent from the gameplay model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Mapping


class CorpusSplit(str, Enum):
    CALIBRATION = "calibration"
    TEST = "test"
    DEVELOPMENT = "development"


class EventKind(str, Enum):
    SINGLE_NOTE = "single_note"
    CHORD = "chord"
    TECHNIQUE = "technique"
    SILENCE = "silence"


@dataclass(frozen=True)
class CorpusExpectedNote:
    midi: int
    string: int | None = None
    fret: int | None = None
    role: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CorpusExpectedNote":
        return cls(
            midi=int(str(data["midi"])),
            string=int(str(data["string"])) if data.get("string") is not None else None,
            fret=int(str(data["fret"])) if data.get("fret") is not None else None,
            role=str(data["role"]) if data.get("role") is not None else None,
        )


@dataclass(frozen=True)
class CorpusCase:
    """One independently scored corpus item.

    ``expected_present`` labels whether the requested note/chord is actually in
    the recording.  This permits genuine wrong-note and wrong-chord examples;
    they must not be inferred by merely transposing the expected label.

    For technique cases, ``technique_present`` independently labels whether the
    base pitch is played with the authored articulation.  A plain fretted note
    can therefore be a positive pitch example and a negative vibrato example.
    """

    case_id: str
    audio_path: str
    source: str
    split: CorpusSplit
    event_kind: EventKind
    start_s: float
    end_s: float
    expected_present: bool
    notes: tuple[CorpusExpectedNote, ...] = ()
    technique: str | None = None
    technique_present: bool | None = None
    expected_onset_s: float | None = None
    annotation_confidence: float = 1.0
    negative_reason: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    technique_context: dict[str, object] = field(default_factory=dict)
    window_before_ms: float = 120.0
    window_after_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if self.start_s < 0.0 or self.end_s <= self.start_s:
            raise ValueError(
                f"invalid time range for {self.case_id}: {self.start_s}..{self.end_s}"
            )
        if not 0.0 <= self.annotation_confidence <= 1.0:
            raise ValueError("annotation_confidence must be in [0, 1]")
        if self.event_kind == EventKind.SILENCE:
            if self.notes or self.technique:
                raise ValueError("silence cases cannot declare notes or techniques")
            if self.expected_present:
                raise ValueError("silence cases must use expected_present=false")
        else:
            if not self.notes:
                raise ValueError(f"{self.event_kind.value} case requires expected notes")
        if self.event_kind == EventKind.SINGLE_NOTE and len(self.notes) != 1:
            raise ValueError("single_note cases require exactly one note")
        if self.event_kind == EventKind.TECHNIQUE:
            if not self.technique:
                raise ValueError("technique cases require technique")
            if self.technique_present is None:
                raise ValueError("technique cases require technique_present")
        if self.technique_present is not None and not self.technique:
            raise ValueError("technique_present requires technique")
        if not self.expected_present and not self.negative_reason:
            raise ValueError("negative cases require negative_reason")

    @property
    def expected_midis(self) -> tuple[int, ...]:
        return tuple(note.midi for note in self.notes)

    def resolve_audio_path(self, manifest_path: Path) -> Path:
        path = Path(self.audio_path).expanduser()
        if path.is_absolute():
            return path
        return (manifest_path.parent / path).resolve()

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CorpusCase":
        version = int(str(data.get("schema_version", 1)))
        if version != 1:
            raise ValueError(f"unsupported corpus schema_version {version}")
        raw_notes = data.get("notes", ())
        if not isinstance(raw_notes, (list, tuple)):
            raise ValueError("notes must be an array")
        raw_metadata = data.get("metadata", {})
        raw_context = data.get("technique_context", {})
        if not isinstance(raw_metadata, dict) or not isinstance(raw_context, dict):
            raise ValueError("metadata and technique_context must be objects")
        return cls(
            case_id=str(data["case_id"]),
            audio_path=str(data["audio_path"]),
            source=str(data.get("source", "local")),
            split=CorpusSplit(str(data.get("split", CorpusSplit.TEST.value))),
            event_kind=EventKind(str(data["event_kind"])),
            start_s=float(str(data.get("start_s", 0.0))),
            end_s=float(str(data["end_s"])),
            expected_present=bool(data.get("expected_present", True)),
            notes=tuple(CorpusExpectedNote.from_dict(note) for note in raw_notes),
            technique=(
                str(data["technique"]) if data.get("technique") is not None else None
            ),
            technique_present=(
                bool(data["technique_present"])
                if data.get("technique_present") is not None
                else None
            ),
            expected_onset_s=(
                float(str(data["expected_onset_s"]))
                if data.get("expected_onset_s") is not None
                else None
            ),
            annotation_confidence=float(str(data.get("annotation_confidence", 1.0))),
            negative_reason=(
                str(data["negative_reason"])
                if data.get("negative_reason") is not None
                else None
            ),
            metadata={str(k): str(v) for k, v in raw_metadata.items()},
            technique_context=dict(raw_context),
            window_before_ms=float(str(data.get("window_before_ms", 120.0))),
            window_after_ms=(
                float(str(data["window_after_ms"]))
                if data.get("window_after_ms") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["schema_version"] = 1
        data["split"] = self.split.value
        data["event_kind"] = self.event_kind.value
        return data


def iter_manifest(path: str | Path) -> Iterator[CorpusCase]:
    manifest_path = Path(path)
    seen: set[str] = set()
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
                case = CorpusCase.from_dict(payload)
            except Exception as exc:
                raise ValueError(
                    f"invalid corpus case at {manifest_path}:{line_number}: {exc}"
                ) from exc
            if case.case_id in seen:
                raise ValueError(
                    f"duplicate case_id {case.case_id!r} at {manifest_path}:{line_number}"
                )
            seen.add(case.case_id)
            yield case


def load_manifest(
    path: str | Path,
    *,
    split: CorpusSplit | None = None,
) -> list[CorpusCase]:
    cases = list(iter_manifest(path))
    if split is not None:
        cases = [case for case in cases if case.split == split]
    return cases


def write_manifest(
    path: str | Path,
    cases: Iterable[CorpusCase],
    *,
    append: bool = False,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with destination.open(mode, encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case.to_dict(), sort_keys=True) + "\n")
