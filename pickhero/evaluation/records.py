"""Serializable per-case observations emitted by the corpus evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


@dataclass
class EvaluationRecord:
    case_id: str
    source: str
    split: str
    event_kind: str
    mode: str
    audio_path: str
    start_s: float
    end_s: float
    expected_present: bool
    predicted_present: bool
    score: float
    expected_midis: tuple[int, ...]
    annotation_confidence: float
    metadata: dict[str, str] = field(default_factory=dict)
    onset_expected: bool = False
    onset_detected: bool = False
    onset_error_ms: float | None = None
    cents_error: float | None = None
    alias_risk: float | None = None
    chord_verdict: str | None = None
    chord_score: float | None = None
    missing_roles: tuple[str, ...] = ()
    extra_pitch_classes: tuple[int, ...] = ()
    technique: str | None = None
    technique_expected: bool | None = None
    technique_detected: bool | None = None
    technique_uncertain: bool | None = None
    technique_quality: float | None = None
    peak_dbfs: float = -120.0
    rms_dbfs: float = -120.0
    dc_offset: float = 0.0
    clipped_fraction: float = 0.0
    failure_reasons: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "EvaluationRecord":
        payload = dict(data)
        for key in (
            "expected_midis",
            "missing_roles",
            "extra_pitch_classes",
            "failure_reasons",
        ):
            if key in payload and isinstance(payload[key], list):
                payload[key] = tuple(payload[key])
        return cls(**payload)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    return str(value)
