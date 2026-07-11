"""Runtime state for the single gameplay scoring authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pickhero.audio.evidence import TechniqueVerification


class EventState(Enum):
    PENDING = "pending"
    ATTACKING = "attacking"
    PITCHED = "pitched"
    SUSTAINING = "sustaining"
    RELEASED = "released"
    HIT = "hit"
    PARTIAL = "partial"
    MISS = "miss"


@dataclass
class EventRuntime:
    """Mutable evidence accumulated for one immutable chart event."""

    state: EventState = EventState.PENDING
    onset_ms: float | None = None
    first_pitch_ms: float | None = None
    confidence_peak: float = 0.0
    sustain_hits: int = 0
    sustain_checks: int = 0
    checked_sustain_points: set[float] = field(default_factory=set)
    technique_evidence: list[TechniqueVerification] = field(default_factory=list)
    attack_quality: float = 0.0
    release_quality: float | None = None
    transition_quality: float | None = None
    technique_finalized: bool = False
    sustain_feedback_emitted: bool = False
    terminal_emitted: bool = False
    last_evaluated_ms: float = -1.0

    @property
    def sustain_ratio(self) -> float:
        if self.sustain_checks <= 0:
            return 1.0
        return self.sustain_hits / self.sustain_checks


@dataclass(frozen=True)
class ChordRoleVerdict:
    root_detected: bool = False
    third_detected: bool | None = None
    seventh_detected: bool | None = None
    fifth_detected: bool = False
    extra_pitch_classes: int = 0

    @property
    def is_hit(self) -> bool:
        return (
            self.root_detected
            and self.third_detected is not False
            and self.seventh_detected is not False
            and self.extra_pitch_classes == 0
        )

    @property
    def is_partial(self) -> bool:
        return self.root_detected and bool(self.third_detected or self.seventh_detected)

    @property
    def is_close(self) -> bool:
        return self.root_detected and not bool(
            self.third_detected or self.seventh_detected or self.fifth_detected
        ) and self.extra_pitch_classes == 0
