"""Event state machine data types for unified scoring.

Replaces the three-path scoring (verify_hit_zone, process_detected_notes,
verify_chord_at) with one state machine that accepts evidence from all
sources and produces a single verdict per (timestamp, string) event.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field


class EventState(Enum):
    """States for a single note/chord event through the scoring lifecycle."""
    PENDING = "pending"
    ATTACKING = "attacking"
    PITCHED = "pitched"
    SUSTAINING = "sustaining"
    RELEASED = "released"
    HIT = "hit"
    PARTIAL = "partial"
    MISS = "miss"


@dataclass
class PitchVerdict:
    """Result of pitch detection for an event."""
    correct: bool = False
    midi: int | None = None
    confidence: float = 0.0
    cents_error: float | None = None


@dataclass
class TimingVerdict:
    """Result of onset timing for an event."""
    early: bool = False
    late: bool = False
    exact: bool = False
    error_ms: float = 0.0


@dataclass
class TechniqueVerdict:
    """Result of technique verification for an event.

    Technique never affects the base note verdict. It only modifies
    the accuracy score for skill tracking and recommendations.
    """
    technique: str = ""
    present: bool = False
    uncertain: bool = False
    confidence: float = 0.0


@dataclass
class ChordRoleVerdict:
    """Chord detection quality from observed pitch classes.

    Critical roles: root, third (if present), seventh (if present).
    Non-critical: fifth, duplicated notes.
    Extra pitch classes: notes detected that are not in the chord.
    """
    root_detected: bool = False
    third_detected: bool | None = None
    seventh_detected: bool | None = None
    fifth_detected: bool = False
    extra_pitch_classes: int = 0

    @property
    def is_hit(self) -> bool:
        """All critical roles present, no extra notes."""
        if not self.root_detected:
            return False
        if self.third_detected is False:  # explicitly detected as absent
            return False
        if self.seventh_detected is False:
            return False
        if self.extra_pitch_classes > 0:
            return False
        return True

    @property
    def is_partial(self) -> bool:
        """Root + at least one other critical role (third or seventh),
        possibly extra notes."""
        if not self.root_detected:
            return False
        return bool(self.third_detected or self.seventh_detected)

    @property
    def is_close(self) -> bool:
        """Root only, no other detected roles and no extra notes."""
        if not self.root_detected:
            return False
        if self.third_detected or self.seventh_detected or self.fifth_detected:
            return False
        return True
