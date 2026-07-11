"""Evidence types for expected-event verification.

This module defines the data model for the verification architecture:
instead of "what note did I hear?", the pipeline answers "did the expected
note/technique happen in this audio window?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class ExpectedNote:
    """A single expected note in a chord, preserving string/fret identity."""
    midi: int
    string: int | None = None
    fret: int | None = None
    event_id: str | None = None  # "timestamp_ms:string" for correlation back to NoteEvent


class EvidenceType(Enum):
    """Category of musical event being verified."""

    SINGLE_NOTE = "single_note"
    CHORD = "chord"
    BEND = "bend"
    SLIDE = "slide"
    VIBRATO = "vibrato"
    HARMONIC = "harmonic"
    DEAD_NOTE = "dead_note"
    PALM_MUTE = "palm_mute"
    NOISE_GESTURE = "noise_gesture"


@dataclass
class PitchEvidence:
    """A single pitch observation supporting (or contradicting) a note."""

    midi_note: int
    cents_error: float | None
    confidence: float
    source: str  # e.g. "YIN", "CREPE", "spectral", "chroma"


@dataclass
class NoteVerification:
    """Verification result for a single expected note."""

    is_pitch_present: bool       # pitch present in window (renamed from is_present)
    is_onset_present: bool       # attack evidence in window (new)
    pitch_evidence: PitchEvidence | None
    onset_ms: float | None
    harmonic_score: float
    timing_error_ms: float | None
    alias_risk: float = 0.0      # 0.0-1.0, best competitor / target score (new)

    @property
    def confidence(self) -> float:
        """Aggregate confidence for the verification."""
        if self.pitch_evidence is not None:
            return self.pitch_evidence.confidence
        return self.harmonic_score

@dataclass
class ChordVerification:
    """Verification result for an expected chord.

    ``observed_pitch_classes`` and ``pitch_class_energy`` expose the complete
    chroma observation, not only yes/no answers for authored notes. This lets
    the scorer distinguish a harmless missing duplicate from a defining chord
    tone or a genuinely foreign note.
    """

    notes: list[NoteVerification]
    partial: bool
    total_harmonic_energy: float
    observed_pitch_classes: frozenset[int] = frozenset()
    pitch_class_energy: dict[int, float] = field(default_factory=dict)
    quality_score: float = 0.0
    missing_roles: tuple[str, ...] = ()
    extra_pitch_classes: tuple[int, ...] = ()


@dataclass
class TechniqueVerification:
    """Verification result for an expected technique.

    Technique evidence is deliberately tri-state. ``uncertain`` means the
    audio does not justify either awarding or rejecting the articulation; it
    must never invalidate an otherwise correct base note.
    """

    technique: str
    is_present: bool
    confidence: float
    details: dict = field(default_factory=dict)
    uncertain: bool = False
    quality: float | None = None


@dataclass
class VerificationResult:
    """Complete verification result for an expected event at a point in time."""

    expected_midi: set[int]
    expected_techniques: list[str]
    verified: NoteVerification | ChordVerification
    techniques: list[TechniqueVerification]
    timestamp_ms: float
