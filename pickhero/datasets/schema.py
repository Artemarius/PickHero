"""Unified event schema for external guitar datasets.

Datasets are not redistributed by PickHero.  Users download them separately
and point the registry at the directories; the importers normalize the
annotations into :class:`ClipEvent` records.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClipExpectedNote:
    """A single expected note in a chord, preserving string/fret identity."""

    midi: int
    string: int | None = None
    fret: int | None = None
    role: str | None = None  # "root", "third", "fifth", etc.


@dataclass(frozen=True)
class ClipEvent:
    """A single annotated event from a real-world guitar recording.

    Attributes:
        clip_id: Stable identifier, e.g. "goat/foo/bar".
        source: Dataset name, e.g. "GOAT", "GuitarSet", "IDMT".
        start_s: Event start in seconds.
        end_s: Event end in seconds.
        midi: Single-note MIDI (None for chords).
        technique: Technique label.
        confidence: Annotation confidence (1.0 = perfect).
        audio_path: Path to the audio file (WAV/FLAC).
        notes: Per-note records for chords (empty for single notes).
        string: Guitar string number, 1-indexed from high E.
        fret: Fret number.
    """

    clip_id: str
    source: str
    start_s: float
    end_s: float
    midi: int | None
    technique: str
    confidence: float
    audio_path: str
    notes: tuple[ClipExpectedNote, ...] = ()
    string: int | None = None
    fret: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def midi_notes(self) -> frozenset[int] | None:
        """Computed view for callers that only need the pitch set."""
        if not self.notes:
            return None
        return frozenset(n.midi for n in self.notes)

    def __post_init__(self) -> None:
        if self.midi is not None and self.notes:
            raise ValueError("ClipEvent cannot have both midi and notes")
        if self.midi is None and not self.notes:
            raise ValueError("ClipEvent must have midi or notes")
