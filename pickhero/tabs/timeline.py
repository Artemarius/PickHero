"""Song timeline data structures.

NoteEvent represents a single note in a tab. Timeline holds a sorted sequence
of NoteEvents and provides efficient range queries for the game loop.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pickhero.audio.performance import TechniqueSpec


@dataclass(frozen=True)
class NoteEvent:
    """A single note event in the song timeline."""

    timestamp_ms: float
    duration_ms: float
    midi_note: int
    string: int  # 1-N (1 = highest pitched string in the tab)
    fret: int    # 0=open
    measure: int = 0  # measure index (0-based)
    techniques: tuple["TechniqueSpec", ...] = ()  # expected techniques from the tab
    # Derived enrichment fields — populated by Timeline._enrich_arrangement.
    # Marked compare=False so original and enriched events compare equal by
    # authored/core fields only.
    phrase_id: int = field(default=-1, compare=False)
    """Stable phrase identifier. Auto-derived from four-measure blocks when
    the source format does not provide authored phrases."""
    difficulty_level: int = field(default=0, compare=False)
    """Arrangement layer 1-5. Zero means derive a coherent fallback layer."""
    chord_id: str | None = field(default=None, compare=False)
    """Identity shared by simultaneous notes. Mono scoring uses this as one
    musical event rather than pretending every physical string is observable."""
    pick_required: bool = field(default=True, compare=False)
    """False for tied hammer-ons, pull-offs and legato slide destinations."""
    sustain_checkpoints: tuple[float, ...] = field(default=(), compare=False)
    """Absolute song times at which sustain quality should be sampled."""

    def __post_init__(self):
        if self.timestamp_ms < 0:
            raise ValueError(f"timestamp_ms must be >= 0, got {self.timestamp_ms}")
        if self.duration_ms < 0:
            raise ValueError(f"duration_ms must be >= 0, got {self.duration_ms}")
        if not 0 <= self.midi_note <= 127:
            raise ValueError(f"midi_note must be 0-127, got {self.midi_note}")
        if not 1 <= self.string <= 12:
            raise ValueError(f"string must be 1-12, got {self.string}")
        if self.fret < 0:
            raise ValueError(f"fret must be >= 0, got {self.fret}")

    @property
    def end_ms(self) -> float:
        return self.timestamp_ms + self.duration_ms


@dataclass(frozen=True)
class MeasureInfo:
    """Time range for a single measure/bar."""
    index: int
    start_ms: float
    end_ms: float

@dataclass
class SongMetadata:
    """Metadata extracted from a GP file."""

    title: str = ""
    artist: str = ""
    album: str = ""
    track_name: str = ""
    tempo: int = 120
    tuning: dict[int, int] = field(default_factory=dict)
    num_strings: int = 6
    track_index: int = 0
class Timeline:
    """Sorted collection of NoteEvents with efficient range queries."""

    def __init__(self, notes: list[NoteEvent], metadata: SongMetadata | None = None,
                 measures: list[MeasureInfo] | None = None):
        self._notes = self._enrich_arrangement(notes)
        self._notes.sort(key=lambda n: (n.timestamp_ms, n.string))
        self._timestamps = [n.timestamp_ms for n in self._notes]
        self.metadata = metadata or SongMetadata()
        self._measures = measures or []
        self._cursor = 0
        # Active-window cursor for get_active_notes_at_time optimization
        self._active_cursor = 0
        self._last_query_time = -1.0


    @staticmethod
    def _enrich_arrangement(notes: list[NoteEvent]) -> list[NoteEvent]:
        """Fill phrase, chord, difficulty and sustain metadata deterministically.

        Guitar Pro files do not always carry Rocksmith-style phrase and dynamic
        difficulty metadata. The fallback keeps layers musically coherent:
        bass/root anchors appear first, rhythmic guide notes next, complete
        voicings after that, and expressive articulations in upper layers.
        """
        if not notes:
            return []
        groups: dict[float, list[NoteEvent]] = {}
        for note in notes:
            groups.setdefault(round(note.timestamp_ms, 3), []).append(note)

        phrase_onset_index: dict[int, int] = {}
        enriched: list[NoteEvent] = []
        for onset, group in sorted(groups.items()):
            phrase_id = min(n.measure for n in group) // 4
            onset_index = phrase_onset_index.get(phrase_id, 0)
            phrase_onset_index[phrase_id] = onset_index + 1
            chord_id = f"{onset:.3f}" if len(group) > 1 else None
            ordered = sorted(group, key=lambda n: (n.midi_note, n.string))
            seen_pc: set[int] = set()
            for position, note in enumerate(ordered):
                level = note.difficulty_level
                if level <= 0:
                    if len(group) > 1:
                        pc = note.midi_note % 12
                        if position == 0:
                            level = 1  # bass/root guide
                        elif pc in seen_pc:
                            level = 4  # doubled voicing tone
                        elif position == 1:
                            level = 2
                        else:
                            level = 3
                        seen_pc.add(pc)
                    else:
                        level = 1 if onset_index % 4 == 0 else (2 if onset_index % 2 == 0 else 3)
                    if note.techniques:
                        level = max(level, 4)
                        if any(t.kind in ("bend", "vibrato", "harmonic") for t in note.techniques):
                            level = 5

                tied = any(
                    t.kind in ("hammer_on", "pull_off", "slide")
                    and getattr(t, "tied_to_previous", False)
                    for t in note.techniques
                )
                checkpoints = note.sustain_checkpoints
                if not checkpoints and note.duration_ms >= 300.0:
                    checkpoints = tuple(
                        note.timestamp_ms + note.duration_ms * fraction
                        for fraction in (0.25, 0.5, 0.75)
                    )
                enriched.append(replace(
                    note,
                    phrase_id=note.phrase_id if note.phrase_id >= 0 else phrase_id,
                    difficulty_level=max(1, min(5, level)),
                    chord_id=note.chord_id or chord_id,
                    pick_required=note.pick_required and not tied,
                    sustain_checkpoints=checkpoints,
                ))
        return enriched

    def __len__(self) -> int:
        return len(self._notes)

    def __repr__(self) -> str:
        title = self.metadata.title or "Untitled"
        return f"Timeline('{title}', {len(self)} notes, {self.duration_ms:.0f}ms)"

    @property
    def notes(self) -> list[NoteEvent]:
        return list(self._notes)

    @property
    def measures(self) -> list[MeasureInfo]:
        return list(self._measures)

    @property
    def duration_ms(self) -> float:
        if not self._notes:
            return 0.0
        return max(n.end_ms for n in self._notes)

    def get_notes_in_range(self, start_ms: float, end_ms: float) -> list[NoteEvent]:
        """Return notes whose timestamp_ms falls within [start_ms, end_ms)."""
        left = bisect.bisect_left(self._timestamps, start_ms)
        right = bisect.bisect_left(self._timestamps, end_ms)
        return self._notes[left:right]

    def get_notes_before(self, end_ms: float, from_index: int = 0) -> tuple[list[NoteEvent], int]:
        """Return (notes with timestamp_ms < end_ms starting at from_index, new_index).

        Callers tracking a monotonic scan cursor pass their last index as
        ``from_index`` and receive the new index just past the returned notes,
        so each call scans only newly-passed notes — O(new) per call, not O(total).
        """
        if from_index >= len(self._notes):
            return [], from_index
        right = bisect.bisect_left(self._timestamps, end_ms, lo=from_index)
        if right <= from_index:
            return [], from_index
        return self._notes[from_index:right], right

    def get_active_notes_at_time(self, time_ms: float, window_ms: float = 100.0) -> list[NoteEvent]:
        """Return notes that overlap the window [time_ms - window_ms, time_ms + window_ms].

        A note is active if its sounding range [timestamp_ms, end_ms] overlaps the window.
        Uses a monotonic cursor for amortized O(1) per call during forward playback.
        Falls back to scanning from 0 on backward queries (seek).
        """
        window_start = time_ms - window_ms
        window_end = time_ms + window_ms

        # Reset cursor on backward seek
        if time_ms < self._last_query_time:
            self._active_cursor = 0
        self._last_query_time = time_ms

        # Find candidates: notes that start before window_end
        right = bisect.bisect_right(self._timestamps, window_end)

        # Advance cursor past notes that have fully ended before the window
        while self._active_cursor < right:
            note = self._notes[self._active_cursor]
            if note.end_ms < window_start:
                self._active_cursor += 1
            else:
                break

        result = []
        for i in range(self._active_cursor, right):
            note = self._notes[i]
            if note.end_ms >= window_start and note.timestamp_ms <= window_end:
                result.append(note)
        return result

    def seek(self, time_ms: float) -> None:
        """Move the cursor to the first note at or after time_ms."""
        self._cursor = bisect.bisect_left(self._timestamps, time_ms)
        self._active_cursor = self._cursor
        self._last_query_time = time_ms

    def get_next_notes(self, count: int = 1) -> list[NoteEvent]:
        """Return up to `count` notes from the cursor, advancing it."""
        end = min(self._cursor + count, len(self._notes))
        result = self._notes[self._cursor:end]
        self._cursor = end
        return result
