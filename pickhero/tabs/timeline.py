"""Song timeline data structures.

NoteEvent represents a single note in a tab. Timeline holds a sorted sequence
of NoteEvents and provides efficient range queries for the game loop.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field


@dataclass(frozen=True)
class NoteEvent:
    """A single note event in the song timeline."""

    timestamp_ms: float
    duration_ms: float
    midi_note: int
    string: int  # 1-N (1 = highest pitched string in the tab)
    fret: int    # 0=open
    measure: int = 0  # measure index (0-based)
    expected_articulation: str | None = None  # "hammer_on", "pull_off", "bend", "vibrato", "slide", "palm_mute", "harmonic"

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
        self._notes = sorted(notes, key=lambda n: (n.timestamp_ms, n.string))
        self._timestamps = [n.timestamp_ms for n in self._notes]
        self.metadata = metadata or SongMetadata()
        self._measures = measures or []
        self._cursor = 0
        # Active-window cursor for get_active_notes_at_time optimization
        self._active_cursor = 0
        self._last_query_time = -1.0

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
