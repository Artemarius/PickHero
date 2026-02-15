"""Note matching engine.

Compares detected audio notes against the tab timeline to produce
hit/close/miss feedback. No pygame dependency — pure logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pickhero.audio.note_utils import semitone_distance
from pickhero.audio.input import TimestampedNote
from pickhero.tabs.timeline import NoteEvent, Timeline


class MatchType(Enum):
    PENDING = "pending"
    HIT = "hit"
    CLOSE = "close"
    MISS = "miss"


@dataclass
class MatchResult:
    """Result of matching a detected note against the timeline."""
    match_type: MatchType
    matched_events: list[NoteEvent] = field(default_factory=list)
    semitone_distance: int | None = None


class NoteMatcher:
    """Matches detected audio notes against tab timeline events.

    Each NoteEvent in the timeline is tracked by state (PENDING -> HIT/CLOSE/MISS).
    Detected notes are compared to PENDING events within the timing window.
    """

    def __init__(
        self,
        timeline: Timeline,
        timing_window_ms: float = 100.0,
        audio_offset_ms: float = 0.0,
        chord_threshold_ms: float = 50.0,
    ):
        self._timeline = timeline
        self._timing_window_ms = timing_window_ms
        self._audio_offset_ms = audio_offset_ms
        self._chord_threshold_ms = chord_threshold_ms

        # State per note event, keyed by (timestamp_ms, string)
        self._note_states: dict[tuple[float, int], MatchType] = {}

        # Statistics
        self.hits = 0
        self.close = 0
        self.misses = 0

    @property
    def audio_offset_ms(self) -> float:
        return self._audio_offset_ms

    @audio_offset_ms.setter
    def audio_offset_ms(self, value: float) -> None:
        self._audio_offset_ms = value

    def _note_key(self, event: NoteEvent) -> tuple[float, int]:
        return (event.timestamp_ms, event.string)

    def _get_state(self, event: NoteEvent) -> MatchType:
        return self._note_states.get(self._note_key(event), MatchType.PENDING)

    def _set_state(self, event: NoteEvent, state: MatchType) -> None:
        self._note_states[self._note_key(event)] = state

    def get_note_state(self, event: NoteEvent) -> MatchType:
        """Get the current match state of a timeline note."""
        return self._get_state(event)

    def _find_chord_siblings(self, event: NoteEvent) -> list[NoteEvent]:
        """Find notes within chord_threshold_ms of the given event."""
        return [
            n for n in self._timeline.get_active_notes_at_time(
                event.timestamp_ms, self._chord_threshold_ms
            )
            if abs(n.timestamp_ms - event.timestamp_ms) <= self._chord_threshold_ms
        ]

    def _mark_missed_notes(self, playback_ms: float) -> list[MatchResult]:
        """Mark PENDING notes that have passed the timing window as MISS."""
        results = []
        cutoff = playback_ms - self._timing_window_ms
        if cutoff <= 0:
            return results

        # Check notes that should have been played by now
        candidates = self._timeline.get_notes_in_range(0, cutoff)
        for note in candidates:
            if self._get_state(note) == MatchType.PENDING:
                self._set_state(note, MatchType.MISS)
                self.misses += 1
                results.append(MatchResult(
                    match_type=MatchType.MISS,
                    matched_events=[note],
                    semitone_distance=None,
                ))
        return results

    def process_detected_notes(
        self, detected: list[TimestampedNote], playback_ms: float
    ) -> list[MatchResult]:
        """Process detected notes against the timeline.

        Args:
            detected: Notes from AudioCapture.get_notes()
            playback_ms: Current playback position in the song

        Returns:
            List of match results for this frame.
        """
        results = []

        # First, mark any notes that have passed the window as missed
        results.extend(self._mark_missed_notes(playback_ms))

        # Process each detected note with an onset
        for ts_note in detected:
            if not ts_note.note.is_onset:
                continue

            adjusted_ms = ts_note.timestamp_ms + self._audio_offset_ms
            detected_midi = ts_note.note.midi_note

            # Find tab notes active near this time
            candidates = self._timeline.get_active_notes_at_time(
                adjusted_ms, self._timing_window_ms
            )

            # Filter to PENDING only
            pending = [n for n in candidates if self._get_state(n) == MatchType.PENDING]
            if not pending:
                continue

            # Find closest match by semitone distance
            best = None
            best_dist = None
            for note in pending:
                dist = semitone_distance(detected_midi, note.midi_note)
                if best_dist is None or dist < best_dist:
                    best = note
                    best_dist = dist

            if best is None or best_dist is None:
                continue

            # Classify match
            if best_dist == 0:
                match_type = MatchType.HIT
            elif best_dist == 1:
                match_type = MatchType.CLOSE
            else:
                # Too far off — ignore this detection, no penalty
                continue

            # Mark the matched note and its chord siblings
            siblings = self._find_chord_siblings(best)
            matched_events = []
            for sibling in siblings:
                if self._get_state(sibling) == MatchType.PENDING:
                    self._set_state(sibling, match_type)
                    matched_events.append(sibling)
                    if match_type == MatchType.HIT:
                        self.hits += 1
                    else:
                        self.close += 1

            # Ensure the best note itself is included
            if best not in matched_events:
                if self._get_state(best) == MatchType.PENDING:
                    self._set_state(best, match_type)
                    matched_events.append(best)
                    if match_type == MatchType.HIT:
                        self.hits += 1
                    else:
                        self.close += 1

            results.append(MatchResult(
                match_type=match_type,
                matched_events=matched_events,
                semitone_distance=best_dist,
            ))

        return results

    def get_statistics(self) -> dict:
        """Return current match statistics."""
        total = self.hits + self.close + self.misses
        accuracy = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "close": self.close,
            "misses": self.misses,
            "total": total,
            "accuracy_percent": accuracy,
        }

    def reset(self) -> None:
        """Clear all state. Call on seek/restart."""
        self._note_states.clear()
        self.hits = 0
        self.close = 0
        self.misses = 0
