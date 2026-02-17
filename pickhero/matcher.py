"""Note matching engine.

Compares detected audio notes against the tab timeline to produce
hit/close/miss feedback. No pygame dependency — pure logic.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

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
        note_filter: Callable[[NoteEvent], bool] | None = None,
        chord_partial_credit: bool = True,
    ):
        self._timeline = timeline
        self._timing_window_ms = timing_window_ms
        self._audio_offset_ms = audio_offset_ms
        self._chord_threshold_ms = chord_threshold_ms
        self.note_filter = note_filter
        self.chord_partial_credit = chord_partial_credit

        # State per note event, keyed by (timestamp_ms, string)
        self._note_states: dict[tuple[float, int], MatchType] = {}

        # Statistics
        self.hits = 0
        self.close = 0
        self.misses = 0

        # Per-measure statistics: {measure_idx: {"hits": n, "close": n, "misses": n}}
        self._measure_stats: dict[int, dict[str, int]] = defaultdict(
            lambda: {"hits": 0, "close": 0, "misses": 0}
        )

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

    def _is_filtered(self, event: NoteEvent) -> bool:
        """Return True if this note should be excluded by the difficulty filter."""
        if self.note_filter is None:
            return False
        return not self.note_filter(event)

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

    def _record_match(self, event: NoteEvent, match_type: MatchType) -> None:
        """Record a match for a note, updating stats and measure stats."""
        self._set_state(event, match_type)
        if match_type == MatchType.HIT:
            self.hits += 1
            self._measure_stats[event.measure]["hits"] += 1
        elif match_type == MatchType.CLOSE:
            self.close += 1
            self._measure_stats[event.measure]["close"] += 1
        elif match_type == MatchType.MISS:
            self.misses += 1
            self._measure_stats[event.measure]["misses"] += 1

    def has_pending_notes_at(self, playback_ms: float) -> bool:
        """Return True if there are unmatched notes at or before playback_ms."""
        window_start = playback_ms - self._timing_window_ms
        candidates = self._timeline.get_notes_in_range(window_start, playback_ms + 1)
        for note in candidates:
            if self._is_filtered(note):
                continue
            if self._get_state(note) == MatchType.PENDING:
                return True
        return False

    def _mark_missed_notes(self, playback_ms: float) -> list[MatchResult]:
        """Mark PENDING notes that have passed the timing window as MISS."""
        results = []
        cutoff = playback_ms - self._timing_window_ms
        if cutoff <= 0:
            return results

        # Check notes that should have been played by now
        candidates = self._timeline.get_notes_in_range(0, cutoff)
        for note in candidates:
            if self._is_filtered(note):
                continue
            if self._get_state(note) == MatchType.PENDING:
                self._record_match(note, MatchType.MISS)
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

            # Filter to PENDING and non-filtered only
            pending = [
                n for n in candidates
                if self._get_state(n) == MatchType.PENDING and not self._is_filtered(n)
            ]
            if not pending:
                continue

            # Find closest match by semitone distance (with octave equivalence)
            best = None
            best_dist = None
            for note in pending:
                dist = semitone_distance(detected_midi, note.midi_note)
                # Octave equivalence: if off by ~12 semitones, treat as 0
                octave_dist = dist % 12 if dist >= 12 else dist
                effective = min(dist, octave_dist)
                if best_dist is None or effective < best_dist:
                    best = note
                    best_dist = effective

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

            # Chord handling
            siblings = self._find_chord_siblings(best)
            # Filter out excluded notes from siblings
            siblings = [s for s in siblings if not self._is_filtered(s)]

            if self.chord_partial_credit and len(siblings) > 1:
                # Partial credit mode: only mark the matched note
                matched_events = []
                if self._get_state(best) == MatchType.PENDING:
                    self._record_match(best, match_type)
                    matched_events.append(best)

                # Check if majority of chord is now matched
                total_in_chord = len(siblings)
                needed = math.ceil(total_in_chord / 2)
                matched_count = sum(
                    1 for s in siblings
                    if self._get_state(s) in (MatchType.HIT, MatchType.CLOSE)
                )
                if matched_count >= needed:
                    # Auto-complete remaining pending notes
                    for s in siblings:
                        if self._get_state(s) == MatchType.PENDING:
                            self._record_match(s, match_type)
                            matched_events.append(s)
            else:
                # Easy mode (old behavior): mark all chord siblings
                matched_events = []
                for sibling in siblings:
                    if self._get_state(sibling) == MatchType.PENDING:
                        self._record_match(sibling, match_type)
                        matched_events.append(sibling)

                # Ensure the best note itself is included
                if best not in matched_events:
                    if self._get_state(best) == MatchType.PENDING:
                        self._record_match(best, match_type)
                        matched_events.append(best)

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

    def get_weakest_sections(
        self, threshold: float = 0.6, min_length: int = 2
    ) -> list[tuple[int, int, float]]:
        """Find contiguous measures below accuracy threshold.

        Returns list of (start_measure, end_measure, accuracy) sorted by
        accuracy ascending. Only returns sections of at least min_length measures.
        """
        if not self._measure_stats:
            return []

        max_measure = max(self._measure_stats.keys())
        weak_runs: list[tuple[int, int, float]] = []
        run_start = None
        run_hits = 0
        run_total = 0

        for m in range(max_measure + 1):
            stats = self._measure_stats.get(m)
            if stats is None:
                # No notes in this measure — not weak, break any run
                if run_start is not None and (m - run_start) >= min_length:
                    acc = run_hits / run_total if run_total > 0 else 0.0
                    weak_runs.append((run_start, m - 1, acc * 100))
                run_start = None
                run_hits = 0
                run_total = 0
                continue

            total = stats["hits"] + stats["close"] + stats["misses"]
            if total == 0:
                if run_start is not None and (m - run_start) >= min_length:
                    acc = run_hits / run_total if run_total > 0 else 0.0
                    weak_runs.append((run_start, m - 1, acc * 100))
                run_start = None
                run_hits = 0
                run_total = 0
                continue

            acc = stats["hits"] / total
            if acc < threshold:
                if run_start is None:
                    run_start = m
                    run_hits = 0
                    run_total = 0
                run_hits += stats["hits"]
                run_total += total
            else:
                if run_start is not None and (m - run_start) >= min_length:
                    run_acc = run_hits / run_total if run_total > 0 else 0.0
                    weak_runs.append((run_start, m - 1, run_acc * 100))
                run_start = None
                run_hits = 0
                run_total = 0

        # Close any open run
        if run_start is not None and (max_measure + 1 - run_start) >= min_length:
            acc = run_hits / run_total if run_total > 0 else 0.0
            weak_runs.append((run_start, max_measure, acc * 100))

        # Sort by accuracy ascending (weakest first)
        weak_runs.sort(key=lambda x: x[2])
        return weak_runs

    def reset(self) -> None:
        """Clear all state. Call on seek/restart."""
        self._note_states.clear()
        self.hits = 0
        self.close = 0
        self.misses = 0
        self._measure_stats.clear()
