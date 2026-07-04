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
from pickhero.timing import (
    PitchVerdict,
    TimingObservation,
    TimingVerdict,
    classify_pitch_distance,
    classify_timing_error,
    compute_stats,
)


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
        timing_judge_enabled: bool = False,
        pitch_strict: bool = False,
    ):
        self._timeline = timeline
        self._timing_window_ms = timing_window_ms
        self._audio_offset_ms = audio_offset_ms
        self._chord_threshold_ms = chord_threshold_ms
        self.note_filter = note_filter
        self.chord_partial_credit = chord_partial_credit
        self._timing_judge_enabled = timing_judge_enabled
        self._pitch_strict = pitch_strict

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

        # Timing Judge observations
        self._timing_observations: list[TimingObservation] = []

    @property
    def audio_offset_ms(self) -> float:
        return self._audio_offset_ms

    @audio_offset_ms.setter
    def audio_offset_ms(self, value: float) -> None:
        self._audio_offset_ms = value

    @property
    def timing_judge_enabled(self) -> bool:
        return self._timing_judge_enabled

    @timing_judge_enabled.setter
    def timing_judge_enabled(self, value: bool) -> None:
        self._timing_judge_enabled = value

    @property
    def pitch_strict(self) -> bool:
        return self._pitch_strict

    @pitch_strict.setter
    def pitch_strict(self, value: bool) -> None:
        self._pitch_strict = value

    def get_timing_observations(self) -> list[TimingObservation]:
        """Return all timing observations recorded so far."""
        return list(self._timing_observations)

    def get_timing_stats(self) -> "TimingStats":
        """Compute and return timing statistics from observations."""
        from pickhero.timing import TimingStats
        return compute_stats(self._timing_observations)

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
                # Record timing observation for missed note
                if self._timing_judge_enabled:
                    self._timing_observations.append(TimingObservation(
                        detected_ms=float("nan"),
                        expected_ms=note.timestamp_ms,
                        timing_error_ms=float("nan"),
                        verdict=TimingVerdict.MISSED,
                        midi_note=0,
                        expected_midi=note.midi_note,
                        measure=note.measure,
                        confidence=0.0,
                        pitch_verdict=PitchVerdict.UNKNOWN,
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
                # Timing Judge: record extra (stray) onset with no matching note
                if self._timing_judge_enabled:
                    # Find nearest note for reference (even if already matched)
                    nearest = None
                    nearest_dist = None
                    for n in candidates:
                        d = abs(n.timestamp_ms - adjusted_ms)
                        if nearest_dist is None or d < nearest_dist:
                            nearest = n
                            nearest_dist = d
                    expected_ms = nearest.timestamp_ms if nearest else adjusted_ms
                    measure = nearest.measure if nearest else -1
                    expected_midi = nearest.midi_note if nearest else 0
                    self._timing_observations.append(TimingObservation(
                        detected_ms=adjusted_ms,
                        expected_ms=expected_ms,
                        timing_error_ms=adjusted_ms - expected_ms,
                        verdict=TimingVerdict.EXTRA,
                        midi_note=detected_midi,
                        expected_midi=expected_midi,
                        measure=measure,
                        confidence=ts_note.note.confidence,
                        pitch_verdict=PitchVerdict.UNKNOWN,
                        articulation=ts_note.note.articulation,
                    ))
                continue

            # Find closest match by semitone distance
            best = None
            best_dist = None
            for note in pending:
                dist = semitone_distance(detected_midi, note.midi_note)
                if self._pitch_strict:
                    # Strict mode: no octave equivalence
                    effective = dist
                else:
                    # Arcade mode: octave equivalence
                    octave_dist = dist % 12 if dist >= 12 else dist
                    effective = min(dist, octave_dist)
                if best_dist is None or effective < best_dist:
                    best = note
                    best_dist = effective

            if best is None or best_dist is None:
                continue

            # Tab-guided octave correction: in strict mode, an exact octave
            # (12 semitones) means the fundamental was detected as a harmonic — snap to correct.
            if self._pitch_strict and best_dist == 12:
                best_dist = 0
            # Classify match
            if best_dist == 0:
                match_type = MatchType.HIT
            elif best_dist == 1 and not self._pitch_strict:
                match_type = MatchType.CLOSE
            else:
                # Too far off, or strict mode with !=0 — ignore but record timing
                if self._timing_judge_enabled:
                    pitch_v = classify_pitch_distance(best_dist)
                    self._timing_observations.append(TimingObservation(
                        detected_ms=adjusted_ms,
                        expected_ms=best.timestamp_ms,
                        timing_error_ms=adjusted_ms - best.timestamp_ms,
                        verdict=TimingVerdict.EXTRA,
                        midi_note=detected_midi,
                        expected_midi=best.midi_note,
                        measure=best.measure,
                        confidence=ts_note.note.confidence,
                        pitch_verdict=pitch_v,
                        articulation=ts_note.note.articulation,
                    ))
                continue

            # Record timing observation for the matched note
            if self._timing_judge_enabled:
                timing_error_ms = adjusted_ms - best.timestamp_ms
                verdict = classify_timing_error(timing_error_ms)
                pitch_v = classify_pitch_distance(best_dist)
                self._timing_observations.append(TimingObservation(
                    detected_ms=adjusted_ms,
                    expected_ms=best.timestamp_ms,
                    timing_error_ms=timing_error_ms,
                    verdict=verdict,
                    midi_note=detected_midi,
                    expected_midi=best.midi_note,
                    measure=best.measure,
                    confidence=ts_note.note.confidence,
                    pitch_verdict=pitch_v,
                    articulation=ts_note.note.articulation,
                ))

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

    def verify_chord_at(
        self,
        playback_ms: float,
        chord_detector=None,
    ) -> list[MatchResult]:
        """Verify chords at the hit zone using FFT spectral analysis.

        Called when there are multiple pending notes at the same timestamp.
        Uses the ChordDetector to check if expected frequencies are present.
        """
        if chord_detector is None:
            return []

        results: list[MatchResult] = []

        # Find pending notes at the current playback position
        candidates = self._timeline.get_active_notes_at_time(
            playback_ms, self._timing_window_ms
        )
        pending = [
            n for n in candidates
            if self._get_state(n) == MatchType.PENDING and not self._is_filtered(n)
        ]
        if len(pending) < 2:
            return []  # Not a chord — single notes use YIN

        # Group by timestamp (chord = same timestamp)
        chord_groups: dict[float, list[NoteEvent]] = {}
        for note in pending:
            ts = round(note.timestamp_ms, 0)
            chord_groups.setdefault(ts, []).append(note)

        for ts, group in chord_groups.items():
            if len(group) < 2:
                continue  # Single note, not a chord

            # Verify via FFT
            expected_midi = [n.midi_note for n in group]
            present = chord_detector.verify_chord(expected_midi)

            matched_events = []
            all_present = True
            for note, is_present in zip(group, present):
                if is_present:
                    if self._get_state(note) == MatchType.PENDING:
                        self._record_match(note, MatchType.HIT)
                        matched_events.append(note)
                else:
                    all_present = False

            if matched_events:
                results.append(MatchResult(
                    match_type=MatchType.HIT if all_present else MatchType.CLOSE,
                    matched_events=matched_events,
                    semitone_distance=0,
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
        self._timing_observations.clear()
