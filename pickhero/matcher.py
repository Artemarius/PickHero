"""Note matching engine.

Compares detected audio notes against the tab timeline to produce
hit/close/miss feedback. No pygame dependency — pure logic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Callable

from pickhero.audio.note_utils import semitone_distance
from pickhero.audio.input import TimestampedNote
from pickhero.audio.performance import PerformanceEvent, TechniqueSpec, TechniqueVerdict
from pickhero.tabs.timeline import NoteEvent, Timeline
from pickhero.timing import (
    PitchVerdict,
    TimingObservation,
    TimingVerdict,
    classify_pitch_distance,
    classify_timing_error,
    compute_stats,
)

if TYPE_CHECKING:
    from pickhero.audio.analyzer import PerformanceAnalyzer
    from pickhero.config import ToneProfile


class MatchType(Enum):
    PENDING = "pending"
    HIT = "hit"
    CLOSE = "close"
    MISS = "miss"


class MatchMode(Enum):
    """Matching strictness profile.

    ARCADE: forgiving — octave equivalence, chord auto-complete (mark all siblings).
    PRACTICE: partial credit — only the matched note is marked, no auto-complete.
    JUDGE: strict — every note must be independently supported; pitch_strict forced on.
    """
    ARCADE = "arcade"
    PRACTICE = "practice"
    JUDGE = "judge"


def _coerce_match_mode(value: MatchMode | str) -> MatchMode:
    """Accept a MatchMode or its string value (e.g. from config)."""
    if isinstance(value, MatchMode):
        return value
    normalized = str(value).strip().lower()
    for m in MatchMode:
        if m.value == normalized:
            return m
    raise ValueError(f"unknown MatchMode: {value!r}")


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
        chord_partial_credit: bool | None = None,
        timing_judge_enabled: bool | None = None,
        pitch_strict: bool | None = None,
        mode: MatchMode | str | None = None,
    ):
        self._timeline = timeline
        self._timing_window_ms = timing_window_ms
        self._audio_offset_ms = audio_offset_ms
        self._chord_threshold_ms = chord_threshold_ms
        self.note_filter = note_filter

        # Resolve the match mode. Explicit `mode` wins; otherwise derive from
        # the legacy booleans for backward compatibility (migration path).
        if mode is not None:
            self._mode = _coerce_match_mode(mode)
        elif timing_judge_enabled:
            self._mode = MatchMode.JUDGE
        elif chord_partial_credit is False:
            # Legacy chord_partial_credit=False meant "easy mode / mark all",
            # which maps to ARCADE (forgiving auto-complete).
            self._mode = MatchMode.ARCADE
        elif chord_partial_credit is True:
            # Legacy chord_partial_credit=True meant partial credit; closest
            # new equivalent is PRACTICE (only matched note, no auto-complete).
            self._mode = MatchMode.PRACTICE
        else:
            self._mode = MatchMode.ARCADE

        # Legacy boolean aliases (derived from mode). These remain settable
        # for backward compatibility; setting them re-derives the mode.
        # Kept through one release, then removed.
        self.chord_partial_credit = self._mode != MatchMode.ARCADE

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

        # Matched (PerformanceEvent, NoteEvent) pairs handed to the analyzer.
        # Populated as onsets match tab notes; drained by analyze_performance().
        self._matched_pairs: list[tuple[PerformanceEvent, NoteEvent]] = []
        # Verdicts produced by the after-take analyzer (None until analyzed).
        self._analyzed_verdicts: list[TechniqueVerdict] | None = None

        # Miss-scan cursor: index into the timeline's note list marking how far
        # _mark_missed_notes has already scanned. Each frame only inspects newly
        # passed notes instead of re-scanning from 0 — O(new) not O(total).
        self._miss_scan_cursor: int = 0

        # Debug match log (PICKHERO_DEBUG_MATCH=1 enables). Capped at 500.
        self._debug_match: bool = __import__("os").environ.get("PICKHERO_DEBUG_MATCH", "") == "1"
        self._match_log: list[str] = []

    @property
    def audio_offset_ms(self) -> float:
        return self._audio_offset_ms

    @audio_offset_ms.setter
    def audio_offset_ms(self, value: float) -> None:
        self._audio_offset_ms = value

    @property
    def match_mode(self) -> MatchMode:
        """The active matching strictness profile."""
        return self._mode

    @match_mode.setter
    def match_mode(self, value: MatchMode | str) -> None:
        self._mode = _coerce_match_mode(value)
        self.chord_partial_credit = self._mode != MatchMode.ARCADE

    @property
    def timing_judge_enabled(self) -> bool:
        # JUDGE mode implies the timing judge is on; PRACTICE/ARCADE do not.
        return self._mode == MatchMode.JUDGE

    @timing_judge_enabled.setter
    def timing_judge_enabled(self, value: bool) -> None:
        # Legacy setter: enabling the judge promotes to JUDGE mode; disabling
        # drops to ARCADE (the forgiving default). Kept for backward compat.
        if value and self._mode != MatchMode.JUDGE:
            self._mode = MatchMode.JUDGE
        elif not value and self._mode == MatchMode.JUDGE:
            self._mode = MatchMode.ARCADE
        self.chord_partial_credit = self._mode != MatchMode.ARCADE

    @property
    def pitch_strict(self) -> bool:
        # JUDGE mode forces strict pitch (no octave equivalence).
        return self._mode == MatchMode.JUDGE

    @pitch_strict.setter
    def pitch_strict(self, value: bool) -> None:
        # Legacy setter: strict pitch implies JUDGE mode. Kept for backward compat.
        if value and self._mode != MatchMode.JUDGE:
            self._mode = MatchMode.JUDGE
        self.chord_partial_credit = self._mode != MatchMode.ARCADE

    def get_match_log(self) -> list[str]:
        """Return the debug match log (PICKHERO_DEBUG_MATCH=1)."""
        return list(self._match_log)

    def _log(self, msg: str) -> None:
        if self._debug_match:
            import sys
            print(msg, file=sys.stderr)
            if len(self._match_log) < 500:
                self._match_log.append(msg)

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

        # Scan only notes newly past the window since last frame — the
        # _miss_scan_cursor advances monotonically, so each call is O(new)
        # rather than re-scanning every note from the start of the song.
        candidates, new_cursor = self._timeline.get_notes_before(
            cutoff, self._miss_scan_cursor
        )
        self._miss_scan_cursor = new_cursor
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
                if self.timing_judge_enabled:
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

        # Process each detected note. Onset events always match (existing
        # behavior). Non-onset events match only when the tab expects the
        # corresponding technique — this lets HO/PO/slide/tap destinations
        # match without a pick onset.
        for ts_note in detected:
            event_kind = "pick_onset"
            if ts_note.note.performance is not None:
                event_kind = ts_note.note.performance.event_kind

            midi = ts_note.note.midi_note
            ts = ts_note.timestamp_ms
            onset = ts_note.note.is_onset

            if event_kind == "pick_onset":
                # Existing behavior: require onset.
                if not onset:
                    self._log(f"SKIP   midi={midi} ts={ts:.0f} onset=False kind=pick_onset")
                    continue
                mr = self._match_one(ts_note, playback_ms, require_onset=True)
            elif event_kind in ("legato_transition", "slide_landing", "bend_target"):
                # Match only if a pending NoteEvent expects the matching technique.
                expected_kind = {
                    "legato_transition": ("hammer_on", "pull_off"),
                    "slide_landing": ("slide",),
                    "bend_target": ("bend",),
                }[event_kind]
                self._log(f"ROUTED midi={midi} ts={ts:.0f} onset={onset} kind={event_kind}")
                mr = self._match_one(
                    ts_note, playback_ms, require_onset=False,
                    restrict_techniques=expected_kind,
                )
            elif event_kind == "noise_gesture":
                # Dead-note / rake: match by timing window only (pitch may be 0).
                self._log(f"ROUTED midi={midi} ts={ts:.0f} onset={onset} kind=noise_gesture")
                mr = self._match_one(
                    ts_note, playback_ms, require_onset=False,
                    restrict_techniques=("dead_note",),
                    ignore_pitch=True,
                )
            else:
                # sustain_update / release: no matching action here.
                self._log(f"SKIP   midi={midi} ts={ts:.0f} kind={event_kind} (non-action)")
                mr = None
            if mr is not None:
                results.append(mr)

        return results

    def _match_one(
        self,
        ts_note: TimestampedNote,
        playback_ms: float,
        require_onset: bool,
        restrict_techniques: tuple[str, ...] | None = None,
        ignore_pitch: bool = False,
    ) -> MatchResult | None:
        """Match a single detected note against pending NoteEvents.

        ``require_onset``: when True, skip non-onset detected notes (legacy
        behavior for pick_onset events).
        ``restrict_techniques``: when set, only pending NoteEvents whose
        ``techniques`` contain one of these kinds are candidates.
        ``ignore_pitch``: when True, match by timing window only (dead-note /
        rake events carry no reliable pitch).
        """
        if require_onset and not ts_note.note.is_onset:
            self._log(f"DROP   midi={ts_note.note.midi_note} ts={ts_note.timestamp_ms:.0f} require_onset but is_onset=False")
            return None

        adjusted_ms = ts_note.timestamp_ms + self._audio_offset_ms
        detected_midi = ts_note.note.midi_note

        candidates = self._timeline.get_active_notes_at_time(
            adjusted_ms, self._timing_window_ms
        )

        pending = [
            n for n in candidates
            if self._get_state(n) == MatchType.PENDING and not self._is_filtered(n)
        ]
        if restrict_techniques is not None:
            pending = [
                n for n in pending
                if any(t.kind in restrict_techniques for t in n.techniques)
            ]

        self._log(
            f"CAND   midi={detected_midi} ts={adjusted_ms:.0f} "
            f"pending={len(pending)} restrict={restrict_techniques}"
        )
        if not pending:
            # Timing Judge: record extra (stray) onset with no matching note
            if self.timing_judge_enabled and require_onset:
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
                    techniques=nearest.techniques if nearest else None,
                ))
            return None

        # Log the candidate list (which tab notes are in range)
        self._log(
            f"PEND   midi={detected_midi} ts={adjusted_ms:.0f} "
            f"cands=" + ",".join(
                f"m{n.midi_note}@{n.timestamp_ms:.0f}/s{n.string}/f{n.fret}"
                for n in pending
            )
        )
        # Find closest match by semitone distance
        best = None
        best_dist = None
        for note in pending:
            if ignore_pitch:
                # Timing-only match: pick the closest in time (already narrowed
                # to the timing window; take the first pending).
                best = note
                best_dist = 0
                break
            dist = semitone_distance(detected_midi, note.midi_note)
            if self.pitch_strict:
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
            return None

        # Classify match (dead-note / pitch-ignored matches are always HITs)
        if ignore_pitch:
            match_type = MatchType.HIT
            self._log(
                f"MATCH  midi={detected_midi} ts={adjusted_ms:.0f} "
                f"→ m{best.midi_note}@{best.timestamp_ms:.0f}/s{best.string}/f{best.fret} "
                f"dist={best_dist} type={match_type.value} (ignore_pitch)"
            )
        elif best_dist == 0:
            match_type = MatchType.HIT
            self._log(
                f"MATCH  midi={detected_midi} ts={adjusted_ms:.0f} "
                f"→ m{best.midi_note}@{best.timestamp_ms:.0f}/s{best.string}/f{best.fret} "
                f"dist={best_dist} type={match_type.value}"
            )
        elif best_dist == 1 and not self.pitch_strict:
            match_type = MatchType.CLOSE
            self._log(
                f"MATCH  midi={detected_midi} ts={adjusted_ms:.0f} "
                f"→ m{best.midi_note}@{best.timestamp_ms:.0f}/s{best.string}/f{best.fret} "
                f"dist={best_dist} type={match_type.value}"
            )
        else:
            # Too far off — record timing if judge enabled, then skip.
            # Judge A fix: route to timing-observation even for non-onset
            # (restrict_techniques) events, so JUDGE-mode players get feedback
            # for near-miss slides/bends instead of silent drops.
            self._log(
                f"MISS   midi={detected_midi} ts={adjusted_ms:.0f} "
                f"→ best=m{best.midi_note}@{best.timestamp_ms:.0f}/s{best.string} "
                f"dist={best_dist}"
            )
            if self.timing_judge_enabled:
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
                    techniques=best.techniques,
                ))
            return None

        # Resolve hammer/pull direction from the neighbor pitch delta.
        best = self._resolve_legato_direction(best)

        # Record timing observation for the matched note
        if self.timing_judge_enabled:
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
                techniques=best.techniques,
            ))

        # Hand the matched (PerformanceEvent, NoteEvent) pair to the analyzer.
        if ts_note.note.performance is not None:
            self._matched_pairs.append((ts_note.note.performance, best))

        # Chord handling — dispatched by match mode.
        siblings = self._find_chord_siblings(best)
        siblings = [s for s in siblings if not self._is_filtered(s)]

        matched_events: list[NoteEvent] = []
        if self._mode == MatchMode.JUDGE:
            if self._get_state(best) == MatchType.PENDING:
                self._record_match(best, match_type)
                matched_events.append(best)
        elif self._mode == MatchMode.PRACTICE:
            if self._get_state(best) == MatchType.PENDING:
                self._record_match(best, match_type)
                matched_events.append(best)
        else:  # ARCADE
            for sibling in siblings:
                if self._get_state(sibling) == MatchType.PENDING:
                    self._record_match(sibling, match_type)
                    matched_events.append(sibling)
            if best not in matched_events:
                if self._get_state(best) == MatchType.PENDING:
                    self._record_match(best, match_type)
                    matched_events.append(best)

        return MatchResult(
            match_type=match_type,
            matched_events=matched_events,
            semitone_distance=best_dist,
        )

    def verify_chord_at(
        self,
        playback_ms: float,
        chord_detector=None,
        has_onset: bool = False,
    ) -> list[MatchResult]:
        """Verify chords at the hit zone using FFT spectral analysis.

        Called when there are multiple pending notes at the same timestamp.
        Uses the ChordDetector to check if expected frequencies are present.
        When ``has_onset`` is True, a fresh FFT analysis is triggered;
        otherwise the cached result from the last onset is returned.
        """
        if chord_detector is None:
            return []

        results: list[MatchResult] = []

        # Find all active notes at the current playback position. A chord in the
        # tab is defined by ≥2 notes at the same timestamp, even if YIN already
        # matched some of them. Verify the full chord so the FFT can pick up
        # notes the monophonic YIN detector missed (e.g., the fifth of a power
        # chord after the root was already hit).
        candidates = self._timeline.get_active_notes_at_time(
            playback_ms, self._timing_window_ms
        )
        chord_groups: dict[float, list[NoteEvent]] = {}
        for note in candidates:
            if self._is_filtered(note):
                continue
            ts = round(note.timestamp_ms, 0)
            chord_groups.setdefault(ts, []).append(note)

        for ts, group in chord_groups.items():
            if len(group) < 2:
                continue  # Single note, not a chord

            # Skip if every note in the chord is already resolved.
            pending = [n for n in group if self._get_state(n) == MatchType.PENDING]
            if not pending:
                continue

            # Verify via FFT (onset-gated: fresh analysis on new strikes,
            # cached result during sustain to avoid flutter).
            # JUDGE mode uses a stricter energy-ratio threshold (0.12 vs 0.08)
            # to reduce false positives from sympathetic resonance.
            expected_midi = [n.midi_note for n in group]
            min_ratio = 0.12 if self._mode == MatchMode.JUDGE else 0.08
            present = chord_detector.verify_chord_with_onset(
                expected_midi, has_onset, min_energy_ratio=min_ratio
            )

            # Guard: if chord_detector returns a mismatched number of results
            # (shouldn't happen, but defensive), skip this group entirely.
            if len(present) != len(group):
                continue

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

    def _resolve_legato_direction(self, note: NoteEvent) -> NoteEvent:
        """Resolve hammer_on vs pull_off from the neighbor pitch delta.

        pyguitarpro encodes both as ``hammer=True``; the tab never expects
        ``pull_off``. We infer ``hammer_on`` if the destination midi is higher
        than the previous note on the same string, else ``pull_off``. Since
        ``TechniqueSpec`` is frozen, the spec is replaced via
        :func:`dataclasses.replace` and the NoteEvent rebuilt with the new
        techniques tuple.
        """
        if not note.techniques:
            return note
        has_legato = any(
            s.kind == "hammer_on" and s.tied_to_previous for s in note.techniques
        )
        if not has_legato:
            return note
        # Find the previous NoteEvent on the same string in the timeline.
        prev_note = self._find_previous_note_on_string(note)
        if prev_note is None:
            return note
        new_specs: list[TechniqueSpec] = []
        for s in note.techniques:
            if s.kind == "hammer_on" and s.tied_to_previous:
                # Direction: dest > prev → hammer_on, else pull_off.
                kind = "hammer_on" if note.midi_note > prev_note.midi_note else "pull_off"
                direction = "up" if kind == "hammer_on" else "down"
                s = replace(s, kind=kind, direction=direction)
            new_specs.append(s)
        return replace(note, techniques=tuple(new_specs))

    def _find_previous_note_on_string(self, note: NoteEvent) -> NoteEvent | None:
        """Return the most recent NoteEvent on the same string before `note`."""
        # Scan backwards in time from the note's timestamp.
        window_start = max(0.0, note.timestamp_ms - 10_000.0)
        candidates = self._timeline.get_notes_in_range(window_start, note.timestamp_ms)
        prev = None
        for c in candidates:
            if c.string == note.string and c.timestamp_ms < note.timestamp_ms:
                if prev is None or c.timestamp_ms > prev.timestamp_ms:
                    prev = c
        return prev

    def analyze_performance(
        self, tone_profile: "ToneProfile | None" = None
    ) -> list[PerformanceEvent]:
        """Run the after-take analyzer over collected matched pairs.

        Instantiates :class:`~pickhero.audio.analyzer.PerformanceAnalyzer`,
        runs it over ``self._matched_pairs``, stores the resulting verdicts for
        ``get_statistics``, and returns the graded events. Called by the game
        loop at song end. Also clears ``_matched_pairs``.
        """
        from pickhero.audio.analyzer import PerformanceAnalyzer

        if not self._matched_pairs:
            self._analyzed_verdicts = []
            return []
        analyzer = PerformanceAnalyzer(tone_profile)
        events = analyzer.analyze(self._matched_pairs)
        # Keep a snapshot of the pairs for the optional offline polyphonic pass
        # (Patch 6d) before clearing.
        self._last_pairs = list(self._matched_pairs)
        # Flatten verdicts for stats.
        all_verdicts: list[TechniqueVerdict] = []
        for ev in events:
            all_verdicts.extend(ev.verdicts)
        self._analyzed_verdicts = all_verdicts
        self._matched_pairs = []
        return events

    def analyze_performance_offline(
        self,
        raw_audio: "np.ndarray | None",
        sample_rate: int,
        tone_profile: "ToneProfile | None" = None,
    ) -> list[TechniqueVerdict]:
        """Run the offline polyphonic analyzer over the take audio.

        Called by the game loop after :meth:`analyze_performance` when the
        active preset has ``offline_deep_analysis=True`` and a take was
        recorded. Returns new verdicts (unison bends, pinch harmonic
        verification) to be merged into ``all_verdicts``.
        """
        if raw_audio is None or len(raw_audio) == 0:
            return []
        pairs = getattr(self, "_last_pairs", None) or []
        if not pairs:
            return []
        from pickhero.audio.polyphonic_analyzer import PolyphonicAnalyzer
        analyzer = PolyphonicAnalyzer(raw_audio, sample_rate, pairs)
        verdicts = analyzer.analyze()
        # Merge into the stored verdicts for get_statistics.
        if self._analyzed_verdicts is not None:
            self._analyzed_verdicts.extend(verdicts)
        return verdicts

    def get_statistics(self) -> dict:
        """Return current match statistics."""
        total = self.hits + self.close + self.misses
        accuracy = (self.hits / total * 100) if total > 0 else 0.0

        # Technique accuracy: from analyzer verdicts if analyzed, else 0.
        # technique_correct = verdicts with grade in {good, ok};
        # technique_total = count of verdicts produced.
        technique_correct = 0
        technique_total = 0
        if self._analyzed_verdicts is not None:
            for v in self._analyzed_verdicts:
                technique_total += 1
                if v.grade in ("good", "ok"):
                    technique_correct += 1
        technique_accuracy = (
            (technique_correct / technique_total * 100)
            if technique_total > 0 else 0.0
        )

        return {
            "hits": self.hits,
            "close": self.close,
            "misses": self.misses,
            "total": total,
            "accuracy_percent": accuracy,
            "technique_correct": technique_correct,
            "technique_total": technique_total,
            "technique_accuracy_percent": technique_accuracy,
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
        self._matched_pairs.clear()
        self._analyzed_verdicts = None
        self._miss_scan_cursor = 0
        self._match_log.clear()
