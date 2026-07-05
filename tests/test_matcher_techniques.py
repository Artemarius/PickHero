"""Tests for the NoteMatcher technique-handling flow.

Covers:
- _resolve_legato_direction (hammer_on / pull_off resolution from pitch delta)
- analyze_performance (verdict population from matched pairs)
- get_statistics (technique-accuracy reporting from verdicts)
"""

import numpy as np
import pytest

from pickhero.audio.detector import DetectedNote
from pickhero.audio.input import TimestampedNote
from pickhero.audio.performance import PerformanceEvent, TechniqueSpec, TechniqueVerdict
from pickhero.matcher import NoteMatcher
from pickhero.tabs.timeline import NoteEvent, SongMetadata, Timeline


def _note(
    timestamp_ms: float,
    midi_note: int = 64,
    string: int = 1,
    fret: int = 0,
    duration_ms: float = 500.0,
    techniques: tuple = (),
    measure: int = 0,
) -> NoteEvent:
    return NoteEvent(
        timestamp_ms=timestamp_ms,
        duration_ms=duration_ms,
        midi_note=midi_note,
        string=string,
        fret=fret,
        techniques=techniques,
        measure=measure,
    )


def _detected(
    midi_note: int,
    timestamp_ms: float,
    is_onset: bool = True,
    confidence: float = 0.95,
    performance=None,
) -> TimestampedNote:
    return TimestampedNote(
        note=DetectedNote(
            midi_note=midi_note,
            frequency=440.0,
            confidence=confidence,
            name="A4",
            is_onset=is_onset,
            performance=performance,
        ),
        timestamp_ms=timestamp_ms,
    )


def _make_matcher(
    notes: list[NoteEvent],
    timing_window_ms: float = 100.0,
    audio_offset_ms: float = 0.0,
    chord_threshold_ms: float = 50.0,
    mode: str = "judge",
) -> NoteMatcher:
    timeline = Timeline(notes, SongMetadata(title="Test", tempo=120))
    return NoteMatcher(
        timeline,
        timing_window_ms=timing_window_ms,
        audio_offset_ms=audio_offset_ms,
        chord_threshold_ms=chord_threshold_ms,
        mode=mode,
    )


# ── _resolve_legato_direction ────────────────────────────────────────────────


class TestResolveLegatoDirection:
    """Test that _resolve_legato_direction picks the right kind & direction."""

    def test_ascending_keeps_hammer_on(self):
        """prev midi=40 string=6, cur midi=42 string=6 → hammer_on, direction='up'."""
        prev = _note(500.0, midi_note=40, string=6)
        cur = _note(
            1000.0,
            midi_note=42,
            string=6,
            techniques=(
                TechniqueSpec(kind="hammer_on", tied_to_previous=True),
            ),
        )
        matcher = _make_matcher([prev, cur])
        result = matcher._resolve_legato_direction(cur)
        spec = result.techniques[0]
        assert spec.kind == "hammer_on"
        assert spec.direction == "up"

    def test_descending_flips_to_pull_off(self):
        """prev midi=42 string=6, cur midi=40 string=6 → pull_off, direction='down'."""
        prev = _note(500.0, midi_note=42, string=6)
        cur = _note(
            1000.0,
            midi_note=40,
            string=6,
            techniques=(
                TechniqueSpec(kind="hammer_on", tied_to_previous=True),
            ),
        )
        matcher = _make_matcher([prev, cur])
        result = matcher._resolve_legato_direction(cur)
        spec = result.techniques[0]
        assert spec.kind == "pull_off"
        assert spec.direction == "down"

    def test_no_legato_spec_unchanged(self):
        """A note with no legato technique is returned unchanged."""
        note = _note(1000.0, midi_note=64, string=1)
        matcher = _make_matcher([note])
        result = matcher._resolve_legato_direction(note)
        assert result is note
        assert result.techniques == ()

    def test_non_legato_technique_not_mutated(self):
        """A non-hammer_on spec is left untouched even when tied_to_previous is False."""
        note = _note(
            1000.0,
            midi_note=64,
            string=1,
            techniques=(TechniqueSpec(kind="bend", target_cents=100.0, tied_to_previous=True),),
        )
        matcher = _make_matcher([note])
        result = matcher._resolve_legato_direction(note)
        # No hammer_on+legato → function returns early, same object
        assert result is note
        assert result.techniques[0].kind == "bend"
        assert result.techniques[0].direction is None


# ── analyze_performance ──────────────────────────────────────────────────────


class TestAnalyzePerformance:
    """Test that analyze_performance grades matched pairs."""

    def _make_bend_event(self, target_cents: float, n_frames: int = 18) -> PerformanceEvent:
        """Build a synthetic bend PerformanceEvent with a rising f0 curve."""
        times = np.linspace(0, 0.2, n_frames)
        max_cents = min(target_cents * 0.95, target_cents)  # slightly undershoot for realism
        cents = np.minimum(times / 0.2 * max_cents, target_cents)
        return PerformanceEvent(
            onset_ms=0.0,
            f0_curve=[
                (t * 1000, 82.41 * 2 ** (c / 1200), float(c)) for t, c in zip(times, cents)
            ],
            energy_envelope=[(t * 1000, 0.5) for t in times],
            midi_note=40,
        )

    def test_populates_verdicts(self):
        """A matched pair with a bend spec gets a 'good' bend verdict."""
        event = self._make_bend_event(target_cents=100.0)
        note = _note(
            0.0,
            midi_note=40,
            string=6,
            fret=0,
            techniques=(TechniqueSpec(kind="bend", target_cents=100.0),),
        )
        matcher = _make_matcher([note])
        matcher._matched_pairs.append((event, note))
        events = matcher.analyze_performance()
        assert len(events) == 1
        assert len(events[0].verdicts) >= 1
        assert events[0].verdicts[0].kind == "bend"
        assert events[0].verdicts[0].grade == "good"

    def test_empty_pairs_returns_empty(self):
        """analyze_performance with no matched pairs returns [] and stats show 0."""
        matcher = _make_matcher([])
        events = matcher.analyze_performance()
        assert events == []
        stats = matcher.get_statistics()
        assert stats["technique_total"] == 0
        assert stats["technique_accuracy_percent"] == 0.0

    def test_verdict_explanation_present(self):
        """The verdict for a bend includes a human-readable explanation."""
        event = self._make_bend_event(target_cents=100.0)
        note = _note(
            0.0,
            midi_note=40,
            string=6,
            fret=0,
            techniques=(TechniqueSpec(kind="bend", target_cents=100.0),),
        )
        matcher = _make_matcher([note])
        matcher._matched_pairs.append((event, note))
        matcher.analyze_performance()
        v = matcher._analyzed_verdicts[0]
        assert v.explanation != ""
        assert "Bend reached" in v.explanation


# ── get_statistics (technique accuracy) ──────────────────────────────────────


class TestGetStatistics:
    """Test technique-accuracy reporting from verdicts."""

    def _make_bend_event(self, target_cents: float, n_frames: int = 18) -> PerformanceEvent:
        times = np.linspace(0, 0.2, n_frames)
        cents = np.minimum(times / 0.2 * target_cents, target_cents)
        return PerformanceEvent(
            onset_ms=0.0,
            f0_curve=[
                (t * 1000, 82.41 * 2 ** (c / 1200), float(c)) for t, c in zip(times, cents)
            ],
            energy_envelope=[(t * 1000, 0.5) for t in times],
            midi_note=40,
        )

    def test_reports_technique_accuracy(self):
        """One good bend + one weak bend → 50% technique accuracy."""
        good_event = self._make_bend_event(target_cents=100.0)
        # A weak bend: only reaches ~30 cents of a 100-cent target.
        weak_event = self._make_bend_event(target_cents=100.0, n_frames=18)
        # Manually craft a curve that only rises 30 cents → should grade 'weak'.
        times = np.linspace(0, 0.2, 18)
        weak_cents = np.minimum(times / 0.2 * 30, 30)
        weak_event = PerformanceEvent(
            onset_ms=0.0,
            f0_curve=[
                (t * 1000, 82.41 * 2 ** (c / 1200), float(c))
                for t, c in zip(times, weak_cents)
            ],
            energy_envelope=[(t * 1000, 0.5) for t in times],
            midi_note=40,
        )
        notes = [
            _note(0.0, midi_note=40, string=6, fret=0,
                  techniques=(TechniqueSpec(kind="bend", target_cents=100.0),)),
            _note(500.0, midi_note=40, string=6, fret=0,
                  techniques=(TechniqueSpec(kind="bend", target_cents=100.0),)),
        ]
        matcher = _make_matcher(notes)
        matcher._matched_pairs.append((good_event, notes[0]))
        matcher._matched_pairs.append((weak_event, notes[1]))
        matcher.analyze_performance()
        stats = matcher.get_statistics()
        assert stats["technique_total"] == 2
        assert stats["technique_correct"] == 1
        assert stats["technique_accuracy_percent"] == 50.0

    def test_no_verdicts_before_analysis(self):
        """Before analyze_performance, technique stats are 0."""
        notes = [_note(0.0, midi_note=64)]
        matcher = _make_matcher(notes)
        stats = matcher.get_statistics()
        assert stats["technique_total"] == 0
        assert stats["technique_accuracy_percent"] == 0.0

    def test_all_good_techniques_100_pct(self):
        """Two good bends → 100% technique accuracy."""
        ev1 = self._make_bend_event(target_cents=100.0)
        ev2 = self._make_bend_event(target_cents=100.0)
        notes = [
            _note(0.0, midi_note=40, string=6, fret=0,
                  techniques=(TechniqueSpec(kind="bend", target_cents=100.0),)),
            _note(500.0, midi_note=40, string=6, fret=0,
                  techniques=(TechniqueSpec(kind="bend", target_cents=100.0),)),
        ]
        matcher = _make_matcher(notes)
        matcher._matched_pairs.append((ev1, notes[0]))
        matcher._matched_pairs.append((ev2, notes[1]))
        matcher.analyze_performance()
        stats = matcher.get_statistics()
        assert stats["technique_total"] == 2
        assert stats["technique_correct"] == 2
        assert stats["technique_accuracy_percent"] == 100.0


# ── Non-onset event matching (Patch 3b) ──────────────────────────────────────


class TestNonOnsetMatching:
    """Hammer-on / pull-off / slide / bend destinations arrive as non-onset
    events (event_kind = legato_transition / slide_landing / bend_target). The
    matcher must pair them with pending NoteEvents expecting the matching
    technique, even though is_onset is False."""

    def test_hammer_on_destination_matches_without_pick_onset(self):
        """A picked note at t=1000 followed by a hammer-on (legato_transition)
        at t=1100 with no onset matches the hammer_on NoteEvent."""
        notes = [
            _note(1000.0, midi_note=40, string=6, fret=1,
                  techniques=(), measure=0),
            _note(1100.0, midi_note=41, string=6, fret=2,
                  techniques=(TechniqueSpec(kind="hammer_on"),), measure=0),
        ]
        matcher = _make_matcher(notes, timing_window_ms=150.0)
        # Picked onset at t≈1000
        pick = _detected(40, 1000.0, is_onset=True)
        # Hammer-on: no onset, event_kind=legato_transition, pitch=41 (F2)
        ho_perf = PerformanceEvent(onset_ms=1100.0, midi_note=41, confidence=0.9)
        ho_perf.event_kind = "legato_transition"
        ho = _detected(41, 1100.0, is_onset=False, performance=ho_perf)
        results = matcher.process_detected_notes([pick, ho], playback_ms=1100.0)
        # The hammer_on note (notes[1]) must be marked as matched
        matched_ids = {id(n) for r in results for n in r.matched_events}
        assert id(notes[1]) in matched_ids, "hammer_on destination did not match"

    def test_pull_off_destination_matches_without_pick_onset(self):
        """A picked note at t=1000 followed by a pull-off (legato_transition,
        descending pitch) at t=1100 with no onset matches the pull_off note."""
        notes = [
            _note(1000.0, midi_note=43, string=6, fret=4,
                  techniques=(), measure=0),
            _note(1100.0, midi_note=41, string=6, fret=2,
                  techniques=(TechniqueSpec(kind="pull_off"),), measure=0),
        ]
        matcher = _make_matcher(notes, timing_window_ms=150.0)
        pick = _detected(43, 1000.0, is_onset=True)
        po_perf = PerformanceEvent(onset_ms=1100.0, midi_note=41, confidence=0.9)
        po_perf.event_kind = "legato_transition"
        po = _detected(41, 1100.0, is_onset=False, performance=po_perf)
        results = matcher.process_detected_notes([pick, po], playback_ms=1100.0)
        matched_ids = {id(n) for r in results for n in r.matched_events}
        assert id(notes[1]) in matched_ids, "pull_off destination did not match"

    def test_compound_bend_vibrato_emits_two_verdicts(self):
        """A PerformanceEvent carrying both bend and vibrato candidates (built
        via upsert) produces two verdicts from analyze_performance."""
        notes = [
            _note(0.0, midi_note=40, string=6, fret=0,
                  techniques=(
                      TechniqueSpec(kind="bend", target_cents=100.0),
                      TechniqueSpec(kind="vibrato"),
                  ), measure=0),
        ]
        matcher = _make_matcher(notes)
        ev = PerformanceEvent(onset_ms=0.0, midi_note=40, confidence=0.9)
        # Build compound candidates via upsert (mirrors the articulation detector)
        ev.upsert_technique_candidate("bend", 0.8, detected_cents=100.0,
                                       target_cents=None)
        ev.upsert_technique_candidate("vibrato", 0.7, metrics={"rate_hz": 5.5})
        matcher._matched_pairs.append((ev, notes[0]))
        matcher.analyze_performance()
        assert len(ev.verdicts) >= 2, (
            f"expected >=2 verdicts for compound bend+vibrato, got {len(ev.verdicts)}"
        )
        kinds = {v.kind for v in ev.verdicts}
        assert "bend" in kinds and "vibrato" in kinds, (
            f"expected bend+vibrato verdicts, got {kinds}"
        )
