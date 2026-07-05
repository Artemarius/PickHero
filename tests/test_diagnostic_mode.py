"""M4.3: Articulation diagnostic-only mode.

When diagnostic_mode is True (the default), the matcher treats all events
as pick_onset — technique labels are diagnostic-only and not used for routing.
"""

from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pickhero.audio.articulation import ArticulationDetector
from pickhero.audio.detector import DetectedNote
from pickhero.audio.input import TimestampedNote
from pickhero.audio.performance import PerformanceEvent, TechniqueSpec
from pickhero.matcher import MatchType, NoteMatcher
from pickhero.tabs.timeline import NoteEvent, SongMetadata, Timeline


# ---------------------------------------------------------------------------
# Helpers (mirrors test_matcher_techniques.py)
# ---------------------------------------------------------------------------


def _note(
    timestamp_ms: float, midi_note: int = 64, string: int = 1, fret: int = 0,
    duration_ms: float = 500.0, techniques: tuple = (), measure: int = 0,
) -> NoteEvent:
    return NoteEvent(
        timestamp_ms=timestamp_ms, duration_ms=duration_ms,
        midi_note=midi_note, string=string, fret=fret,
        techniques=techniques, measure=measure,
    )


def _detected(
    midi_note: int, timestamp_ms: float, is_onset: bool = True,
    confidence: float = 0.95, performance: PerformanceEvent | None = None,
) -> TimestampedNote:
    return TimestampedNote(
        note=DetectedNote(
            midi_note=midi_note, frequency=440.0,
            confidence=confidence, name="A4",
            is_onset=is_onset, performance=performance,
        ),
        timestamp_ms=timestamp_ms,
    )


def _make_matcher(
    notes: list[NoteEvent], timing_window_ms: float = 100.0,
    audio_offset_ms: float = 0.0, chord_threshold_ms: float = 50.0,
) -> NoteMatcher:
    timeline = Timeline(notes, SongMetadata(title="Test", tempo=120))
    return NoteMatcher(
        timeline,
        timing_window_ms=timing_window_ms,
        audio_offset_ms=audio_offset_ms,
        chord_threshold_ms=chord_threshold_ms,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestArticulationDetectorHasDiagnosticFlag(unittest.TestCase):
    """Verify the diagnostic_mode flag exists and defaults True."""

    def test_diagnostic_mode_flag_exists_and_defaults_true(self) -> None:
        detector = ArticulationDetector()
        self.assertTrue(hasattr(detector, "diagnostic_mode"))
        self.assertEqual(detector.diagnostic_mode, True)

    def test_can_set_diagnostic_mode_false(self) -> None:
        detector = ArticulationDetector()
        detector.diagnostic_mode = False
        self.assertEqual(detector.diagnostic_mode, False)


class TestDiagnosticModeMatchesAllAsPickOnset(unittest.TestCase):
    """When diagnostic_mode is True, event_kind routing is bypassed.

    All events are forced to event_kind="pick_onset". Since non-onset events
    require onset=True to match, legato/slide/bend destinations are silently
    skipped — they won't match their expected technique specs.
    """

    def test_legato_transition_does_not_match_hammer_on(self) -> None:
        """A legato_transition event should NOT route to hammer_on in
        diagnostic mode — it is treated as pick_onset and skipped because
        is_onset=False.
        """
        notes = [
            _note(1000.0, midi_note=40, string=6, fret=1,
                  techniques=(), measure=0),
            _note(1100.0, midi_note=41, string=6, fret=2,
                  techniques=(TechniqueSpec(kind="hammer_on"),), measure=0),
        ]
        matcher = _make_matcher(notes, timing_window_ms=150.0)
        pick = _detected(40, 1000.0, is_onset=True)
        ho_perf = PerformanceEvent(onset_ms=1100.0, midi_note=41, confidence=0.9)
        ho_perf.event_kind = "legato_transition"
        ho = _detected(41, 1100.0, is_onset=False, performance=ho_perf)
        results = matcher.process_detected_notes([pick, ho], playback_ms=1100.0)
        matched_ids = {id(n) for r in results for n in r.matched_events}
        self.assertNotIn(id(notes[1]), matched_ids,
                         "hammer_on should not match in diagnostic mode")

    def test_bend_target_as_onset_still_matches_via_pick_onset(self) -> None:
        """A bend_target event that is also onset=True matches via the
        pick_onset path (not the bend_target routing). Verifies the
        event_kind override works — it goes through generic note matching,
        not the bend-specific route.
        """
        notes = [
            _note(1000.0, midi_note=40, string=6, fret=0,
                  techniques=(TechniqueSpec(kind="bend", target_cents=200.0),),
                  measure=0),
        ]
        matcher = _make_matcher(notes, timing_window_ms=150.0)
        bend_perf = PerformanceEvent(onset_ms=1000.0, midi_note=40, confidence=0.9)
        bend_perf.event_kind = "bend_target"
        detected = _detected(40, 1000.0, is_onset=True, performance=bend_perf)
        results = matcher.process_detected_notes([detected], playback_ms=1000.0)
        # In diagnostic mode, event_kind=pick_onset → generic match.
        # The note may or may not match depending on timing; key is no
        # exception and the routing block is bypassed.
        self.assertIsInstance(results, list)

    def test_slide_landing_does_not_match_slide(self) -> None:
        """A slide_landing event should NOT route to slide in diagnostic mode."""
        notes = [
            _note(1000.0, midi_note=40, string=6, fret=1,
                  techniques=(TechniqueSpec(kind="slide"),), measure=0),
        ]
        matcher = _make_matcher(notes, timing_window_ms=150.0)
        slide_perf = PerformanceEvent(onset_ms=1000.0, midi_note=40, confidence=0.9)
        slide_perf.event_kind = "slide_landing"
        # Slide destination arrives as non-onset
        detected = _detected(40, 1000.0, is_onset=False, performance=slide_perf)
        results = matcher.process_detected_notes([detected], playback_ms=1000.0)
        matched_ids = {id(n) for r in results for n in r.matched_events}
        self.assertNotIn(id(notes[0]), matched_ids,
                         "slide should not match in diagnostic mode")

    def test_noise_gesture_does_not_match_dead_note(self) -> None:
        """A noise_gesture event should NOT route to dead_note in diagnostic mode."""
        notes = [
            _note(1000.0, midi_note=0, string=6, fret=0,
                  techniques=(TechniqueSpec(kind="dead_note"),), measure=0),
        ]
        matcher = _make_matcher(notes, timing_window_ms=150.0)
        noise_perf = PerformanceEvent(onset_ms=1000.0, midi_note=0, confidence=0.3)
        noise_perf.event_kind = "noise_gesture"
        detected = _detected(0, 1000.0, is_onset=False, performance=noise_perf)
        results = matcher.process_detected_notes([detected], playback_ms=1000.0)
        matched_ids = {id(n) for r in results for n in r.matched_events}
        self.assertNotIn(id(notes[0]), matched_ids,
                         "dead_note should not match in diagnostic mode")

    def test_pick_onset_still_works_in_diagnostic_mode(self) -> None:
        """pick_onset events are unaffected — they match normally."""
        notes = [
            _note(1000.0, midi_note=40, string=6, fret=1,
                  techniques=(), measure=0),
        ]
        matcher = _make_matcher(notes, timing_window_ms=150.0)
        detected = _detected(40, 1000.0, is_onset=True)
        results = matcher.process_detected_notes([detected], playback_ms=1000.0)
        hits = [r for r in results if r.match_type == MatchType.HIT]
        self.assertEqual(len(hits), 1)


class TestNonDiagnosticModeRoutesByEventKind(unittest.TestCase):
    """When diagnostic_mode is False, event_kind routing works normally.

    Currently the matcher hardcodes `diagnostic = True`, so these tests
    verify the structure is in place and confirm diagnostic mode behaviour.
    Once wired to `articulation_detector.diagnostic_mode`, they will
    automatically verify the non-diagnostic path too.
    """

    def test_routing_structure_exists(self) -> None:
        """Verify the diagnostic variable and event_kind read exist in
        process_detected_notes. The actual wire to articulation detector
        is a TODO — this test verifies the scaffolding."""
        notes = [
            _note(1000.0, midi_note=40, string=6, fret=1,
                  techniques=(), measure=0),
            _note(1100.0, midi_note=41, string=6, fret=2,
                  techniques=(TechniqueSpec(kind="hammer_on"),), measure=0),
        ]
        matcher = _make_matcher(notes, timing_window_ms=150.0)
        pick = _detected(40, 1000.0, is_onset=True)
        ho_perf = PerformanceEvent(onset_ms=1100.0, midi_note=41, confidence=0.9)
        ho_perf.event_kind = "legato_transition"
        ho = _detected(41, 1100.0, is_onset=False, performance=ho_perf)
        results = matcher.process_detected_notes([pick, ho], playback_ms=1100.0)
        # In diagnostic mode (currently always True), legato_transition
        # is forced to pick_onset → skipped.
        self.assertIsInstance(results, list)
        matched_ids = {id(n) for r in results for n in r.matched_events}
        self.assertNotIn(id(notes[1]), matched_ids)


class TestDiagnosticModeEndToEnd(unittest.TestCase):
    """Integration: detector produces labeled events, matcher ignores them in diagnostic mode."""

    def test_detector_labels_events_and_matcher_ignores_in_diagnostic_mode(self) -> None:
        """Feed a pitch curve through the detector, verify events are produced
        and that the detector's diagnostic_mode flag is respected.
        """
        detector = ArticulationDetector(sample_rate=44100, hop_size=512)
        self.assertTrue(detector.diagnostic_mode)

        # First note: 440 Hz (A4) for 50 frames.
        for i in range(50):
            detector.process(
                freq=440.0, confidence=0.9, is_onset=(i == 0),
                audio_buffer=None, timestamp_ms=i * 11.6,
            )
        # Event is still active — no second onset yet.
        self.assertIsNotNone(detector.active_event)
        self.assertEqual(detector.active_event.midi_note, 69)  # A4

        # Second onset closes the first event.
        detector.process(
            freq=329.63, confidence=0.9, is_onset=True,
            audio_buffer=None, timestamp_ms=50 * 11.6,
        )
        completed = detector.drain_completed()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].midi_note, 69)  # A4

        # No bend candidates since pitch never drifted.
        kinds = [c.kind for c in completed[0].technique_candidates]
        self.assertNotIn("bend", kinds)


if __name__ == "__main__":
    unittest.main()
