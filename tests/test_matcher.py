"""Tests for pickhero.matcher module."""

import pytest

from pickhero.audio.detector import DetectedNote
from pickhero.audio.input import TimestampedNote
from pickhero.matcher import MatchType, MatchResult, NoteMatcher
from pickhero.tabs.timeline import NoteEvent, SongMetadata, Timeline
from pickhero.timing import PitchVerdict, TimingVerdict


def _note_event(timestamp_ms: float, midi_note: int = 64, string: int = 1,
                fret: int = 0, duration_ms: float = 500.0) -> NoteEvent:
    return NoteEvent(
        timestamp_ms=timestamp_ms,
        duration_ms=duration_ms,
        midi_note=midi_note,
        string=string,
        fret=fret,
    )


def _detected(midi_note: int, timestamp_ms: float, is_onset: bool = True,
              confidence: float = 0.95) -> TimestampedNote:
    return TimestampedNote(
        note=DetectedNote(
            midi_note=midi_note,
            frequency=440.0,  # placeholder
            confidence=confidence,
            name="A4",  # placeholder
            is_onset=is_onset,
        ),
        timestamp_ms=timestamp_ms,
    )


def _make_matcher(notes: list[NoteEvent], timing_window_ms: float = 100.0,
                  audio_offset_ms: float = 0.0,
                  chord_threshold_ms: float = 50.0) -> NoteMatcher:
    timeline = Timeline(notes, SongMetadata(title="Test", tempo=120))
    return NoteMatcher(
        timeline,
        timing_window_ms=timing_window_ms,
        audio_offset_ms=audio_offset_ms,
        chord_threshold_ms=chord_threshold_ms,
    )


class TestExactHit:
    def test_exact_midi_match(self):
        """Detected MIDI matches tab note exactly -> HIT."""
        tab_note = _note_event(1000.0, midi_note=64)
        matcher = _make_matcher([tab_note])

        detected = [_detected(64, 1000.0)]
        results = matcher.process_detected_notes(detected, 1050.0)

        hits = [r for r in results if r.match_type == MatchType.HIT]
        assert len(hits) == 1
        assert hits[0].semitone_distance == 0
        assert tab_note in hits[0].matched_events
        assert matcher.get_note_state(tab_note) == MatchType.HIT

    def test_exact_hit_updates_statistics(self):
        tab_note = _note_event(1000.0, midi_note=64)
        matcher = _make_matcher([tab_note])

        detected = [_detected(64, 1000.0)]
        matcher.process_detected_notes(detected, 1050.0)

        stats = matcher.get_statistics()
        assert stats["hits"] == 1
        assert stats["accuracy_percent"] == 100.0


class TestCloseMatch:
    def test_one_semitone_above(self):
        """±1 semitone -> CLOSE."""
        tab_note = _note_event(1000.0, midi_note=64)
        matcher = _make_matcher([tab_note])

        detected = [_detected(65, 1000.0)]  # +1 semitone
        results = matcher.process_detected_notes(detected, 1050.0)

        close = [r for r in results if r.match_type == MatchType.CLOSE]
        assert len(close) == 1
        assert close[0].semitone_distance == 1
        assert matcher.get_note_state(tab_note) == MatchType.CLOSE

    def test_one_semitone_below(self):
        tab_note = _note_event(1000.0, midi_note=64)
        matcher = _make_matcher([tab_note])

        detected = [_detected(63, 1000.0)]  # -1 semitone
        results = matcher.process_detected_notes(detected, 1050.0)

        close = [r for r in results if r.match_type == MatchType.CLOSE]
        assert len(close) == 1


class TestWrongNoteIgnored:
    def test_far_off_note_no_penalty(self):
        """>1 semitone away, no matching candidate -> no penalty."""
        tab_note = _note_event(1000.0, midi_note=64)
        matcher = _make_matcher([tab_note])

        detected = [_detected(70, 1000.0)]  # 6 semitones off
        results = matcher.process_detected_notes(detected, 1050.0)

        # No HIT or CLOSE results, note stays PENDING
        hit_close = [r for r in results if r.match_type in (MatchType.HIT, MatchType.CLOSE)]
        assert len(hit_close) == 0
        assert matcher.get_note_state(tab_note) == MatchType.PENDING


class TestMissedNote:
    def test_advance_past_window_without_detection(self):
        """Advance past window with no detection -> MISS."""
        tab_note = _note_event(1000.0, midi_note=64)
        matcher = _make_matcher([tab_note], timing_window_ms=100.0)

        # Advance playback to 1200ms (past 1000 + 100 window)
        results = matcher.process_detected_notes([], 1200.0)

        misses = [r for r in results if r.match_type == MatchType.MISS]
        assert len(misses) == 1
        assert matcher.get_note_state(tab_note) == MatchType.MISS

    def test_miss_updates_statistics(self):
        tab_note = _note_event(1000.0, midi_note=64)
        matcher = _make_matcher([tab_note], timing_window_ms=100.0)

        matcher.process_detected_notes([], 1200.0)

        stats = matcher.get_statistics()
        assert stats["misses"] == 1
        assert stats["accuracy_percent"] == 0.0


class TestChordMatching:
    def test_match_one_note_of_chord_marks_all(self):
        """Match one note of a simultaneous pair -> both marked HIT."""
        note_a = _note_event(1000.0, midi_note=64, string=1)
        note_b = _note_event(1000.0, midi_note=59, string=2)
        matcher = _make_matcher([note_a, note_b], chord_threshold_ms=50.0)

        # Detect just one note of the chord
        detected = [_detected(64, 1000.0)]
        results = matcher.process_detected_notes(detected, 1050.0)

        hits = [r for r in results if r.match_type == MatchType.HIT]
        assert len(hits) == 1
        # Both notes should be marked
        assert matcher.get_note_state(note_a) == MatchType.HIT
        assert matcher.get_note_state(note_b) == MatchType.HIT

    def test_non_simultaneous_notes_not_grouped(self):
        """Notes far apart in time should not be grouped as chord."""
        note_a = _note_event(1000.0, midi_note=64, string=1)
        note_b = _note_event(2000.0, midi_note=59, string=2)
        matcher = _make_matcher([note_a, note_b], chord_threshold_ms=50.0)

        detected = [_detected(64, 1000.0)]
        matcher.process_detected_notes(detected, 1050.0)

        assert matcher.get_note_state(note_a) == MatchType.HIT
        assert matcher.get_note_state(note_b) == MatchType.PENDING


class TestOnsetOnly:
    def test_non_onset_detections_filtered(self):
        """is_onset=False detections are filtered out."""
        tab_note = _note_event(1000.0, midi_note=64)
        matcher = _make_matcher([tab_note])

        detected = [_detected(64, 1000.0, is_onset=False)]
        results = matcher.process_detected_notes(detected, 1050.0)

        hit_close = [r for r in results if r.match_type in (MatchType.HIT, MatchType.CLOSE)]
        assert len(hit_close) == 0
        assert matcher.get_note_state(tab_note) == MatchType.PENDING


class TestTimingWindow:
    def test_detection_at_edge_of_window_matches(self):
        """Note detected at edge of window still matches."""
        tab_note = _note_event(1000.0, midi_note=64, duration_ms=500.0)
        matcher = _make_matcher([tab_note], timing_window_ms=100.0)

        # Detect at 1099ms — just within the 100ms window of a note at 1000ms
        detected = [_detected(64, 1099.0)]
        results = matcher.process_detected_notes(detected, 1099.0)

        hits = [r for r in results if r.match_type == MatchType.HIT]
        assert len(hits) == 1

    def test_detection_outside_window_no_match(self):
        """Detection outside window -> no match."""
        tab_note = _note_event(1000.0, midi_note=64, duration_ms=100.0)
        matcher = _make_matcher([tab_note], timing_window_ms=50.0)

        # Detect at 1200ms — the note ended at 1100ms, window is 50ms
        # get_active_notes_at_time checks [1200-50, 1200+50] = [1150, 1250]
        # note range is [1000, 1100] which doesn't overlap
        detected = [_detected(64, 1200.0)]
        results = matcher.process_detected_notes(detected, 1200.0)

        hits = [r for r in results if r.match_type == MatchType.HIT]
        assert len(hits) == 0


class TestStatistics:
    def test_mixed_results(self):
        notes = [
            _note_event(1000.0, midi_note=64, string=1),
            _note_event(2000.0, midi_note=59, string=2),
            _note_event(3000.0, midi_note=55, string=3),
        ]
        matcher = _make_matcher(notes, timing_window_ms=100.0)

        # Hit first note
        matcher.process_detected_notes([_detected(64, 1000.0)], 1050.0)
        # Close on second note
        matcher.process_detected_notes([_detected(60, 2000.0)], 2050.0)
        # Miss third note
        matcher.process_detected_notes([], 3200.0)

        stats = matcher.get_statistics()
        assert stats["hits"] == 1
        assert stats["close"] == 1
        assert stats["misses"] == 1
        assert stats["total"] == 3
        assert stats["accuracy_percent"] == pytest.approx(100 / 3)


class TestReset:
    def test_reset_clears_state(self):
        tab_note = _note_event(1000.0, midi_note=64)
        matcher = _make_matcher([tab_note])

        matcher.process_detected_notes([_detected(64, 1000.0)], 1050.0)
        assert matcher.get_note_state(tab_note) == MatchType.HIT

        matcher.reset()
        assert matcher.get_note_state(tab_note) == MatchType.PENDING
        stats = matcher.get_statistics()
        assert stats["hits"] == 0
        assert stats["close"] == 0
        assert stats["misses"] == 0


class TestNoDoubleMatch:
    def test_same_note_cannot_be_matched_twice(self):
        """Same tab note can't be matched twice."""
        tab_note = _note_event(1000.0, midi_note=64)
        matcher = _make_matcher([tab_note])

        # First detection -> HIT
        matcher.process_detected_notes([_detected(64, 1000.0)], 1050.0)
        assert matcher.hits == 1

        # Second detection of same pitch -> should not increment
        matcher.process_detected_notes([_detected(64, 1010.0)], 1060.0)
        assert matcher.hits == 1


class TestAudioOffset:
    def test_offset_shifts_detection_time(self):
        """audio_offset_ms shifts the detected timestamp to song time."""
        tab_note = _note_event(5000.0, midi_note=64, duration_ms=500.0)
        matcher = _make_matcher([tab_note], timing_window_ms=100.0,
                                audio_offset_ms=5000.0)

        # Detection at 0ms audio time + 5000ms offset = 5000ms song time
        detected = [_detected(64, 0.0)]
        results = matcher.process_detected_notes(detected, 5050.0)

        hits = [r for r in results if r.match_type == MatchType.HIT]
        assert len(hits) == 1

class TestTimingObservations:
    """Test the Timing Judge observation recording."""

    def _make_timing_matcher(self, notes, **kwargs):
        timeline = Timeline(notes, SongMetadata(title="Test", tempo=120))
        return NoteMatcher(
            timeline,
            timing_window_ms=kwargs.get("timing_window_ms", 100.0),
            audio_offset_ms=kwargs.get("audio_offset_ms", 0.0),
            chord_threshold_ms=kwargs.get("chord_threshold_ms", 50.0),
            timing_judge_enabled=True,
            pitch_strict=kwargs.get("pitch_strict", False),
        )

    def test_on_time_observation(self):
        """Detection within ±25ms produces ON_TIME."""
        tab = _note_event(1000.0, midi_note=64)
        matcher = self._make_timing_matcher([tab])
        matcher.process_detected_notes([_detected(64, 1010.0)], 1050.0)
        obs = matcher.get_timing_observations()
        assert len(obs) == 1
        assert obs[0].verdict == TimingVerdict.ON_TIME
        assert obs[0].timing_error_ms == pytest.approx(10.0)
        assert obs[0].pitch_verdict == PitchVerdict.CORRECT

    def test_early_observation(self):
        """Detection >25ms early produces EARLY."""
        tab = _note_event(1000.0, midi_note=64)
        matcher = self._make_timing_matcher([tab])
        matcher.process_detected_notes([_detected(64, 960.0)], 1050.0)
        obs = matcher.get_timing_observations()
        assert len(obs) == 1
        assert obs[0].verdict == TimingVerdict.EARLY
        assert obs[0].timing_error_ms == pytest.approx(-40.0)

    def test_late_observation(self):
        """Detection >25ms late produces LATE."""
        tab = _note_event(1000.0, midi_note=64)
        matcher = self._make_timing_matcher([tab])
        matcher.process_detected_notes([_detected(64, 1040.0)], 1050.0)
        obs = matcher.get_timing_observations()
        assert len(obs) == 1
        assert obs[0].verdict == TimingVerdict.LATE
        assert obs[0].timing_error_ms == pytest.approx(40.0)

    def test_missed_note_observation(self):
        """A note that passes the window unmatched produces a MISSED observation."""
        tab = _note_event(1000.0, midi_note=64)
        matcher = self._make_timing_matcher([tab])
        # Advance past the window (1000 + 100 = 1100 cutoff)
        matcher.process_detected_notes([], 1200.0)
        obs = matcher.get_timing_observations()
        assert len(obs) == 1
        assert obs[0].verdict == TimingVerdict.MISSED
        assert obs[0].expected_ms == 1000.0

    def test_extra_onset_observation(self):
        """An onset with no pending note produces EXTRA."""
        tab = _note_event(1000.0, midi_note=64)
        matcher = self._make_timing_matcher([tab])
        # First, match the tab note
        matcher.process_detected_notes([_detected(64, 1000.0)], 1050.0)
        # Now detect another onset — no pending notes left
        matcher.process_detected_notes([_detected(64, 1050.0)], 1050.0)
        obs = matcher.get_timing_observations()
        # First obs is the match, second is extra
        assert len(obs) == 2
        assert obs[1].verdict == TimingVerdict.EXTRA

    def test_pitch_strict_reclassifies_close_as_miss(self):
        """In pitch_strict mode, ±1 semitone is not CLOSE — it's ignored."""
        tab = _note_event(1000.0, midi_note=64)
        matcher = self._make_timing_matcher([tab], pitch_strict=True)
        matcher.process_detected_notes([_detected(65, 1000.0)], 1050.0)
        # Should not match (strict), but should record EXTRA with NEAR pitch
        obs = matcher.get_timing_observations()
        assert len(obs) == 1
        assert obs[0].pitch_verdict == PitchVerdict.NEAR
        # The note should remain PENDING (not matched in strict mode)
        assert matcher.get_note_state(tab) == MatchType.PENDING

    def test_no_observations_when_timing_judge_disabled(self):
        """Arcade mode (timing_judge_enabled=False) produces no observations."""
        tab = _note_event(1000.0, midi_note=64)
        matcher = _make_matcher([tab])
        matcher.process_detected_notes([_detected(64, 1000.0)], 1050.0)
        assert matcher.get_timing_observations() == []

    def test_reset_clears_observations(self):
        """reset() clears timing observations."""
        tab = _note_event(1000.0, midi_note=64)
        matcher = self._make_timing_matcher([tab])
        matcher.process_detected_notes([_detected(64, 1000.0)], 1050.0)
        assert len(matcher.get_timing_observations()) == 1
        matcher.reset()
        assert matcher.get_timing_observations() == []

    def test_get_timing_stats(self):
        """get_timing_stats returns computed stats."""
        tab = _note_event(1000.0, midi_note=64)
        matcher = self._make_timing_matcher([tab])
        matcher.process_detected_notes([_detected(64, 1000.0)], 1050.0)
        stats = matcher.get_timing_stats()
        assert stats.count == 1
        assert stats.on_time_count == 1