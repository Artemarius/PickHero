"""Tests for pickhero.matcher module."""

import numpy as np
import pytest

from pickhero.audio.detector import DetectedNote
from pickhero.audio.input import TimestampedNote
from pickhero.audio.performance import PerformanceEvent, TechniqueSpec, TechniqueCandidate
from pickhero.matcher import MatchType, MatchResult, NoteMatcher
from pickhero.tabs.timeline import NoteEvent, SongMetadata, Timeline
from pickhero.timing import PitchVerdict, TimingVerdict


def _note_event(timestamp_ms: float, midi_note: int = 64, string: int = 1,
                fret: int = 0, duration_ms: float = 500.0,
                techniques: tuple = ()) -> NoteEvent:
    return NoteEvent(
        timestamp_ms=timestamp_ms,
        duration_ms=duration_ms,
        midi_note=midi_note,
        string=string,
        fret=fret,
        techniques=techniques,
    )


def _detected(midi_note: int, timestamp_ms: float, is_onset: bool = True,
              confidence: float = 0.95,
              performance: "PerformanceEvent | None" = None) -> TimestampedNote:
    return TimestampedNote(
        note=DetectedNote(
            midi_note=midi_note,
            frequency=440.0,  # placeholder
            confidence=confidence,
            name="A4",  # placeholder
            is_onset=is_onset,
            performance=performance,
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
        """ARCADE mode: match one note of a simultaneous pair -> both marked HIT."""
        note_a = _note_event(1000.0, midi_note=64, string=1)
        note_b = _note_event(1000.0, midi_note=59, string=2)
        matcher = _make_matcher([note_a, note_b], chord_threshold_ms=50.0)
        # Explicit ARCADE mode (forgiving: chord auto-complete)
        matcher.match_mode = "arcade"

        # Detect just one note of the chord
        detected = [_detected(64, 1000.0)]
        results = matcher.process_detected_notes(detected, 1050.0)

        hits = [r for r in results if r.match_type == MatchType.HIT]
        assert len(hits) == 1
        # Both notes should be marked (ARCADE auto-complete)
        assert matcher.get_note_state(note_a) == MatchType.HIT
        assert matcher.get_note_state(note_b) == MatchType.HIT

    def test_judge_mode_no_chord_autocomplete(self):
        """JUDGE mode: matching the root of a 2-note chord must NOT auto-complete the fifth.

        Regression test for the auto-complete bug where needed = ceil(2/2) = 1
        let a single matched root auto-complete the fifth. In JUDGE mode only
        the matched note is marked; the fifth stays PENDING then MISS after window.
        """
        from pickhero.audio.match_mode import MatchMode
        note_a = _note_event(1000.0, midi_note=40, string=6)  # E2 (root)
        note_b = _note_event(1000.0, midi_note=47, string=5)  # B2 (fifth)
        matcher = _make_matcher([note_a, note_b], chord_threshold_ms=50.0)
        matcher.match_mode = MatchMode.JUDGE

        # Detect just the root
        detected = [_detected(40, 1000.0)]
        matcher.process_detected_notes(detected, 1050.0)

        # Root is HIT, fifth stays PENDING (no auto-complete)
        assert matcher.get_note_state(note_a) == MatchType.HIT
        assert matcher.get_note_state(note_b) == MatchType.PENDING

        # Advance past the window — fifth becomes MISS
        matcher.process_detected_notes([], 1200.0)
        assert matcher.get_note_state(note_b) == MatchType.MISS

    def test_non_simultaneous_notes_not_grouped(self):
        """Notes far apart in time should not be grouped as chord."""
        note_a = _note_event(1000.0, midi_note=64, string=1)
        note_b = _note_event(2000.0, midi_note=59, string=2)
        matcher = _make_matcher([note_a, note_b], chord_threshold_ms=50.0)

        detected = [_detected(64, 1000.0)]
        matcher.process_detected_notes(detected, 1050.0)

        assert matcher.get_note_state(note_a) == MatchType.HIT
        assert matcher.get_note_state(note_b) == MatchType.PENDING

    def test_fft_chord_verification_matches_missed_fifth(self):
        """FFT chord verification can match the fifth even if YIN missed it."""
        from pickhero.audio.chord_detector import ChordDetector

        root = _note_event(1000.0, midi_note=40, string=6)  # E2
        fifth = _note_event(1000.0, midi_note=47, string=5)  # B2
        matcher = _make_matcher([root, fifth], chord_threshold_ms=50.0)

        chord_detector = ChordDetector(sample_rate=48000)
        # Feed a synthetic E5 power chord into the chord detector.
        sr = 48000
        samples = int(sr * 0.5)
        t = np.arange(samples) / sr
        signal = np.zeros(samples)
        for midi in (40, 47):
            freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            for h in range(1, 6):
                signal += np.sin(2 * np.pi * freq * h * t) / h
        signal = (signal / np.max(np.abs(signal)) * 0.5).astype(np.float32)
        chord_detector.push_audio(signal)

        # YIN misses both notes. Chord verification should still pick up the chord.
        matcher.process_detected_notes([], 1050.0)
        assert matcher.get_note_state(root) == MatchType.PENDING
        assert matcher.get_note_state(fifth) == MatchType.PENDING

        results = matcher.verify_chord_at(1050.0, chord_detector)
        assert any(r.match_type in (MatchType.HIT, MatchType.CLOSE) for r in results)
        assert matcher.get_note_state(root) == MatchType.HIT
        assert matcher.get_note_state(fifth) == MatchType.HIT

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
        assert stats.on_time_count == 1

    def test_pitch_strict_octave_snapped(self):
        """In JUDGE mode, an exact octave (12 semitones) is NOT snapped to correct.

        The old tab-guided octave correction silently snapped a 12-semitone
        match to a HIT. In JUDGE mode this is removed: a 12-semitone match is
        recorded as EXTRA with PitchVerdict.WRONG, so octave errors surface
        instead of being forgiven silently.
        """
        tab = _note_event(1000.0, midi_note=64)  # E4
        matcher = self._make_timing_matcher([tab], pitch_strict=True)
        # Detect MIDI 76 (E5, exactly 12 semitones up)
        matcher.process_detected_notes([_detected(76, 1000.0)], 1050.0)
        obs = matcher.get_timing_observations()
        assert obs[0].pitch_verdict == PitchVerdict.WRONG
        assert obs[0].verdict == TimingVerdict.EXTRA
    def test_techniques_recorded_on_match(self):
        """Matched note records the tab's expected techniques on the observation."""
        tab = _note_event(1000.0, midi_note=64, techniques=(TechniqueSpec(kind="hammer_on", tied_to_previous=True),))
        matcher = self._make_timing_matcher([tab])
        matcher.process_detected_notes([_detected(64, 1000.0)], 1050.0)
        obs = matcher.get_timing_observations()
        assert len(obs) == 1
        assert obs[0].techniques is not None
        assert obs[0].techniques[0].kind == "hammer_on"

    def test_no_techniques_recorded_for_normal_note(self):
        """A normal note (no techniques) records techniques=() on the observation."""
        tab = _note_event(1000.0, midi_note=64)
        matcher = self._make_timing_matcher([tab])
        matcher.process_detected_notes([_detected(64, 1000.0)], 1050.0)
        obs = matcher.get_timing_observations()
        assert len(obs) == 1
        # techniques defaults to () on the NoteEvent, recorded as () on obs
        assert obs[0].techniques == ()

    def test_legato_direction_resolved_to_pull_off(self):
        """A hammer_on spec on a descending note resolves to pull_off."""
        prev = _note_event(500.0, midi_note=64, string=1, fret=0)
        cur = _note_event(1000.0, midi_note=62, string=1, fret=0,
                          techniques=(TechniqueSpec(kind="hammer_on", tied_to_previous=True),))
        matcher = self._make_timing_matcher([prev, cur])
        # Match the prev note first so it isn't marked missed
        matcher.process_detected_notes([_detected(64, 500.0)], 550.0)
        matcher.process_detected_notes([_detected(62, 1000.0)], 1050.0)
        # The matched note's techniques should now show pull_off
        # (resolution happens in process_detected_notes before recording)
        obs = matcher.get_timing_observations()
        # Second observation is the cur note (first was prev)
        assert len(obs) >= 2
        assert obs[-1].techniques[0].kind == "pull_off"

    def test_analyze_performance_populates_verdicts(self):
        """analyze_performance runs the analyzer and fills event.verdicts."""
        # Build a synthetic bend performance event
        times = np.linspace(0, 0.2, 18)
        cents = np.minimum(times / 0.2 * 100, 100)
        event = PerformanceEvent(
            onset_ms=0,
            f0_curve=[(t * 1000, 82.41 * 2 ** (c / 1200), c) for t, c in zip(times, cents)],
            energy_envelope=[(t * 1000, 0.5) for t in times],
            midi_note=40,
        )
        note = _note_event(0.0, midi_note=40, string=6, fret=0,
                           techniques=(TechniqueSpec(kind="bend", target_cents=100.0),))
        matcher = self._make_timing_matcher([note])
        # Simulate the matched pair being collected
        matcher._matched_pairs.append((event, note))
        events = matcher.analyze_performance()
        assert len(events) == 1
        assert len(events[0].verdicts) >= 1
        assert events[0].verdicts[0].kind == "bend"

    def test_technique_accuracy_from_verdicts(self):
        """get_statistics reports technique accuracy from analyzer verdicts."""
        # good bend
        times = np.linspace(0, 0.2, 18)
        cents = np.minimum(times / 0.2 * 100, 100)
        good_event = PerformanceEvent(
            onset_ms=0,
            f0_curve=[(t * 1000, 82.41 * 2 ** (c / 1200), c) for t, c in zip(times, cents)],
            energy_envelope=[(t * 1000, 0.5) for t in times],
            midi_note=40,
        )
        # weak bend (only reaches 30 cents of a 100-cent target)
        weak_cents = np.minimum(times / 0.2 * 30, 30)
        weak_event = PerformanceEvent(
            onset_ms=0,
            f0_curve=[(t * 1000, 82.41 * 2 ** (c / 1200), c) for t, c in zip(times, weak_cents)],
            energy_envelope=[(t * 1000, 0.5) for t in times],
            midi_note=40,
        )
        notes = [
            _note_event(0.0, midi_note=40, string=6, fret=0,
                        techniques=(TechniqueSpec(kind="bend", target_cents=100.0),)),
            _note_event(500.0, midi_note=40, string=6, fret=0,
                        techniques=(TechniqueSpec(kind="bend", target_cents=100.0),)),
        ]
        matcher = self._make_timing_matcher(notes)
        matcher._matched_pairs.append((good_event, notes[0]))
        matcher._matched_pairs.append((weak_event, notes[1]))
        matcher.analyze_performance()
        stats = matcher.get_statistics()
        assert stats["technique_total"] == 2
        # good bend counts as correct; weak bend (grade 'weak' or 'missed') does not
        assert stats["technique_correct"] == 1
        assert stats["technique_accuracy_percent"] == 50.0


class TestVerifierPrimary:
    """Tests that the expected-event verifier drives matching when available."""

    def test_verifier_overrides_semitone_distance(self):
        """When the audio window contains the expected note, verifier wins."""
        from pickhero.audio.match_mode import MatchMode
        from pickhero.audio.note_utils import midi_to_freq
        from pickhero.audio.verifier_composite import CompositeVerifier

        tab_note = _note_event(1000.0, midi_note=64)
        verifier = CompositeVerifier(sample_rate=48000)
        timeline = Timeline([tab_note], SongMetadata(title="Test", tempo=120))
        matcher = NoteMatcher(
            timeline,
            timing_window_ms=100.0,
            audio_offset_ms=0.0,
            mode=MatchMode.ARCADE,
            verifier=verifier,
        )

        # Build an audio window containing the *expected* note (MIDI 64).
        freq = midi_to_freq(64)
        samples = int(48000 * 200 / 1000)
        window = (0.5 * np.sin(2 * np.pi * freq * np.arange(samples) / 48000)).astype(np.float32)

        # Pass a "detected" note that is far off (MIDI 60) plus the window.
        detected = [_detected(60, 1000.0)]
        results = matcher.process_detected_notes(
            detected, 1050.0, audio_window=window
        )
        hits = [r for r in results if r.match_type == MatchType.HIT]
        assert len(hits) == 1, "verifier should match expected note via audio evidence"
        assert tab_note in hits[0].matched_events

    def test_judge_mode_rejects_wrong_note_with_verifier(self):
        """In JUDGE mode, verifier must reject a non-present expected note."""
        from pickhero.audio.match_mode import MatchMode
        from pickhero.audio.verifier_composite import CompositeVerifier

        tab_note = _note_event(1000.0, midi_note=64)
        verifier = CompositeVerifier(sample_rate=48000)
        timeline = Timeline([tab_note], SongMetadata(title="Test", tempo=120))
        matcher = NoteMatcher(
            timeline,
            timing_window_ms=100.0,
            audio_offset_ms=0.0,
            mode=MatchMode.JUDGE,
            verifier=verifier,
        )

        # Silence should not verify as the expected note.
        window = np.zeros(int(48000 * 200 / 1000), dtype=np.float32)
        detected = [_detected(64, 1000.0)]
        results = matcher.process_detected_notes(
            detected, 1050.0, audio_window=window
        )
        hits = [r for r in results if r.match_type == MatchType.HIT]
        assert len(hits) == 0, "silence should not verify in JUDGE mode"


class TestVerifyHitZone:
    """Integration tests for verify_hit_zone()."""

    def _sine_window(self, midi: int, duration_ms: float = 200.0,
                     sample_rate: int = 48000, amplitude: float = 0.5) -> np.ndarray:
        from pickhero.audio.note_utils import midi_to_freq
        freq = midi_to_freq(midi)
        samples = int(sample_rate * duration_ms / 1000.0)
        t = np.arange(samples) / sample_rate
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def _ramped_sine_window(self, midi: int, duration_ms: float = 200.0,
                            sample_rate: int = 48000, amplitude: float = 0.5,
                            attack_ms: float = 100.0) -> np.ndarray:
        from pickhero.audio.note_utils import midi_to_freq
        freq = midi_to_freq(midi)
        samples = int(sample_rate * duration_ms / 1000.0)
        t = np.arange(samples) / sample_rate
        attack_samples = int(sample_rate * attack_ms / 1000.0)
        ramp_samples = int(sample_rate * 5 / 1000.0)
        envelope = np.zeros(samples, dtype=np.float32)
        if attack_samples < samples:
            envelope[attack_samples:] = 1.0
        if attack_samples + ramp_samples <= samples:
            envelope[attack_samples:attack_samples + ramp_samples] = np.linspace(
                0.0, 1.0, ramp_samples
            )
        else:
            envelope[attack_samples:] = np.linspace(0.0, 1.0, samples - attack_samples)
        signal = amplitude * np.sin(2 * np.pi * freq * t) * envelope
        return signal.astype(np.float32)

    def _harmonic_rich_window(self, midi: int, duration_ms: float = 200.0,
                              sample_rate: int = 48000, amplitude: float = 0.5,
                              attack_ms: float = 100.0) -> np.ndarray:
        """Harmonic-rich signal that starts at ``attack_ms``."""
        from pickhero.audio.note_utils import midi_to_freq
        freq = midi_to_freq(midi)
        samples = int(sample_rate * duration_ms / 1000.0)
        t = np.arange(samples) / sample_rate
        attack_samples = int(sample_rate * attack_ms / 1000.0)
        ramp_samples = int(sample_rate * 5 / 1000.0)
        envelope = np.zeros(samples, dtype=np.float32)
        if attack_samples < samples:
            envelope[attack_samples:] = 1.0
        if attack_samples + ramp_samples <= samples:
            envelope[attack_samples:attack_samples + ramp_samples] = np.linspace(
                0.0, 1.0, ramp_samples
            )
        else:
            envelope[attack_samples:] = np.linspace(0.0, 1.0, samples - attack_samples)
        signal = np.zeros(samples, dtype=np.float32)
        for h, weight in enumerate([1.0, 0.5, 0.25, 0.125, 0.0625], start=1):
            signal += weight * np.sin(2 * np.pi * freq * h * t)
        signal = amplitude * signal * envelope
        return signal.astype(np.float32)

    def test_verify_hit_zone_matches_without_detected_notes(self):
        """verify_hit_zone should match pending notes from audio alone."""
        from pickhero.audio.verifier_composite import CompositeVerifier
        from pickhero.tabs.timeline import SongMetadata
        notes = [_note_event(1000.0, midi_note=64, string=1)]
        timeline = Timeline(notes, SongMetadata(title="Test", tempo=120))
        verifier = CompositeVerifier(sample_rate=48000)
        matcher = NoteMatcher(timeline, timing_window_ms=100.0, verifier=verifier)
        # Window covers [850, 1050] ms; note at 1000ms has expected onset at 150ms.
        window = self._ramped_sine_window(64, duration_ms=200.0, attack_ms=150.0)
        results = matcher.verify_hit_zone(1050.0, window, window_start_ms=850.0)
        assert any(r.match_type == MatchType.HIT for r in results)

    def test_verify_hit_zone_rejects_onset_outside_tolerance(self):
        """An attack outside the expected onset window should not score."""
        from pickhero.audio.verifier_composite import CompositeVerifier
        from pickhero.tabs.timeline import SongMetadata
        notes = [_note_event(1000.0, midi_note=64, string=1)]
        timeline = Timeline(notes, SongMetadata(title="Test", tempo=120))
        verifier = CompositeVerifier(sample_rate=48000)
        matcher = NoteMatcher(timeline, timing_window_ms=100.0, verifier=verifier)
        # Attack at 150ms, but expected onset offset claims 50ms.
        window = self._ramped_sine_window(64, duration_ms=200.0, attack_ms=150.0)
        results = matcher.verify_hit_zone(1050.0, window, window_start_ms=975.0)
        assert all(r.match_type != MatchType.HIT for r in results)

    def test_verify_hit_zone_records_timing_observation(self):
        """A verifier-driven HIT should produce a TimingObservation."""
        from pickhero.audio.match_mode import MatchMode
        from pickhero.audio.verifier_composite import CompositeVerifier
        from pickhero.tabs.timeline import SongMetadata
        notes = [_note_event(1000.0, midi_note=64, string=1)]
        timeline = Timeline(notes, SongMetadata(title="Test", tempo=120))
        verifier = CompositeVerifier(sample_rate=48000)
        matcher = NoteMatcher(
            timeline, timing_window_ms=100.0, verifier=verifier, mode=MatchMode.JUDGE
        )
        # Harmonic-rich attack at 150ms; expected at 150ms -> detected_ms = 1000ms.
        window = self._harmonic_rich_window(64, duration_ms=200.0, attack_ms=150.0)
        results = matcher.verify_hit_zone(1050.0, window, window_start_ms=850.0)
        assert any(r.match_type == MatchType.HIT for r in results)
        obs = matcher.get_timing_observations()
        assert len(obs) == 1
        assert obs[0].expected_ms == 1000.0
        assert obs[0].timing_error_ms == pytest.approx(0.0, abs=20.0)

    def test_verify_hit_zone_dead_note_accepted(self):
        """A dead note is accepted without pitch detection."""
        from pickhero.audio.performance import TechniqueSpec
        from pickhero.audio.verifier_composite import CompositeVerifier
        from pickhero.tabs.timeline import SongMetadata
        notes = [_note_event(
            1000.0, midi_note=64, string=1,
            techniques=(TechniqueSpec(kind="dead_note"),)
        )]
        timeline = Timeline(notes, SongMetadata(title="Test", tempo=120))
        verifier = CompositeVerifier(sample_rate=48000)
        matcher = NoteMatcher(timeline, timing_window_ms=100.0, verifier=verifier)
        # Broadband percussive burst with fast decay in the middle of the window.
        sr = 48000
        samples = int(sr * 200 / 1000)
        t = np.arange(samples) / sr
        window = (np.sin(2 * np.pi * 100 * t) * np.exp(-t / 0.02)).astype(np.float32)
        results = matcher.verify_hit_zone(1050.0, window, window_start_ms=850.0)
        assert any(r.match_type == MatchType.HIT for r in results)

    def test_verify_hit_zone_returns_empty_without_verifier(self):
        """Without a verifier, verify_hit_zone returns []."""
        from pickhero.tabs.timeline import SongMetadata
        notes = [_note_event(1000.0, midi_note=64, string=1)]
        timeline = Timeline(notes, SongMetadata(title="Test", tempo=120))
        matcher = NoteMatcher(timeline, timing_window_ms=100.0, verifier=None)
        window = self._sine_window(64, duration_ms=200.0)
        results = matcher.verify_hit_zone(1050.0, window)
        assert results == []

    def test_verify_hit_zone_returns_empty_for_silence(self):
        """Silence should not match any notes via verify_hit_zone."""
        from pickhero.audio.verifier_composite import CompositeVerifier
        from pickhero.tabs.timeline import SongMetadata
        notes = [_note_event(1000.0, midi_note=64, string=1)]
        timeline = Timeline(notes, SongMetadata(title="Test", tempo=120))
        verifier = CompositeVerifier(sample_rate=48000)
        matcher = NoteMatcher(timeline, timing_window_ms=100.0, verifier=verifier)
        window = np.zeros(4800, dtype=np.float32)
        results = matcher.verify_hit_zone(1050.0, window)
        assert all(r.match_type != MatchType.HIT for r in results)


class TestStateMachine:
    """Tests for the unified event state machine (advance_state_machine)."""

    def _sine_window(self, midi: int, duration_ms: float = 200.0,
                     sample_rate: int = 48000, amplitude: float = 0.5) -> np.ndarray:
        from pickhero.audio.note_utils import midi_to_freq
        freq = midi_to_freq(midi)
        samples = int(sample_rate * duration_ms / 1000.0)
        t = np.arange(samples) / sample_rate
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def _ramped_sine_window(self, midi: int, duration_ms: float = 200.0,
                            sample_rate: int = 48000, amplitude: float = 0.5,
                            attack_ms: float = 100.0) -> np.ndarray:
        from pickhero.audio.note_utils import midi_to_freq
        freq = midi_to_freq(midi)
        samples = int(sample_rate * duration_ms / 1000.0)
        t = np.arange(samples) / sample_rate
        attack_samples = int(sample_rate * attack_ms / 1000.0)
        ramp_samples = int(sample_rate * 5 / 1000.0)
        envelope = np.zeros(samples, dtype=np.float32)
        if attack_samples < samples:
            envelope[attack_samples:] = 1.0
        if attack_samples + ramp_samples <= samples:
            envelope[attack_samples:attack_samples + ramp_samples] = np.linspace(
                0.0, 1.0, ramp_samples
            )
        else:
            envelope[attack_samples:] = np.linspace(0.0, 1.0, samples - attack_samples)
        signal = amplitude * np.sin(2 * np.pi * freq * t) * envelope
        return signal.astype(np.float32)

    def _make_matcher_with_verifier(self, notes, timing_window_ms=100.0):
        from pickhero.audio.verifier_composite import CompositeVerifier
        from pickhero.tabs.timeline import SongMetadata
        timeline = Timeline(notes, SongMetadata(title="Test", tempo=120))
        verifier = CompositeVerifier(sample_rate=48000)
        return NoteMatcher(
            timeline,
            timing_window_ms=timing_window_ms,
            verifier=verifier,
        )

    def test_initial_state_is_pending(self):
        """Event state defaults to PENDING before any evidence."""
        from pickhero.audio.event_state import EventState
        notes = [_note_event(1000.0, midi_note=64, string=1)]
        matcher = self._make_matcher_with_verifier(notes)
        state = matcher._get_event_state((1000.0, 1))
        assert state == EventState.PENDING

    def test_returns_empty_without_verifier(self):
        """Without a verifier, advance_state_machine returns []."""
        notes = [_note_event(1000.0, midi_note=64, string=1)]
        matcher = _make_matcher(notes, timing_window_ms=100.0)
        window = self._sine_window(64)
        results = matcher.advance_state_machine(
            playback_ms=1050.0,
            audio_window=window,
            detected_notes=[],
        )
        assert results == []

    def test_returns_empty_for_silence(self):
        """Silence should not produce any terminal results."""
        notes = [_note_event(1000.0, midi_note=64, string=1)]
        matcher = self._make_matcher_with_verifier(notes)
        window = np.zeros(4800, dtype=np.float32)
        results = matcher.advance_state_machine(
            playback_ms=1050.0,
            audio_window=window,
            detected_notes=[],
        )
        assert all(r.match_type != MatchType.HIT for r in results)

    def test_pending_to_hit_with_onset_and_pitch(self):
        """Normal picked note: onset + pitch → PITCHED → HIT.

        Feed a frame with matching onset+pitch, then advance past duration
        to trigger RELEASED → HIT.
        """
        from pickhero.audio.event_state import EventState
        notes = [_note_event(1000.0, midi_note=64, string=1, duration_ms=100.0)]
        matcher = self._make_matcher_with_verifier(notes, timing_window_ms=100.0)
        # Frame 1: onset + matching pitch at the note's timestamp.
        window = self._ramped_sine_window(64, duration_ms=200.0, attack_ms=100.0)
        detected = [_detected(64, 1000.0, is_onset=True)]
        matcher.advance_state_machine(
            playback_ms=1100.0,
            audio_window=window,
            detected_notes=detected,
        )
        # Should have transitioned to PITCHED (onset + pitch match).
        state = matcher._get_event_state((1000.0, 1))
        assert state in (EventState.PITCHED, EventState.RELEASED, EventState.HIT), (
            f"Expected PITCHED or beyond, got {state}"
        )

    def test_miss_on_timing_expiry(self):
        """Note that expires without any onset → MISS."""
        from pickhero.audio.event_state import EventState
        notes = [_note_event(1000.0, midi_note=64, string=1, duration_ms=100.0)]
        matcher = self._make_matcher_with_verifier(notes, timing_window_ms=100.0)
        # Frame 1: note is in the timing window but no onset/pitch detected.
        # This enters the event into _event_states as PENDING.
        window = np.zeros(4800, dtype=np.float32)
        matcher.advance_state_machine(
            playback_ms=1050.0,  # judge_ms=950, range=[850,1050] — note at 1000 is in range
            audio_window=window,
            detected_notes=[],
        )
        assert matcher._get_event_state((1000.0, 1)) == EventState.PENDING
        # Frame 2: advance well past the timing window with silence.
        results = matcher.advance_state_machine(
            playback_ms=1300.0,  # 300ms past note — exceeds 2× timing window
            audio_window=window,
            detected_notes=[],
        )
        # The miss-expire path should have fired for the PENDING event.
        assert any(r.match_type == MatchType.MISS for r in results)
        state = matcher._get_event_state((1000.0, 1))
        assert state == EventState.MISS

    def test_technique_does_not_veto_hit(self):
        """A note with a technique spec still reaches HIT when pitch+onset match.

        Technique verdicts are never examined for transition decisions.
        """
        from pickhero.audio.event_state import EventState
        from pickhero.audio.performance import TechniqueSpec
        # Note with vibrato technique — should not prevent HIT.
        notes = [_note_event(
            1000.0, midi_note=64, string=1, duration_ms=100.0,
            techniques=(TechniqueSpec(kind="vibrato"),),
        )]
        matcher = self._make_matcher_with_verifier(notes, timing_window_ms=100.0)
        window = self._ramped_sine_window(64, duration_ms=200.0, attack_ms=100.0)
        detected = [_detected(64, 1000.0, is_onset=True)]
        matcher.advance_state_machine(
            playback_ms=1100.0,
            audio_window=window,
            detected_notes=detected,
        )
        state = matcher._get_event_state((1000.0, 1))
        # Should have progressed past PENDING despite technique spec.
        assert state != EventState.PENDING, (
            f"Technique spec should not block transition; state={state}"
        )

    def test_terminal_state_not_re_processed(self):
        """Once an event reaches a terminal state, it's not re-evaluated."""
        from pickhero.audio.event_state import EventState
        notes = [_note_event(1000.0, midi_note=64, string=1, duration_ms=100.0)]
        matcher = self._make_matcher_with_verifier(notes, timing_window_ms=100.0)
        # Force terminal state.
        matcher._event_states[(1000.0, 1)] = EventState.HIT
        matcher.hits = 1
        window = self._sine_window(64)
        detected = [_detected(64, 1000.0, is_onset=True)]
        results = matcher.advance_state_machine(
            playback_ms=1100.0,
            audio_window=window,
            detected_notes=detected,
        )
        # Should not produce new results for already-terminal event.
        hit_results = [r for r in results if r.match_type == MatchType.HIT
                       and any(e.timestamp_ms == 1000.0 for e in r.matched_events)]
        assert len(hit_results) == 0

    def test_reset_clears_event_states(self):
        """reset() clears the _event_states dict."""
        from pickhero.audio.event_state import EventState
        notes = [_note_event(1000.0, midi_note=64, string=1)]
        matcher = self._make_matcher_with_verifier(notes)
        matcher._event_states[(1000.0, 1)] = EventState.PITCHED
        matcher.reset()
        assert len(matcher._event_states) == 0
        assert matcher._get_event_state((1000.0, 1)) == EventState.PENDING

    def test_transition_ignores_technique(self):
        """_transition only checks pitch, onset, timing — never technique."""
        from pickhero.audio.event_state import EventState
        from pickhero.audio.performance import TechniqueSpec
        # Note with bend technique — _transition should not look at it.
        note = _note_event(
            1000.0, midi_note=64, string=1, duration_ms=500.0,
            techniques=(TechniqueSpec(kind="bend", target_cents=200),),
        )
        matcher = self._make_matcher_with_verifier([note])
        window = self._sine_window(64)
        # PENDING + onset + pitch match → PITCHED (technique irrelevant).
        new_state = matcher._transition(
            note, EventState.PENDING, 1050.0,
            detected_midis={64}, has_onset=True, audio_window=window,
        )
        assert new_state == EventState.PITCHED

        # PENDING + onset, no pitch → ATTACKING (technique irrelevant).
        new_state = matcher._transition(
            note, EventState.PENDING, 1050.0,
            detected_midis=set(), has_onset=True, audio_window=window,
        )
        assert new_state == EventState.ATTACKING

        # PENDING + no onset, timing expired → MISS (technique irrelevant).
        new_state = matcher._transition(
            note, EventState.PENDING, 1300.0,
            detected_midis=set(), has_onset=False, audio_window=window,
        )
        assert new_state == EventState.MISS

    def test_chord_fft_catches_voice_missed_by_yin(self):
        """Chord detector feeds evidence for a voice YIN missed.

        Two-note chord (C3=48, E3=52). YIN only detects one note.
        Chord_detector FFT recognizes both. advance_state_machine
        should transition both notes to HIT over two frames.
        """
        from pickhero.audio.event_state import EventState
        from pickhero.audio.chord_detector import ChordDetector
        from pickhero.audio.note_utils import midi_to_freq
        import numpy as np

        chord_notes = [
            _note_event(1000.0, midi_note=48, string=1, duration_ms=300.0),
            _note_event(1000.0, midi_note=52, string=2, duration_ms=300.0),
        ]
        matcher = self._make_matcher_with_verifier(chord_notes, timing_window_ms=200.0)

        # Create chord detector with synthetic audio containing both notes
        chord_detector = ChordDetector(sample_rate=48000, fft_size=8192)
        sample_rate = 48000
        samples = int(sample_rate * 300.0 / 1000.0)
        t = np.arange(samples) / sample_rate
        audio = (0.3 * np.sin(2 * np.pi * midi_to_freq(48) * t) +
                 0.3 * np.sin(2 * np.pi * midi_to_freq(52) * t)).astype(np.float32)
        chord_detector.push_audio(audio)

        # YIN only detects C3; E3 is missed by YIN
        detected = [_detected(48, 1000.0, is_onset=True)]

        # Frame 1: onset + pitch → PITCHED
        matcher.advance_state_machine(
            playback_ms=1150.0,
            audio_window=audio,
            detected_notes=detected,
            chord_detector=chord_detector,
        )
        state_c3 = matcher._get_event_state((1000.0, 1))
        state_e3 = matcher._get_event_state((1000.0, 2))
        assert state_c3 == EventState.PITCHED, f"C3 expected PITCHED after frame 1, got {state_c3}"
        assert state_e3 == EventState.PITCHED, (
            f"E3 expected PITCHED (FFT fallback) after frame 1, got {state_e3}"
        )

        # Frame 2: duration expired (1000+300=1300 < 1400) → RELEASED → HIT
        results = matcher.advance_state_machine(
            playback_ms=1400.0,
            audio_window=audio,
            detected_notes=detected,
            chord_detector=chord_detector,
        )
        assert matcher._get_event_state((1000.0, 1)) == EventState.HIT
        assert matcher._get_event_state((1000.0, 2)) == EventState.HIT
        hit_results = [r for r in results if r.match_type == MatchType.HIT]
        assert len(hit_results) >= 1, "Expected at least one HIT result"

    def test_chord_fft_without_detector_no_fallback(self):
        """Without chord_detector, chord group uses YIN detected_midis only."""
        from pickhero.audio.event_state import EventState

        chord_notes = [
            _note_event(1000.0, midi_note=48, string=1, duration_ms=300.0),
            _note_event(1000.0, midi_note=52, string=2, duration_ms=300.0),
        ]
        matcher = self._make_matcher_with_verifier(chord_notes, timing_window_ms=200.0)

        # Only C3 detected by YIN
        detected = [_detected(48, 1000.0, is_onset=True)]
        audio = np.zeros(int(48000 * 0.3), dtype=np.float32)

        # Frame 1: C3 → PITCHED, E3 stays PENDING (no chord detector)
        # Frame 1: C3 → PITCHED (YIN detected). E3 → ATTACKING (shared onset
        # from the chord strum, but no pitch evidence since YIN missed it).
        matcher.advance_state_machine(
            playback_ms=1150.0,
            audio_window=audio,
            detected_notes=detected,
        )
        assert matcher._get_event_state((1000.0, 1)) == EventState.PITCHED, "C3 should be PITCHED"
        assert matcher._get_event_state((1000.0, 2)) == EventState.ATTACKING, (
            "E3 should be ATTACKING (shared strum onset, no pitch yet)"
        )

        # Frame 2: playback at 1400ms keeps C3 in candidate window
        # (judge=1400-200=1200, range=[1000,1400]).
        # C3 duration 1000+300=1300 < 1400 → PITCHED→HIT.
        # E3 ATTACKING: 1400-1000=400 < 3×200=600 → not expired yet, stays ATTACKING.
        results = matcher.advance_state_machine(
            playback_ms=1400.0,
            audio_window=audio,
            detected_notes=detected,
        )
        assert matcher._get_event_state((1000.0, 1)) == EventState.HIT, "C3 should be HIT"
        assert matcher._get_event_state((1000.0, 2)) == EventState.ATTACKING, (
            "E3 still ATTACKING (3× window = 600 > 400 elapsed)"
        )
        # Frame 3: both notes outside candidate window. Safety net fires:
        # E3 ATTACKING: 1700-1000=700 > 3×200=600 → MISS via safety net
        # C3 PITCHED: 1700 < 1000+4×200=1800 → stays PITCHED (not yet expired by safety net)
        results = matcher.advance_state_machine(
            playback_ms=1700.0,
            audio_window=audio,
            detected_notes=detected,
        )
        assert matcher._get_event_state((1000.0, 1)) == EventState.HIT, "C3 still HIT from frame 2"
        assert matcher._get_event_state((1000.0, 2)) == EventState.MISS, (
            "E3 should be MISS (ATTACKING expired at 3×200=600ms)"
        )
        miss_results = [r for r in results if r.match_type == MatchType.MISS]
        assert len(miss_results) >= 1, "Expected at least one MISS from E3 expiry"