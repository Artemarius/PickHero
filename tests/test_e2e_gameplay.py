"""End-to-end gameplay loop tests.

These tests exercise the full gameplay pipeline:
  AudioCapture → NoteMatcher → FeedbackRenderer

They verify that matches actually flow through to visual feedback when
the gameplay loop processes detected notes. They don't require audio
hardware or a running PyGame display.

The key scenario being tested: when a player plays the correct notes,
matches must appear in the FeedbackRenderer so they're visible on screen.
A previous bug caused the onset-gating cache in ChordDetector to return
stale results for different chords with the same note count, causing
chord verification to silently fail.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pickhero.audio.chord_detector import ChordDetector
from pickhero.audio.detector import DetectedNote
from pickhero.audio.input import TimestampedNote
from pickhero.matcher import MatchType, NoteMatcher
from pickhero.tabs.loader import load_gp_file
from pickhero.tabs.timeline import NoteEvent, SongMetadata, Timeline
from pickhero.ui.feedback import FeedbackRenderer


FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detected(midi_note: int, timestamp_ms: float, is_onset: bool = True) -> TimestampedNote:
    """Build a synthetic detected note."""
    return TimestampedNote(
        note=DetectedNote(
            midi_note=midi_note,
            frequency=440.0 * (2 ** ((midi_note - 69) / 12)),
            confidence=0.95,
            name="N",
            is_onset=is_onset,
        ),
        timestamp_ms=timestamp_ms,
    )


def _timeline_with_chord() -> Timeline:
    """Build a minimal timeline with a 2-note chord at t=1000ms."""
    notes = [
        NoteEvent(timestamp_ms=1000.0, midi_note=40, string=6, fret=0, duration_ms=500.0),
        NoteEvent(timestamp_ms=1000.0, midi_note=47, string=5, fret=2, duration_ms=500.0),
    ]
    meta = SongMetadata(title="Test", artist="Tester", tempo=120, num_strings=6)
    return Timeline(notes, meta)


def _timeline_with_major_chord() -> Timeline:
    """Build a minimal timeline with a 3-note major chord at t=1000ms.

    E major: E2 (open low E), B2 (A string 2nd fret), E3 (D string 2nd fret).
    In partial-credit mode, matching 1 of 3 does NOT auto-complete
    (1 < ceil(3/2) = 2), leaving 2 notes PENDING for verify_chord_at.
    """
    notes = [
        NoteEvent(timestamp_ms=1000.0, midi_note=40, string=6, fret=0, duration_ms=500.0),
        NoteEvent(timestamp_ms=1000.0, midi_note=47, string=5, fret=2, duration_ms=500.0),
        NoteEvent(timestamp_ms=1000.0, midi_note=52, string=4, fret=2, duration_ms=500.0),
    ]
    meta = SongMetadata(title="Test", artist="Tester", tempo=120, num_strings=6)
    return Timeline(notes, meta)

def _timeline_with_two_chords() -> Timeline:
    """Build a timeline with two different chords at different times."""
    notes = [
        # Chord 1 at t=1000ms: E2 + B2 (power chord)
        NoteEvent(timestamp_ms=1000.0, midi_note=40, string=6, fret=0, duration_ms=500.0),
        NoteEvent(timestamp_ms=1000.0, midi_note=47, string=5, fret=2, duration_ms=500.0),
        # Chord 2 at t=2000ms: A2 + E3 (A power chord)
        NoteEvent(timestamp_ms=2000.0, midi_note=45, string=6, fret=0, duration_ms=500.0),
        NoteEvent(timestamp_ms=2000.0, midi_note=52, string=5, fret=2, duration_ms=500.0),
    ]
    meta = SongMetadata(title="Test", artist="Tester", tempo=120, num_strings=6)
    return Timeline(notes, meta)


def _timeline_with_single_notes() -> Timeline:
    """Build a timeline with sequential single notes."""
    notes = [
        NoteEvent(timestamp_ms=1000.0, midi_note=40, string=6, fret=0, duration_ms=500.0),
        NoteEvent(timestamp_ms=2000.0, midi_note=42, string=6, fret=2, duration_ms=500.0),
        NoteEvent(timestamp_ms=3000.0, midi_note=44, string=6, fret=4, duration_ms=500.0),
    ]
    meta = SongMetadata(title="Test", artist="Tester", tempo=120, num_strings=6)
    return Timeline(notes, meta)


def _push_chord_signal(detector: ChordDetector, midi_notes: list[int], sr: int = 48000) -> None:
    """Push a synthetic chord signal into the ChordDetector."""
    from pickhero.audio.note_utils import midi_to_freq

    samples = int(sr * 0.5)
    t = np.arange(samples) / sr
    signal = np.zeros(samples, dtype=np.float32)
    for midi in midi_notes:
        freq = midi_to_freq(midi)
        for h in range(1, 9):
            signal += np.sin(2 * np.pi * freq * h * t) / h
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.5
    # Push in chunks like the real audio callback
    hop = 512
    for i in range(0, len(signal), hop):
        detector.push_audio(signal[i:i + hop])


# ===================================================================
# Single-note matching → feedback
# ===================================================================

class TestSingleNoteMatchFlow:
    """Verify that single-note matches flow through to FeedbackRenderer."""

    def test_correct_note_produces_hit_feedback(self):
        """Playing the correct note at the right time → HIT in feedback."""
        timeline = _timeline_with_single_notes()
        matcher = NoteMatcher(timeline, timing_window_ms=100.0)
        feedback = FeedbackRenderer()

        detected = [_detected(40, 1000.0)]
        results = matcher.process_detected_notes(detected, 1050.0)
        feedback.add_results(results, 1050.0)

        assert matcher.hits >= 1, f"Expected at least 1 hit, got hits={matcher.hits}"
        assert feedback.streak >= 1, f"Expected streak >= 1, got {feedback.streak}"

    def test_wrong_note_does_not_produce_hit(self):
        """Playing the wrong note at the right time → no HIT."""
        timeline = _timeline_with_single_notes()
        matcher = NoteMatcher(timeline, timing_window_ms=100.0)
        feedback = FeedbackRenderer()

        # Play G#2 (MIDI 44) instead of E2 (MIDI 40)
        detected = [_detected(44, 1000.0)]
        results = matcher.process_detected_notes(detected, 1050.0)
        feedback.add_results(results, 1050.0)

        assert matcher.hits == 0, f"Should have 0 hits, got {matcher.hits}"
        assert feedback.streak == 0, f"Streak should be 0, got {feedback.streak}"

    def test_note_color_reflects_match_state(self):
        """FeedbackRenderer.get_note_color returns hit color after a match."""
        timeline = _timeline_with_single_notes()
        matcher = NoteMatcher(timeline, timing_window_ms=100.0)
        feedback = FeedbackRenderer()

        note = timeline.notes[0]  # E2 at 1000ms
        base_color = (100, 100, 100)

        # Before matching: base color
        color = feedback.get_note_color(note, base_color, 500.0, is_past=False)
        assert color == base_color, "Before match, should return base color"

        # After matching: hit color
        detected = [_detected(40, 1000.0)]
        results = matcher.process_detected_notes(detected, 1050.0)
        feedback.add_results(results, 1050.0)

        color = feedback.get_note_color(note, base_color, 1050.0, is_past=False)
        assert color != base_color, "After hit, color should change"

    def test_missed_note_shows_miss_feedback(self):
        """A note that passes the timing window without being played → MISS."""
        timeline = _timeline_with_single_notes()
        matcher = NoteMatcher(timeline, timing_window_ms=100.0)
        feedback = FeedbackRenderer()

        # Advance playback past the first note's window (1000 + 100 = 1100ms)
        results = matcher.process_detected_notes([], 1200.0)
        feedback.add_results(results, 1200.0)

        assert matcher.misses >= 1, f"Expected at least 1 miss, got misses={matcher.misses}"
        assert feedback.streak == 0


# ===================================================================
# Chord matching → feedback (the critical path)
# ===================================================================

class TestChordMatchFlow:
    """Verify that chord matches flow through to FeedbackRenderer.

    This is the path that was broken: the onset-gating cache in
    ChordDetector returned stale results for different chords with the
    same note count, causing chord verification to silently fail.
    """

    def test_chord_hit_via_yin_single_note(self):
        """In easy mode, YIN matching one chord note marks ALL siblings.

        In partial-credit mode, a 2-note chord auto-completes when the
        majority (1 of 2) is matched. To test the chord-verification path
        separately, use easy mode (chord_partial_credit=False) where
        only the matched note is marked and the rest stay PENDING.
        """
        timeline = _timeline_with_chord()
        # easy mode: mark ALL siblings when any is matched
        matcher = NoteMatcher(timeline, timing_window_ms=100.0, chord_partial_credit=False)
        feedback = FeedbackRenderer()

        # Play just the root (E2) — YIN detects it
        detected = [_detected(40, 1000.0)]
        results = matcher.process_detected_notes(detected, 1050.0)
        feedback.add_results(results, 1050.0)

        # Both notes should be hit (easy mode marks all siblings)
        assert matcher.hits >= 2, f"Both chord notes should be hit in easy mode, got hits={matcher.hits}"

    def test_chord_verification_marks_remaining_notes(self):
        """After YIN matches root, verify_chord_at should mark remaining notes via FFT.

        Uses a 3-note chord with partial-credit mode: matching 1 of 3 does
        NOT auto-complete (1 < ceil(3/2) = 2), so 2 notes stay PENDING for
        verify_chord_at to handle.
        """
        timeline = _timeline_with_major_chord()
        matcher = NoteMatcher(timeline, timing_window_ms=100.0, chord_partial_credit=True)
        feedback = FeedbackRenderer()
        chord_detector = ChordDetector(sample_rate=48000)

        # Push chord audio (E2 + B2 + E3) into the detector
        _push_chord_signal(chord_detector, [40, 47, 52])

        # Step 1: YIN matches root (E2) only
        detected = [_detected(40, 1000.0)]
        yin_results = matcher.process_detected_notes(detected, 1050.0)
        feedback.add_results(yin_results, 1050.0)

        # 1 of 3 matched — not enough for auto-complete (need 2)
        assert matcher.hits == 1, f"Only root should be hit after YIN, got {matcher.hits}"

        # Step 2: FFT verifies the full chord — should mark remaining notes
        fft_results = matcher.verify_chord_at(1050.0, chord_detector, has_onset=True)
        feedback.add_results(fft_results, 1050.0)

        assert matcher.hits >= 2, f"At least 2 notes should be hit after FFT, got {matcher.hits}"

    def test_chord_verification_does_not_false_positive_on_silence(self):
        """If the player stops playing, chord verification should not mark notes."""
        timeline = _timeline_with_major_chord()
        matcher = NoteMatcher(timeline, timing_window_ms=100.0, chord_partial_credit=True)
        chord_detector = ChordDetector(sample_rate=48000)

        # Push silence
        silence = np.zeros(int(48000 * 0.5), dtype=np.float32)
        hop = 512
        for i in range(0, len(silence), hop):
            chord_detector.push_audio(silence[i:i + hop])

        # YIN matches root
        detected = [_detected(40, 1000.0)]
        matcher.process_detected_notes(detected, 1050.0)

        # FFT should NOT mark remaining notes (no chord audio present)
        results = matcher.verify_chord_at(1050.0, chord_detector, has_onset=True)

        # At least one of the remaining notes should still be PENDING
        pending_count = sum(
            1 for n in timeline.notes
            if matcher._get_state(n) == MatchType.PENDING
        )
        assert pending_count >= 2, (
            f"Remaining notes should be PENDING (no chord audio), "
            f"got {pending_count} pending"
        )

    def test_onset_cache_does_not_leak_between_different_chords(self):
        """CRITICAL: verify_chord_with_onset cache must not return stale results
        for a different chord with the same note count.

        This was the bug that caused chord verification to silently fail:
        chord A (E2+B2) was cached, then chord B (A2+E3) with the same
        note count (2) got chord A's cached result.
        """
        detector = ChordDetector(sample_rate=48000)

        # Push E2+B2 chord audio and verify with onset
        _push_chord_signal(detector, [40, 47])
        result_a = detector.verify_chord_with_onset([40, 47], has_onset=True)
        assert result_a == [True, True], f"E2+B2 should be [True, True], got {result_a}"

        # Now push silence (player stopped)
        silence = np.zeros(int(48000 * 0.5), dtype=np.float32)
        hop = 512
        for i in range(0, len(silence), hop):
            detector.push_audio(silence[i:i + hop])

        # Query a DIFFERENT chord (A2+E3) without onset — should NOT return
        # the cached [True, True] from E2+B2.
        result_b = detector.verify_chord_with_onset([45, 52], has_onset=False)
        assert result_b == [False, False], (
            f"Silent buffer should return [False, False] for A2+E3, "
            f"got {result_b} (cache leaked from E2+B2!)"
        )

    def test_two_different_chords_both_verified(self):
        """Two different chords in sequence should both be correctly verified."""
        timeline = _timeline_with_two_chords()
        matcher = NoteMatcher(timeline, timing_window_ms=100.0, chord_partial_credit=True)
        feedback = FeedbackRenderer()
        chord_detector = ChordDetector(sample_rate=48000)

        # --- Chord 1: E2+B2 at t=1000ms ---
        _push_chord_signal(chord_detector, [40, 47])
        detected = [_detected(40, 1000.0)]
        matcher.process_detected_notes(detected, 1050.0)
        fft_results = matcher.verify_chord_at(1050.0, chord_detector, has_onset=True)
        feedback.add_results(fft_results, 1050.0)

        assert matcher.hits >= 2, f"Chord 1: both notes should be hit, got {matcher.hits}"

        # --- Chord 2: A2+E3 at t=2000ms ---
        _push_chord_signal(chord_detector, [45, 52])
        detected = [_detected(45, 2000.0)]
        matcher.process_detected_notes(detected, 2050.0)
        fft_results = matcher.verify_chord_at(2050.0, chord_detector, has_onset=True)
        feedback.add_results(fft_results, 2050.0)

        assert matcher.hits >= 4, f"Chord 2: all 4 notes should be hit, got {matcher.hits}"


# ===================================================================
# Full gameplay loop simulation
# ===================================================================

class TestGameplayLoopSimulation:
    """Simulate multiple frames of gameplay and verify match accumulation."""

    def test_sequential_notes_match_over_time(self):
        """Play through 3 sequential notes, verify hits accumulate."""
        timeline = _timeline_with_single_notes()
        matcher = NoteMatcher(timeline, timing_window_ms=100.0)
        feedback = FeedbackRenderer()

        # Frame 1: play E2 at t=1000ms
        detected = [_detected(40, 1000.0)]
        results = matcher.process_detected_notes(detected, 1050.0)
        feedback.add_results(results, 1050.0)
        assert matcher.hits == 1
        assert feedback.streak == 1

        # Frame 2: play F#2 at t=2000ms
        detected = [_detected(42, 2000.0)]
        results = matcher.process_detected_notes(detected, 2050.0)
        feedback.add_results(results, 2050.0)
        assert matcher.hits == 2
        assert feedback.streak == 2

        # Frame 3: play G#2 at t=3000ms
        detected = [_detected(44, 3000.0)]
        results = matcher.process_detected_notes(detected, 3050.0)
        feedback.add_results(results, 3050.0)
        assert matcher.hits == 3
        assert feedback.streak == 3

    def test_miss_breaks_streak(self):
        """A missed note resets the streak counter."""
        timeline = _timeline_with_single_notes()
        matcher = NoteMatcher(timeline, timing_window_ms=100.0)
        feedback = FeedbackRenderer()

        # Hit first note
        detected = [_detected(40, 1000.0)]
        results = matcher.process_detected_notes(detected, 1050.0)
        feedback.add_results(results, 1050.0)
        assert feedback.streak == 1

        # Miss second note (advance past its window without playing)
        results = matcher.process_detected_notes([], 2200.0)
        feedback.add_results(results, 2200.0)
        assert matcher.misses >= 1
        assert feedback.streak == 0, f"Streak should reset to 0, got {feedback.streak}"

    def test_stats_reflect_all_matches(self):
        """After a full playthrough, stats should show hits, misses, and accuracy."""
        timeline = _timeline_with_single_notes()
        matcher = NoteMatcher(timeline, timing_window_ms=100.0)
        feedback = FeedbackRenderer()

        # Hit note 1, miss note 2, hit note 3
        for note_midi, play_ms, advance_ms in [
            (40, 1000.0, 1050.0),   # Hit E2
            (None, 2000.0, 2200.0),  # Miss F#2
            (44, 3000.0, 3050.0),   # Hit G#2
        ]:
            if note_midi is not None:
                detected = [_detected(note_midi, play_ms)]
            else:
                detected = []
            results = matcher.process_detected_notes(detected, advance_ms)
            feedback.add_results(results, advance_ms)

        stats = matcher.get_statistics()
        assert stats["hits"] == 2, f"Expected 2 hits, got {stats['hits']}"
        assert stats["misses"] == 1, f"Expected 1 miss, got {stats['misses']}"
        assert 0 < stats["accuracy_percent"] < 100


# ===================================================================
# Chord detector cache correctness
# ===================================================================

class TestChordDetectorCache:
    """Directly test the onset-gating cache for correctness."""

    def test_cache_returns_same_result_within_ttl(self):
        """Without onset, repeated calls return the cached result."""
        detector = ChordDetector(sample_rate=48000)
        _push_chord_signal(detector, [40, 47])

        r1 = detector.verify_chord_with_onset([40, 47], has_onset=True)
        r2 = detector.verify_chord_with_onset([40, 47], has_onset=False)
        assert r1 == r2, "Cache should return same result within TTL"

    def test_cache_invalidated_on_different_note_count(self):
        """Different note count falls through to fresh analysis."""
        detector = ChordDetector(sample_rate=48000)
        _push_chord_signal(detector, [40, 47])

        r1 = detector.verify_chord_with_onset([40, 47], has_onset=True)
        # Different note count — should not use cache
        r2 = detector.verify_chord_with_onset([40, 47, 50], has_onset=False)
        assert len(r2) == 3, f"Should return 3 results for 3-note chord, got {len(r2)}"

    def test_reset_clears_cache(self):
        """After reset, the cache is empty and returns fresh results."""
        detector = ChordDetector(sample_rate=48000)
        _push_chord_signal(detector, [40, 47])
        detector.verify_chord_with_onset([40, 47], has_onset=True)

        detector.reset()
        # Buffer is empty after reset → all False
        result = detector.verify_chord_with_onset([40, 47], has_onset=False)
        assert result == [False, False], f"After reset, should be [False, False], got {result}"


# ===================================================================
# Resilience: edge cases and adverse conditions
# ===================================================================

class TestChordDetectorResilience:
    """ChordDetector must handle edge cases without crashing."""

    def test_empty_expected_notes(self):
        """verify_chord([]) returns [] without error."""
        d = ChordDetector(sample_rate=48000)
        assert d.verify_chord([]) == []
        assert d.verify_chord_with_onset([], has_onset=True) == []

    def test_duplicate_expected_notes(self):
        """Duplicate MIDI notes in the chord list are handled gracefully."""
        d = ChordDetector(sample_rate=48000)
        _push_chord_signal(d, [40, 47])
        results = d.verify_chord([40, 40, 47])
        assert len(results) == 3, f"Should return 3 results, got {len(results)}"

    def test_single_note_chord(self):
        """A single-note 'chord' is handled (not treated as a chord by matcher,
        but ChordDetector should still verify it correctly)."""
        d = ChordDetector(sample_rate=48000)
        _push_chord_signal(d, [40])
        result = d.verify_chord([40])
        assert result == [True], f"Single E2 should be detected, got {result}"

    def test_sample_rate_change_clears_cache(self):
        """Changing sample rate invalidates the cache and rebuilds bin tables."""
        d = ChordDetector(sample_rate=48000)
        _push_chord_signal(d, [40, 47])
        d.verify_chord_with_onset([40, 47], has_onset=True)

        d.set_sample_rate(44100)
        # After SR change, buffer is empty → all False
        result = d.verify_chord_with_onset([40, 47], has_onset=False)
        assert result == [False, False], (
            f"After SR change, should be [False, False], got {result}"
        )

    def test_seek_does_not_use_stale_audio(self):
        """After reset (simulating a seek), chord verification starts fresh."""
        d = ChordDetector(sample_rate=48000)
        _push_chord_signal(d, [40, 47])
        assert d.verify_chord([40, 47]) == [True, True]

        d.reset()
        # Push silence after reset
        silence = np.zeros(int(48000 * 0.3), dtype=np.float32)
        hop = 512
        for i in range(0, len(silence), hop):
            d.push_audio(silence[i:i + hop])

        result = d.verify_chord([40, 47])
        assert result == [False, False], (
            f"After reset+silence, should be [False, False], got {result}"
        )

    def test_silence_not_detected(self):
        """A silent buffer (all zeros) should not produce any detections."""
        d = ChordDetector(sample_rate=48000)
        silence = np.zeros(int(48000 * 0.5), dtype=np.float32)
        d.push_audio(silence)
        result = d.verify_chord([40])
        assert result == [False], f"Silence should not be detected, got {result}"

    def test_high_note_still_detected(self):
        """Notes near the top of the guitar range (E4, MIDI 64) still work."""
        d = ChordDetector(sample_rate=48000)
        _push_chord_signal(d, [64])
        result = d.verify_chord([64])
        assert result == [True], f"E4 should be detected, got {result}"

    def test_no_crash_on_rapid_push_small_chunks(self):
        """Pushing tiny chunks (1 sample) doesn't crash or corrupt the buffer."""
        d = ChordDetector(sample_rate=48000, fft_size=4096)
        signal = np.random.uniform(-0.5, 0.5, 48000).astype(np.float32)
        for s in signal:
            d.push_audio(np.array([s], dtype=np.float32))
        # Should not crash; result is unpredictable (noise) but must be valid
        result = d.verify_chord([40, 47])
        assert len(result) == 2
        assert all(isinstance(r, bool) for r in result)

    def test_verify_chord_performance(self):
        """verify_chord completes in < 5ms on a 16384-point FFT.

        The gameplay loop calls verify_chord every frame (~60fps = 16.7ms budget).
        With onset-gating, most frames return the cache in <0.01ms; the fresh
        analysis path (FFT + chroma + harmonic scoring) must stay under 5ms.
        """
        import time as perf_time

        d = ChordDetector(sample_rate=48000, fft_size=16384)
        _push_chord_signal(d, [40, 47, 52])

        # Warm up (first call does FFT + cache population)
        d.verify_chord_with_onset([40, 47, 52], has_onset=True)

        # Measure 20 fresh-analysis calls (force onset=True to bypass cache)
        times = []
        for _ in range(20):
            start = perf_time.perf_counter()
            d.verify_chord_with_onset([40, 47, 52], has_onset=True)
            elapsed = (perf_time.perf_counter() - start) * 1000
            times.append(elapsed)

        median_ms = sorted(times)[len(times) // 2]
        assert median_ms < 5.0, (
            f"verify_chord median {median_ms:.2f}ms exceeds 5ms budget. "
            f"All times: {[f'{t:.2f}' for t in sorted(times)]}"
        )

    def test_cached_call_is_fast(self):
        """Cached calls (has_onset=False) return in < 0.1ms."""
        import time as perf_time

        d = ChordDetector(sample_rate=48000)
        _push_chord_signal(d, [40, 47])
        d.verify_chord_with_onset([40, 47], has_onset=True)

        times = []
        for _ in range(100):
            start = perf_time.perf_counter()
            d.verify_chord_with_onset([40, 47], has_onset=False)
            elapsed = (perf_time.perf_counter() - start) * 1000
            times.append(elapsed)

        median_ms = sorted(times)[len(times) // 2]
        assert median_ms < 0.1, (
            f"Cached call median {median_ms:.3f}ms exceeds 0.1ms. "
            f"All times: {[f'{t:.3f}' for t in sorted(times)]}"
        )
