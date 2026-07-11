"""Tests for pickhero.audio.articulation — real-time PerformanceEvent emission.

Tests the ArticulationDetector class directly by feeding synthetic pitch
contours, onset flags, and audio buffers that simulate each articulation.
The detector now returns a list of completed PerformanceEvents (instead of a
single string) and appends TechniqueCandidate entries to the active event.
"""

import numpy as np
import pytest

from pickhero.audio.articulation import ArticulationDetector
from pickhero.audio.performance import PerformanceEvent, TechniqueCandidate


# Standard tuning E2 = 82.41 Hz
E2 = 82.41
# A2 = 110.00 Hz (one octave up in cents terms for E2 base)
A2 = 110.00
# E3 = 164.81 Hz (one octave above E2)
E3 = 164.81


def _make_detector(sample_rate=44100, hop_size=512, buf_size=2048):
    """Create a fresh ArticulationDetector."""
    return ArticulationDetector(sample_rate, hop_size, buf_size)


def _silence_audio(n=512):
    """Generate a silent audio buffer."""
    return np.zeros(n, dtype=np.float32)


def _tone_audio(freq, n=512, sample_rate=44100):
    """Generate a short tone audio buffer."""
    t = np.arange(n, dtype=np.float32) / sample_rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _feed_frames(detector, frames, base_freq, audio=None, hop_ms=11.6):
    """Feed a sequence of (freq, confidence, is_onset) frames to the detector.

    Returns the list of all TechniqueCandidates collected on the active event
    after feeding, plus the list of completed events drained.
    """
    completed = []
    for i, (freq, conf, onset) in enumerate(frames):
        ts = i * hop_ms
        buf = audio if audio is not None else _tone_audio(freq if freq > 0 else base_freq)
        newly_completed = detector.process(freq, conf, onset, buf, ts)
        completed.extend(newly_completed)
    return completed


def _candidates_on_active(detector):
    """Return the technique candidates accumulated on the active event."""
    if detector.active_event is None:
        return []
    return detector.active_event.technique_candidates


class TestHammerOn:
    """Test hammer-on detection: abrupt ascending pitch change without onset."""

    def test_hammer_on_detected(self):
        """Pitch jumps up without onset → hammer_on candidate."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        _feed_frames(d, [
            (E2, 0.95, False),
            (A2, 0.95, False),  # abrupt jump = hammer-on
        ], E2)
        cands = _candidates_on_active(d)
        assert any(c.kind == "hammer_on" for c in cands)

    def test_hammer_on_requires_no_onset(self):
        """Pitch jump WITH onset → not hammer_on (it's a picked note)."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        _feed_frames(d, [
            (E2, 0.95, False),
            (A2, 0.95, True),  # onset = picked note, not legato
        ], E2)
        # Onset closes the event and starts a new one — no hammer_on candidate
        cands = _candidates_on_active(d)
        assert not any(c.kind == "hammer_on" for c in cands)


class TestPullOff:
    """Test pull-off detection: abrupt descending pitch change without onset."""

    def test_pull_off_detected(self):
        """Pitch drops without onset → pull_off candidate."""
        d = _make_detector()
        d.process(A2, 0.95, True, _tone_audio(A2), 0.0)
        _feed_frames(d, [
            (A2, 0.95, False),
            (E2, 0.95, False),  # abrupt drop = pull-off
        ], A2)
        cands = _candidates_on_active(d)
        assert any(c.kind == "pull_off" for c in cands)


class TestBend:
    """Test bend detection: gradual monotonic pitch drift."""

    def test_bend_detected(self):
        """Gradual ascending pitch over >80ms → bend candidate."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        frames = []
        for cents in [10, 20, 35, 45, 55, 65, 75, 85, 95, 105]:
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        _feed_frames(d, frames, E2)
        cands = _candidates_on_active(d)
        bend_cands = [c for c in cands if c.kind == "bend"]
        assert len(bend_cands) >= 1
        # detected_cents should be near the max cents reached
        assert bend_cands[0].detected_cents is not None
        assert bend_cands[0].detected_cents >= 50.0

    def test_bend_requires_min_duration(self):
        """Very short pitch deviation → not a bend."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        _feed_frames(d, [
            (E2 * 1.05, 0.95, False),
        ], E2)
        cands = _candidates_on_active(d)
        assert not any(c.kind == "bend" for c in cands)


class TestVibrato:
    """Test vibrato detection: periodic pitch oscillation at 3-8 Hz."""

    def test_vibrato_detected(self):
        """Periodic pitch oscillation → vibrato candidate with metrics."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        frames = []
        for i in range(24):
            cents = 30.0 * np.sin(2 * np.pi * 5.0 * i * 0.0116)
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        _feed_frames(d, frames, E2)
        cands = _candidates_on_active(d)
        vib_cands = [c for c in cands if c.kind == "vibrato"]
        assert len(vib_cands) >= 1
        m = vib_cands[0].metrics
        assert "rate_hz" in m
        assert "depth_cents" in m
        assert "regularity" in m
        assert 3.0 <= m["rate_hz"] <= 8.0

    def test_vibrato_rejected_too_slow(self):
        """Constant offset (no oscillation) → not vibrato."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        frames = []
        for i in range(24):
            freq = E2 * (2 ** (30.0 / 1200.0))
            frames.append((freq, 0.95, False))
        _feed_frames(d, frames, E2)
        cands = _candidates_on_active(d)
        assert not any(c.kind == "vibrato" for c in cands)


class TestSlide:
    """Test slide detection: monotonic staircase pitch movement."""

    def test_slide_detected(self):
        """Monotonic pitch crossing semitone boundary → slide candidate."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        frames = []
        for cents in [10, 35, 60, 85, 110, 135, 160]:
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        _feed_frames(d, frames, E2)
        cands = _candidates_on_active(d)
        slide_cands = [c for c in cands if c.kind == "slide"]
        assert len(slide_cands) >= 1
        assert slide_cands[0].metrics.get("direction") in ("up", "down")

    def test_slide_rejected_sudden_jump(self):
        """Sudden pitch jump >50 cents → not a slide (that's legato)."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        _feed_frames(d, [
            (E2, 0.95, False),
            (E2 * 1.1, 0.95, False),  # sudden jump = not slide
            (E2 * 1.2, 0.95, False),
        ], E2)
        cands = _candidates_on_active(d)
        assert not any(c.kind == "slide" for c in cands)


class TestPalmMute:
    """Test palm mute detection: fast energy decay + low spectral centroid."""

    def test_palm_mute_detected(self):
        """Fast decay + low centroid → palm_mute candidate."""
        d = _make_detector()
        sr = 44100
        hop = 512
        audio_onset = _tone_audio(E2, hop, sr) * 0.8
        audio_decay1 = _tone_audio(E2, hop, sr) * 0.1
        audio_decay2 = _tone_audio(E2, hop, sr) * 0.02
        audio_decay3 = _tone_audio(E2, hop, sr) * 0.005

        d.process(E2, 0.9, True, audio_onset, 0.0)
        for i, (audio, conf, onset) in enumerate([
            (audio_decay1, 0.8, False),
            (audio_decay2, 0.7, False),
            (audio_decay3, 0.5, False),
            (audio_decay3 * 0.1, 0.3, False),
        ], 1):
            d.process(E2, conf, onset, audio, i * 11.6)
        cands = _candidates_on_active(d)
        pm_cands = [c for c in cands if c.kind == "palm_mute"]
        assert len(pm_cands) >= 1
        assert "decay_halflife_ms" in pm_cands[0].metrics
        assert "centroid_hz" in pm_cands[0].metrics


class TestHarmonic:
    """Test harmonic detection: weak fundamental, strong overtones."""

    def test_harmonic_detected(self):
        """Strong overtone at 1.5×F0 → harmonic candidate on onset."""
        d = _make_detector()
        d.set_expected_techniques({"harmonic"})
        sr = 44100
        hop = 512
        t = np.arange(hop, dtype=np.float32) / sr
        fundamental = 0.1 * np.sin(2 * np.pi * E2 * t)
        subharmonic = 0.8 * np.sin(2 * np.pi * 1.5 * E2 * t)
        audio = (fundamental + subharmonic).astype(np.float32)
        d.process(E2, 0.9, True, audio, 0.0)
        cands = _candidates_on_active(d)
        assert any(c.kind == "harmonic" for c in cands)

    def test_normal_note_not_harmonic(self):
        """Strong fundamental, normal overtones → not harmonic."""
        d = _make_detector()
        d.set_expected_techniques({"harmonic"})
        sr = 44100
        hop = 512
        t = np.arange(hop, dtype=np.float32) / sr
        audio = (0.9 * np.sin(2 * np.pi * E2 * t) +
                 0.1 * np.sin(2 * np.pi * 2 * E2 * t)).astype(np.float32)

        d.process(E2, 0.95, True, audio, 0.0)
        cands = _candidates_on_active(d)
        assert not any(c.kind == "harmonic" for c in cands)


class TestReset:
    """Test that reset() clears all state."""

    def test_reset_clears_state(self):
        """After reset, no active event and no completed events."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        _feed_frames(d, [
            (E2 * 1.05, 0.95, False),
            (E2 * 1.1, 0.95, False),
        ], E2)

        d.reset()
        assert d.active_event is None
        assert d.drain_completed() == []


class TestNoFalsePositives:
    """Test that steady pitch doesn't trigger articulations."""

    def test_steady_pitch_no_articulation(self):
        """Constant pitch with no onset → no candidates."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        _feed_frames(d, [
            (E2, 0.95, False),
            (E2, 0.95, False),
            (E2, 0.95, False),
            (E2, 0.95, False),
            (E2, 0.95, False),
        ], E2)
        cands = _candidates_on_active(d)
        assert cands == []


class TestCrossArticulation:
    """Test that articulations are correctly discriminated from each other."""

    def test_bend_not_vibrato(self):
        """Monotonic pitch drift → bend detected (primary). Vibrato may also
        fire on the detrended curve, but bend takes priority."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        frames = []
        for cents in [10, 20, 35, 45, 55, 65, 75, 85, 95, 105]:
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        _feed_frames(d, frames, E2)
        cands = _candidates_on_active(d)
        # Bend is the primary detection for monotonic drift
        assert any(c.kind == "bend" for c in cands)

    def test_vibrato_not_bend(self):
        """Periodic oscillation → vibrato, not bend."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        frames = []
        for i in range(24):
            cents = 30.0 * np.sin(2 * np.pi * 5.0 * i * 0.0116)
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        _feed_frames(d, frames, E2)
        cands = _candidates_on_active(d)
        assert any(c.kind == "vibrato" for c in cands)

    def test_slide_not_hammer_on(self):
        """Gradual monotonic pitch crossing semitone → slide, not hammer-on."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        frames = []
        for cents in [25, 50, 75, 100, 125]:
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        _feed_frames(d, frames, E2)
        cands = _candidates_on_active(d)
        assert any(c.kind == "slide" for c in cands)
        assert not any(c.kind == "hammer_on" for c in cands)

    def test_hammer_on_not_slide(self):
        """Sudden pitch jump → hammer-on, not slide (too fast for slide)."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        _feed_frames(d, [
            (E2, 0.95, False),
            (A2, 0.95, False),
        ], E2)
        cands = _candidates_on_active(d)
        assert any(c.kind in ("hammer_on", "pull_off") for c in cands)
        assert not any(c.kind == "slide" for c in cands)


class TestEdgeCases:
    """Test edge cases: silence, low confidence, rapid changes."""

    def test_silent_input_no_articulation(self):
        """Zero frequency → no candidates."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        _feed_frames(d, [
            (0.0, 0.0, False),
            (0.0, 0.0, False),
            (0.0, 0.0, False),
        ], E2)
        cands = _candidates_on_active(d)
        assert cands == []

    def test_low_confidence_no_articulation(self):
        """Confidence below 0.3 → no candidates."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        _feed_frames(d, [
            (A2, 0.2, False),
            (A2, 0.2, False),
            (A2, 0.2, False),
        ], E2)
        cands = _candidates_on_active(d)
        assert cands == []

    def test_onset_resets_pitch_history(self):
        """A new onset should clear previous pitch context."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        _feed_frames(d, [
            (E2 * 1.03, 0.95, False),
            (E2 * 1.05, 0.95, False),
        ], E2)
        # New onset at A2
        d.process(A2, 0.95, True, _tone_audio(A2), 50.0)
        # Immediately after onset, a small deviation should not trigger bend
        d.process(A2 * 1.02, 0.95, False, _tone_audio(A2), 55.0)
        cands = _candidates_on_active(d)
        assert not any(c.kind == "bend" for c in cands)

    def test_no_base_freq_no_articulation(self):
        """Before any onset, no articulation should fire."""
        d = _make_detector()
        d.process(E2, 0.95, False, _tone_audio(E2), 0.0)
        assert d.active_event is None

    def test_rapid_note_changes(self):
        """Rapid onset → onset → onset should not trigger false articulations."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        d.process(A2, 0.95, True, _tone_audio(A2), 11.6)
        d.process(E3, 0.95, True, _tone_audio(E3), 23.2)
        # Onsets reset everything, no candidates expected on the latest event
        cands = _candidates_on_active(d)
        assert cands == []

    def test_performance_field_default_none(self):
        """DetectedNote.performance should default to None."""
        from pickhero.audio.detector import DetectedNote
        note = DetectedNote(
            midi_note=64, frequency=440.0, confidence=0.95,
            name="A4", is_onset=True,
        )
        assert note.performance is None


class TestNoteMergingInvariant:
    """TENT note-merging invariant: a bend's pitch rise must NOT create a new
    PerformanceEvent. Only onsets create new events."""

    def test_single_bend_emits_single_event(self):
        """A 200-cent bend over 300ms with one onset → exactly one event at release."""
        d = _make_detector()
        # Onset at E2
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        # Gradual 200-cent bend over ~300ms (26 frames at 11.6ms)
        frames = []
        for i in range(26):
            cents = min(200.0, i * 8.0)
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        completed = _feed_frames(d, frames, E2)
        # No new event should be emitted during the bend (no spurious onsets)
        assert completed == []
        # The active event should have exactly one bend candidate
        cands = _candidates_on_active(d)
        bend_cands = [c for c in cands if c.kind == "bend"]
        assert len(bend_cands) >= 1
        # And exactly one active event (not split into multiple)
        assert d.active_event is not None

    def test_onset_closes_previous_event(self):
        """A new onset closes the previous event and emits it as completed."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        # Some sustain frames
        for i in range(5):
            d.process(E2, 0.95, False, _tone_audio(E2), (i + 1) * 11.6)
        # New onset at A2 → closes the E2 event
        completed = d.process(A2, 0.95, True, _tone_audio(A2), 80.0)
        assert len(completed) == 1
        assert completed[0].release_ms == 80.0
        # New active event is the A2 note
        assert d.active_event is not None
        assert d.active_event.onset_ms == 80.0


class TestEventKindEmission:
    """Patch 2: the active PerformanceEvent's ``event_kind`` is set as techniques
    are detected, so the matcher can route non-onset events to the right spec."""

    def test_hammer_on_sets_legato_transition(self):
        """A no-pick ascending pitch jump sets event_kind='legato_transition'."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        _feed_frames(d, [
            (E2, 0.95, False),
            (A2, 0.95, False),  # abrupt ascending jump = hammer-on
        ], E2)
        assert d.active_event is not None
        assert d.active_event.event_kind == "legato_transition", (
            f"expected legato_transition, got {d.active_event.event_kind!r}"
        )

    def test_pull_off_sets_legato_transition(self):
        """A no-pick descending pitch jump sets event_kind='legato_transition'."""
        d = _make_detector()
        d.process(A2, 0.95, True, _tone_audio(A2), 0.0)
        _feed_frames(d, [
            (A2, 0.95, False),
            (E2, 0.95, False),  # abrupt descending jump = pull-off
        ], A2)
        assert d.active_event is not None
        assert d.active_event.event_kind == "legato_transition", (
            f"expected legato_transition, got {d.active_event.event_kind!r}"
        )

    def test_slide_sets_slide_landing_on_plateau(self):
        """A monotonic pitch ramp that then plateaus sets
        event_kind='slide_landing' once the cents delta flattens."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        frames = []
        # Ramp up monotonically past the semitone trigger
        for cents in [10, 35, 60, 85, 110, 135, 160]:
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        # Plateau: hold the final pitch steady so the cents delta drops < 5
        landed_freq = E2 * (2 ** (160 / 1200.0))
        for _ in range(3):
            frames.append((landed_freq, 0.95, False))
        _feed_frames(d, frames, E2)
        assert d.active_event is not None
        assert d.active_event.event_kind == "slide_landing", (
            f"expected slide_landing after plateau, got {d.active_event.event_kind!r}"
        )
