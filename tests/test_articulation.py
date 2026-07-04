"""Tests for pickhero.audio.articulation — guitar articulation detection.

Tests the ArticulationDetector class directly by feeding synthetic pitch
contours, onset flags, and audio buffers that simulate each articulation.
"""

import numpy as np
import pytest

from pickhero.audio.articulation import ArticulationDetector


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

    Returns the first articulation detected, or None.
    """
    for i, (freq, conf, onset) in enumerate(frames):
        ts = i * hop_ms
        buf = audio if audio is not None else _tone_audio(freq if freq > 0 else base_freq)
        result = detector.process(freq, conf, onset, buf, ts)
        if result is not None:
            return result
    return None


class TestHammerOn:
    """Test hammer-on detection: abrupt ascending pitch change without onset."""

    def test_hammer_on_detected(self):
        """Pitch jumps up without onset → hammer_on."""
        d = _make_detector()
        # First frame: onset at E2
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        # Subsequent frames: pitch jumps to A2 (7 semitones up) without onset
        result = _feed_frames(d, [
            (E2, 0.95, False),
            (A2, 0.95, False),  # abrupt jump = hammer-on
        ], E2)
        assert result == "hammer_on"

    def test_hammer_on_requires_no_onset(self):
        """Pitch jump WITH onset → not hammer_on (it's a picked note)."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        result = _feed_frames(d, [
            (E2, 0.95, False),
            (A2, 0.95, True),  # onset = picked note, not legato
        ], E2)
        # Onset resets, so no legato detection
        assert result != "hammer_on"


class TestPullOff:
    """Test pull-off detection: abrupt descending pitch change without onset."""

    def test_pull_off_detected(self):
        """Pitch drops without onset → pull_off."""
        d = _make_detector()
        # First frame: onset at A2
        d.process(A2, 0.95, True, _tone_audio(A2), 0.0)
        # Subsequent frames: pitch drops to E2 without onset
        result = _feed_frames(d, [
            (A2, 0.95, False),
            (E2, 0.95, False),  # abrupt drop = pull-off
        ], A2)
        assert result == "pull_off"


class TestBend:
    """Test bend detection: gradual monotonic pitch drift."""

    def test_bend_detected(self):
        """Gradual ascending pitch over >80ms → bend."""
        d = _make_detector()
        # Onset at E2
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        # Simulate a gradual bend: 35→70→105→140→175 cents over 10 frames
        frames = []
        for cents in [10, 20, 35, 45, 55, 65, 75, 85, 95, 105]:
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        result = _feed_frames(d, frames, E2)
        assert result == "bend"

    def test_bend_requires_min_duration(self):
        """Very short pitch deviation → not a bend."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        # Only 2 frames — too short for 80ms minimum
        result = _feed_frames(d, [
            (E2 * 1.05, 0.95, False),
        ], E2)
        assert result != "bend"


class TestVibrato:
    """Test vibrato detection: periodic pitch oscillation at 3-8 Hz."""

    def test_vibrato_detected(self):
        """Periodic pitch oscillation → vibrato."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        # Simulate vibrato: 5 Hz oscillation, 30 cents amplitude
        # At hop=512/44100Hz, frame_duration ≈ 11.6ms
        # 5 Hz → period = 200ms → ~17 frames per cycle
        # Need at least 10 frames (VIB_MIN_FRAMES)
        frames = []
        for i in range(24):
            # 5 Hz vibrato, 30 cents amplitude
            cents = 30.0 * np.sin(2 * np.pi * 5.0 * i * 0.0116)
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        result = _feed_frames(d, frames, E2)
        assert result == "vibrato"
    def test_vibrato_rejected_too_slow(self):
        """Constant offset (no oscillation) → not vibrato."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        # Constant pitch offset — no oscillation at all
        frames = []
        for i in range(24):
            # Constant 30 cents offset, no oscillation
            freq = E2 * (2 ** (30.0 / 1200.0))
            frames.append((freq, 0.95, False))
        result = _feed_frames(d, frames, E2)
        # No zero crossings → rate = 0 Hz → rejected
        assert result != "vibrato"


class TestSlide:
    """Test slide detection: monotonic staircase pitch movement."""

    def test_slide_detected(self):
        """Monotonic pitch crossing semitone boundary → slide."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        # Simulate a slide: gradual cents increase crossing 100 cents
        # Each frame adds ~25 cents (below 50 cent sudden jump threshold)
        frames = []
        for cents in [10, 35, 60, 85, 110, 135, 160]:
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        result = _feed_frames(d, frames, E2)
        assert result == "slide"

    def test_slide_rejected_sudden_jump(self):
        """Sudden pitch jump >50 cents → not a slide (that's legato)."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        result = _feed_frames(d, [
            (E2, 0.95, False),
            (E2 * 1.1, 0.95, False),  # sudden jump = not slide
            (E2 * 1.2, 0.95, False),
        ], E2)
        # Should not be classified as slide (sudden jump rejects slide detection)
        # It might be classified as hammer_on instead
        assert result != "slide"


class TestPalmMute:
    """Test palm mute detection: fast energy decay + low spectral centroid."""

    def test_palm_mute_detected(self):
        """Fast decay + low centroid → palm_mute."""
        d = _make_detector()
        sr = 44100
        hop = 512
        # Onset frame with strong signal
        audio_onset = _tone_audio(E2, hop, sr) * 0.8
        # Rapidly decaying subsequent frames (palm mute characteristic)
        audio_decay1 = _tone_audio(E2, hop, sr) * 0.1
        audio_decay2 = _tone_audio(E2, hop, sr) * 0.02
        audio_decay3 = _tone_audio(E2, hop, sr) * 0.005

        # Feed onset + 4 decay frames (need _PALM_MUTE_FRAMES=4 after onset)
        d.process(E2, 0.9, True, audio_onset, 0.0)
        result = None
        for i, (audio, conf, onset) in enumerate([
            (audio_decay1, 0.8, False),
            (audio_decay2, 0.7, False),
            (audio_decay3, 0.5, False),
            (audio_decay3 * 0.1, 0.3, False),  # 4th frame
        ], 1):
            result = d.process(E2, conf, onset, audio, i * 11.6)
            if result is not None:
                break
        assert result == "palm_mute"


class TestHarmonic:
    """Test harmonic detection: weak fundamental, strong overtones."""

    def test_harmonic_detected(self):
        """Strong overtone at 1.5×F0 → harmonic."""
        d = _make_detector()
        sr = 44100
        hop = 512
        # Create audio where the 1.5× subharmonic is strong
        # This simulates a natural harmonic at the 7th fret (1/3 of string)
        t = np.arange(hop, dtype=np.float32) / sr
        # Fundamental at E2, but weak; 1.5×E2 subharmonic strong
        fundamental = 0.1 * np.sin(2 * np.pi * E2 * t)
        subharmonic = 0.8 * np.sin(2 * np.pi * 1.5 * E2 * t)
        audio = (fundamental + subharmonic).astype(np.float32)

        result = d.process(E2, 0.9, True, audio, 0.0)
        assert result == "harmonic"

    def test_normal_note_not_harmonic(self):
        """Strong fundamental, normal overtones → not harmonic."""
        d = _make_detector()
        sr = 44100
        hop = 512
        t = np.arange(hop, dtype=np.float32) / sr
        # Strong fundamental, weak overtones (normal plucked note)
        audio = (0.9 * np.sin(2 * np.pi * E2 * t) +
                 0.1 * np.sin(2 * np.pi * 2 * E2 * t)).astype(np.float32)

        result = d.process(E2, 0.95, True, audio, 0.0)
        # Should not be classified as harmonic
        assert result != "harmonic"


class TestReset:
    """Test that reset() clears all state."""

    def test_reset_clears_state(self):
        """After reset, no stale articulation should fire."""
        d = _make_detector()
        # Feed some data to populate state
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        _feed_frames(d, [
            (E2 * 1.05, 0.95, False),
            (E2 * 1.1, 0.95, False),
        ], E2)

        # Reset
        d.reset()

        # After reset, processing a single frame should not produce articulation
        result = d.process(E2, 0.95, False, _tone_audio(E2), 100.0)
        assert result is None


class TestNoFalsePositives:
    """Test that steady pitch doesn't trigger articulations."""

    def test_steady_pitch_no_articulation(self):
        """Constant pitch with no onset → no articulation."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        result = _feed_frames(d, [
            (E2, 0.95, False),
            (E2, 0.95, False),
            (E2, 0.95, False),
            (E2, 0.95, False),
            (E2, 0.95, False),
        ], E2)
        assert result is None


class TestCrossArticulation:
    """Test that articulations are correctly discriminated from each other."""

    def test_bend_not_vibrato(self):
        """Monotonic pitch drift → bend, not vibrato (no periodicity)."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        # Gradual ascending pitch (bend), not oscillating
        frames = []
        for cents in [10, 20, 35, 45, 55, 65, 75, 85, 95, 105]:
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        result = _feed_frames(d, frames, E2)
        assert result == "bend"

    def test_vibrato_not_bend(self):
        """Periodic oscillation → vibrato, not bend."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        # 5 Hz vibrato — should not be classified as bend
        frames = []
        for i in range(24):
            cents = 30.0 * np.sin(2 * np.pi * 5.0 * i * 0.0116)
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        result = _feed_frames(d, frames, E2)
        assert result == "vibrato"

    def test_slide_not_hammer_on(self):
        """Gradual monotonic pitch crossing semitone → slide, not hammer-on."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        # Gradual slide (25 cents/frame) — should not trigger hammer-on
        frames = []
        for cents in [25, 50, 75, 100, 125]:
            freq = E2 * (2 ** (cents / 1200.0))
            frames.append((freq, 0.95, False))
        result = _feed_frames(d, frames, E2)
        assert result == "slide"

    def test_hammer_on_not_slide(self):
        """Sudden pitch jump → hammer-on, not slide (too fast for slide)."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        # Sudden jump (>50 cents in 1 frame)
        result = _feed_frames(d, [
            (E2, 0.95, False),
            (A2, 0.95, False),
        ], E2)
        assert result == "hammer_on"


class TestEdgeCases:
    """Test edge cases: silence, low confidence, rapid changes."""

    def test_silent_input_no_articulation(self):
        """Zero frequency → no articulation."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        result = _feed_frames(d, [
            (0.0, 0.0, False),
            (0.0, 0.0, False),
            (0.0, 0.0, False),
        ], E2)
        assert result is None

    def test_low_confidence_no_articulation(self):
        """Confidence below 0.3 → no articulation."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        result = _feed_frames(d, [
            (A2, 0.2, False),
            (A2, 0.2, False),
            (A2, 0.2, False),
        ], E2)
        assert result is None

    def test_onset_resets_pitch_history(self):
        """A new onset should clear previous pitch context."""
        d = _make_detector()
        # First note: E2
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        # Feed some bend-like data
        _feed_frames(d, [
            (E2 * 1.03, 0.95, False),
            (E2 * 1.05, 0.95, False),
        ], E2)
        # New onset at A2
        d.process(A2, 0.95, True, _tone_audio(A2), 50.0)
        # Immediately after onset, a small deviation should not trigger bend
        # (bend requires >35 cents sustained for >80ms)
        result = d.process(A2 * 1.02, 0.95, False, _tone_audio(A2), 55.0)
        assert result is None

    def test_no_base_freq_no_articulation(self):
        """Before any onset, no articulation should fire."""
        d = _make_detector()
        result = d.process(E2, 0.95, False, _tone_audio(E2), 0.0)
        assert result is None

    def test_rapid_note_changes(self):
        """Rapid onset → onset → onset should not trigger false articulations."""
        d = _make_detector()
        d.process(E2, 0.95, True, _tone_audio(E2), 0.0)
        d.process(A2, 0.95, True, _tone_audio(A2), 11.6)
        result = d.process(E3, 0.95, True, _tone_audio(E3), 23.2)
        # Onsets reset everything, no articulation expected
        assert result is None

    def test_articulation_field_default_none(self):
        """DetectedNote.articulation should default to None."""
        from pickhero.audio.detector import DetectedNote
        note = DetectedNote(
            midi_note=64, frequency=440.0, confidence=0.95,
            name="A4", is_onset=True,
        )
        assert note.articulation is None
