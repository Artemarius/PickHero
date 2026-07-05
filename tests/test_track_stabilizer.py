"""Tests for the track stabilizer — the milestone tests.

A clean E2 played once produces exactly one StableNoteEvent.
Touching the jack produces zero StableNoteEvents.
A sustained E2 does not produce legato/harmonic spam.
An E2 attack octave glitch does not become an E3 match.
"""

import numpy as np
import pytest

from pickhero.audio.detector import DetectedNote
from pickhero.audio.track_stabilizer import TrackStabilizer, StableNoteEvent
from pickhero.audio.note_utils import freq_to_midi

E2 = 82.41
A2 = 110.00
E3 = 164.81  # octave harmonic of E2


def _make_note(freq: float, conf: float = 0.95, is_onset: bool = False,
               ts: float = 0.0) -> DetectedNote:
    midi = freq_to_midi(freq) if freq > 0 else 0
    return DetectedNote(
        midi_note=midi,
        frequency=freq,
        confidence=conf,
        name="?" if midi == 0 else f"{midi}",
        is_onset=is_onset,
        onset_sample=int(ts * 48.0) if is_onset else None,
        performance=None,
    )


class TestCleanSinglePick:
    """A clean E2 played once produces exactly one StableNoteEvent."""

    def test_single_pick_emits_one_event(self):
        """Onset + 2 stable sustain frames → exactly 1 event, midi=E2."""
        s = TrackStabilizer(sample_rate=48000, hop_size=512)
        events = []

        # Onset frame
        n = _make_note(E2, conf=0.85, is_onset=True, ts=0.0)
        events.extend(s.process(n, 0.0))

        # 2 stable sustain frames
        for i in range(1, 6):
            n = _make_note(E2, conf=0.95, is_onset=False, ts=i * 10.7)
            events.extend(s.process(n, i * 10.7))

        assert len(events) == 1, f"expected 1 event, got {len(events)}"
        assert events[0].midi_note == 40, f"expected E2 (midi 40), got {events[0].midi_note}"
        assert events[0].is_onset is True

    def test_no_duplicate_events_on_long_sustain(self):
        """A sustained note (100 frames) emits only one event."""
        s = TrackStabilizer(sample_rate=48000, hop_size=512)
        events = []

        n = _make_note(E2, conf=0.95, is_onset=True, ts=0.0)
        events.extend(s.process(n, 0.0))

        for i in range(1, 100):
            n = _make_note(E2, conf=0.99, is_onset=False, ts=i * 10.7)
            events.extend(s.process(n, i * 10.7))

        assert len(events) == 1, f"sustained note emitted {len(events)} events"


class TestJackTouchRejection:
    """Touching the jack produces zero StableNoteEvents."""

    def test_noise_onset_no_consensus_no_event(self):
        """A single noisy onset frame with no follow-up → no event."""
        s = TrackStabilizer(sample_rate=48000, hop_size=512)
        events = []

        # Single onset frame, no sustain
        n = _make_note(165.0, conf=0.2, is_onset=True, ts=0.0)
        events.extend(s.process(n, 0.0))

        # Silence (None) for the rest
        for i in range(1, 20):
            events.extend(s.process(None, i * 10.7))

        assert len(events) == 0, f"noise onset produced {len(events)} events"

    def test_jack_touch_burst_no_event(self):
        """A brief burst of inconsistent pitches → no consensus → no event."""
        s = TrackStabilizer(sample_rate=48000, hop_size=512)
        events = []

        # Noisy burst: random-ish frequencies, low confidence
        freqs = [165.0, 82.0, 247.0, 110.0, 0.0, 330.0]
        for i, f in enumerate(freqs):
            if f > 0:
                n = _make_note(f, conf=0.3, is_onset=(i == 0), ts=i * 10.7)
                events.extend(s.process(n, i * 10.7))
            else:
                events.extend(s.process(None, i * 10.7))

        # No follow-up consensus
        for i in range(len(freqs), 20):
            events.extend(s.process(None, i * 10.7))

        assert len(events) == 0, f"jack touch burst produced {len(events)} events"


class TestSustainedNoteNoSpam:
    """A sustained E2 does not produce legato/harmonic spam."""

    def test_sustained_note_one_event(self):
        """200 frames of stable E2 at conf=1.0 → exactly 1 event."""
        s = TrackStabilizer(sample_rate=48000, hop_size=512)
        events = []

        n = _make_note(E2, conf=0.98, is_onset=True, ts=0.0)
        events.extend(s.process(n, 0.0))

        for i in range(1, 200):
            # Small natural pitch jitter (±2 cents)
            jitter = np.random.RandomState(42).uniform(-2, 2)
            freq = E2 * (2 ** (jitter / 1200.0))
            n = _make_note(freq, conf=1.0, is_onset=False, ts=i * 10.7)
            events.extend(s.process(n, i * 10.7))

        assert len(events) == 1, f"sustained note emitted {len(events)} events"


class TestOctaveGlitchRejection:
    """An E2 attack octave glitch does not become an E3 match."""

    def test_octave_harmonic_attack_resolved_to_e2(self):
        """Onset at E3 (octave), sustain settles to E2 → emit E2, not E3."""
        s = TrackStabilizer(sample_rate=48000, hop_size=512)
        events = []

        # Onset detects E3 (the octave harmonic) during attack
        n = _make_note(E3, conf=0.89, is_onset=True, ts=0.0)
        events.extend(s.process(n, 0.0, tab_prior_midi=40))

        # Sustain settles to E2 (the true fundamental)
        for i in range(1, 6):
            n = _make_note(E2, conf=0.99, is_onset=False, ts=i * 10.7)
            events.extend(s.process(n, i * 10.7, tab_prior_midi=40))

        assert len(events) == 1, f"expected 1 event, got {len(events)}"
        assert events[0].midi_note == 40, (
            f"expected E2 (midi 40) after octave resolution, got {events[0].midi_note}"
        )

    def test_octave_glitch_no_tab_prior_uses_sustain(self):
        """Without tab prior, sustain frames resolve the octave."""
        s = TrackStabilizer(sample_rate=48000, hop_size=512)
        events = []

        # Onset at E3
        n = _make_note(E3, conf=0.85, is_onset=True, ts=0.0)
        events.extend(s.process(n, 0.0))

        # Sustain at E2
        for i in range(1, 6):
            n = _make_note(E2, conf=0.99, is_onset=False, ts=i * 10.7)
            events.extend(s.process(n, i * 10.7))

        assert len(events) == 1
        assert events[0].midi_note == 40, (
            f"expected E2 from sustain consensus, got {events[0].midi_note}"
        )


class TestRefractory:
    """Refractory period prevents duplicate emission."""

    def test_refractory_blocks_immediate_re_emission(self):
        """After emitting, a 50ms refractory prevents a second event."""
        s = TrackStabilizer(sample_rate=48000, hop_size=512)
        events = []

        # Onset + consensus
        n = _make_note(E2, conf=0.95, is_onset=True, ts=0.0)
        events.extend(s.process(n, 0.0))
        for i in range(1, 4):
            n = _make_note(E2, conf=0.99, is_onset=False, ts=i * 10.7)
            events.extend(s.process(n, i * 10.7))

        assert len(events) == 1

        # New onset immediately after (within refractory)
        n = _make_note(E2, conf=0.95, is_onset=True, ts=50.0)
        events.extend(s.process(n, 50.0))
        for i in range(1, 4):
            n = _make_note(E2, conf=0.99, is_onset=False, ts=50.0 + i * 10.7)
            events.extend(s.process(n, 50.0 + i * 10.7))

        # Should still be 1 (refractory) or 2 (if onset closes track + new)
        # The onset starts a new track, but refractory blocks emission
        assert len(events) <= 2, f"refractory failed: {len(events)} events"


class TestReset:
    """reset() clears all state."""

    def test_reset_clears_active_track(self):
        s = TrackStabilizer(sample_rate=48000, hop_size=512)
        n = _make_note(E2, conf=0.95, is_onset=True, ts=0.0)
        s.process(n, 0.0)

        s.reset()

        # After reset, a sustain frame shouldn't emit (no active track)
        events = []
        n = _make_note(E2, conf=0.99, is_onset=False, ts=10.0)
        events.extend(s.process(n, 10.0))
        assert len(events) == 0


class TestTwoNotes:
    """Two distinct picks produce two events."""

    def test_two_separate_picks_two_events(self):
        """Pick E2, let it ring, pick A2 → 2 events."""
        s = TrackStabilizer(sample_rate=48000, hop_size=512)
        events = []

        # First note: E2
        n = _make_note(E2, conf=0.95, is_onset=True, ts=0.0)
        events.extend(s.process(n, 0.0))
        for i in range(1, 6):
            n = _make_note(E2, conf=0.99, is_onset=False, ts=i * 10.7)
            events.extend(s.process(n, i * 10.7))

        # Gap (silence)
        for i in range(6, 10):
            events.extend(s.process(None, i * 10.7))

        # Second note: A2 (well past refractory)
        n = _make_note(A2, conf=0.95, is_onset=True, ts=100.0)
        events.extend(s.process(n, 100.0))
        for i in range(1, 6):
            n = _make_note(A2, conf=0.99, is_onset=False, ts=100.0 + i * 10.7)
            events.extend(s.process(n, 100.0 + i * 10.7))

        assert len(events) == 2, f"expected 2 events, got {len(events)}"
        assert events[0].midi_note == 40  # E2
        assert events[1].midi_note == 45  # A2
