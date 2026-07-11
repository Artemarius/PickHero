"""Tests for PerformanceEvent snapshot at stabilizer emission.

The stabilizer must capture an immutable snapshot of the PerformanceEvent's
state when emitting a StableNoteEvent. This prevents race conditions where
the mutable PerformanceEvent's event_kind changes between detection,
stabilization, queueing, and matching.
"""
import pytest
from pickhero.audio.track_stabilizer import TrackStabilizer
from pickhero.audio.detector import DetectedNote
from pickhero.audio.performance import PerformanceEvent, TechniqueCandidate

pytestmark = pytest.mark.filterwarnings(
    "ignore:TrackStabilizer is deprecated:DeprecationWarning"
)

def _make_note(freq=440.0, conf=0.95, is_onset=False, ts=0.0, perf=None):
    from pickhero.audio.note_utils import freq_to_midi, midi_to_name
    midi = freq_to_midi(freq) if freq > 0 else 0
    return DetectedNote(
        midi_note=midi,
        frequency=freq,
        confidence=conf,
        name=midi_to_name(midi) if midi > 0 else "?",
        is_onset=is_onset,
        onset_sample=int(ts * 48.0) if is_onset else None,
        performance=perf,
    )


class TestEventSnapshot:
    """StableNoteEvent must carry an immutable snapshot, not a mutable reference."""

    def test_emitted_event_has_snapshot(self):
        """When the stabilizer emits, it snapshots event_kind + candidates."""
        stab = TrackStabilizer(sample_rate=44100, hop_size=512)

        perf = PerformanceEvent(onset_ms=0.0, midi_note=69, confidence=0.9)
        perf.event_kind = "legato_transition"
        perf.upsert_technique_candidate("hammer_on", 0.7, detected_cents=200.0)

        note = _make_note(freq=440.0, conf=0.95, is_onset=True, ts=0.0, perf=perf)
        events = stab.process(note, 0.0)

        # Feed sustain to reach consensus
        for i in range(1, 5):
            sustain = _make_note(freq=440.0, conf=0.99, is_onset=False, ts=i * 11.6, perf=perf)
            events.extend(stab.process(sustain, i * 11.6))

        assert len(events) == 1
        event = events[0]
        assert event.event_snapshot is not None
        assert event.event_snapshot.event_kind == "legato_transition"
        assert len(event.event_snapshot.technique_candidates) == 1
        assert event.event_snapshot.technique_candidates[0].kind == "hammer_on"

    def test_snapshot_unchanged_after_performance_mutates(self):
        """Mutating the PerformanceEvent after emission must not affect the snapshot."""
        stab = TrackStabilizer(sample_rate=44100, hop_size=512)

        perf = PerformanceEvent(onset_ms=0.0, midi_note=69, confidence=0.9)
        perf.event_kind = "legato_transition"

        note = _make_note(freq=440.0, conf=0.95, is_onset=True, ts=0.0, perf=perf)
        events = stab.process(note, 0.0)
        for i in range(1, 5):
            sustain = _make_note(freq=440.0, conf=0.99, is_onset=False, ts=i * 11.6, perf=perf)
            events.extend(stab.process(sustain, i * 11.6))

        assert len(events) == 1
        event = events[0]

        # Mutate the original PerformanceEvent AFTER emission
        perf.event_kind = "sustain_update"
        perf.upsert_technique_candidate("vibrato", 0.8)

        # The snapshot must NOT reflect the mutation
        assert event.event_snapshot.event_kind == "legato_transition", (
            "snapshot was mutated after emission — race condition"
        )
        assert len(event.event_snapshot.technique_candidates) == 0, (
            "snapshot gained candidates after emission — race condition"
        )

    def test_snapshot_none_when_no_performance(self):
        """When there's no PerformanceEvent, the snapshot is None."""
        stab = TrackStabilizer(sample_rate=44100, hop_size=512)

        note = _make_note(freq=440.0, conf=0.95, is_onset=True, ts=0.0, perf=None)
        events = stab.process(note, 0.0)
        for i in range(1, 5):
            sustain = _make_note(freq=440.0, conf=0.99, is_onset=False, ts=i * 11.6, perf=None)
            events.extend(stab.process(sustain, i * 11.6))

        assert len(events) == 1
        assert events[0].event_snapshot is None

    def test_snapshot_captures_midi_and_confidence(self):
        """The snapshot captures midi_note and confidence at emission time."""
        stab = TrackStabilizer(sample_rate=44100, hop_size=512)

        perf = PerformanceEvent(onset_ms=0.0, midi_note=69, confidence=0.9)

        note = _make_note(freq=440.0, conf=0.95, is_onset=True, ts=0.0, perf=perf)
        events = stab.process(note, 0.0)
        for i in range(1, 5):
            sustain = _make_note(freq=440.0, conf=0.99, is_onset=False, ts=i * 11.6, perf=perf)
            events.extend(stab.process(sustain, i * 11.6))

        assert len(events) == 1
        snap = events[0].event_snapshot
        assert snap is not None
        assert snap.midi_note == 69
        assert snap.confidence == 0.9
