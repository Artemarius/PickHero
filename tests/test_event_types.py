"""Tests for pickhero.audio.event_types — frozen snapshot types."""

from dataclasses import FrozenInstanceError
import pytest

from pickhero.audio.event_types import (
    EventKindSnapshot,
    PitchCandidate,
    RawAudioChunk,
    StableNoteEvent,
)
from pickhero.audio.performance import TechniqueCandidate


class TestRawAudioChunkFrozen:
    """RawAudioChunk must be a frozen dataclass."""

    def test_is_immutable_on_field_assignment(self):
        chunk = RawAudioChunk(
            samples=b"\x00\x01\x02\x03",
            sample_index=0,
            timestamp_ms=0.0,
            sample_rate=48000,
        )
        with pytest.raises(FrozenInstanceError):
            chunk.sample_index = 1

    def test_bytes_field_cannot_be_replaced(self):
        chunk = RawAudioChunk(b"\xaa", 0, 0.0, 48000)
        with pytest.raises(FrozenInstanceError):
            chunk.samples = b"\xbb"

    def test_all_fields_accessible(self):
        chunk = RawAudioChunk(b"\x00", 42, 3.14, 44100)
        assert chunk.samples == b"\x00"
        assert chunk.sample_index == 42
        assert chunk.timestamp_ms == 3.14
        assert chunk.sample_rate == 44100


class TestEventKindSnapshotFrozen:
    """EventKindSnapshot must be a frozen dataclass."""

    def test_is_immutable(self):
        snap = EventKindSnapshot(
            event_kind="pick_onset",
            technique_candidates=(),
            onset_ms=100.0,
            midi_note=64,
            confidence=0.9,
        )
        with pytest.raises(FrozenInstanceError):
            snap.event_kind = "legato_transition"

    def test_technique_tuple_is_immutable_reference(self):
        snap = EventKindSnapshot(
            event_kind="pick_onset",
            technique_candidates=(
                TechniqueCandidate(kind="vibrato", confidence=0.8, subtype=None, target_cents=None, detected_cents=None, metrics={}),
            ),
            onset_ms=100.0,
            midi_note=64,
            confidence=0.9,
        )
        assert len(snap.technique_candidates) == 1

    def test_defaults(self):
        snap = EventKindSnapshot(
            event_kind="pick_onset",
            technique_candidates=(),
            onset_ms=0.0,
            midi_note=None,
            confidence=0.0,
        )
        assert snap.event_kind == "pick_onset"
        assert snap.technique_candidates == ()
        assert snap.onset_ms == 0.0
        assert snap.midi_note is None
        assert snap.confidence == 0.0


class TestStableNoteEventCarriesSnapshot:
    """StableNoteEvent must accept and expose an EventKindSnapshot."""

    def test_snapshot_is_accessible(self):
        snap = EventKindSnapshot(
            event_kind="pick_onset",
            technique_candidates=(),
            onset_ms=500.0,
            midi_note=40,
            confidence=0.85,
        )
        evt = StableNoteEvent(
            midi_note=40,
            frequency=82.41,
            confidence=0.85,
            name="E2",
            is_onset=True,
            onset_sample=24000,
            timestamp_ms=500.0,
            consensus_frames=5,
            event_snapshot=snap,
        )
        assert evt.event_snapshot is snap
        assert evt.event_snapshot.event_kind == "pick_onset"

    def test_snapshot_none_by_default(self):
        evt = StableNoteEvent(
            midi_note=40,
            frequency=82.41,
            confidence=0.85,
            name="E2",
            is_onset=True,
            onset_sample=24000,
            timestamp_ms=500.0,
            consensus_frames=5,
        )
        assert evt.event_snapshot is None

    def test_stable_note_event_is_frozen(self):
        evt = StableNoteEvent(
            midi_note=40,
            frequency=82.41,
            confidence=0.85,
            name="E2",
            is_onset=True,
            onset_sample=24000,
            timestamp_ms=500.0,
            consensus_frames=5,
        )
        with pytest.raises(FrozenInstanceError):
            evt.midi_note = 41


class TestEventKindSnapshotIsolation:
    """EventKindSnapshot must capture state independently of its source PerformanceEvent."""

    def test_snapshot_unchanged_after_performance_mutates(self):
        """Capture event_kind at emission time; mutating the original event
        must NOT change the snapshot's stored kind."""
        from pickhero.audio.performance import PerformanceEvent

        perf = PerformanceEvent(onset_ms=100.0, event_kind="pick_onset")
        snap = EventKindSnapshot(
            event_kind=perf.event_kind,
            technique_candidates=tuple(perf.technique_candidates),
            onset_ms=perf.onset_ms,
            midi_note=perf.midi_note,
            confidence=perf.confidence,
        )
        assert snap.event_kind == "pick_onset"

        perf.event_kind = "legato_transition"
        perf.midi_note = 64

        assert snap.event_kind == "pick_onset", \
            "Snapshot should not reflect post-emission mutations"
        assert snap.midi_note is None, \
            "Snapshot midi_note should still be None (was captured before set)"


class TestPitchCandidateSourceField:
    """PitchCandidate must carry a source field distinguishing detector origins."""

    def test_source_field_exists_and_readonly(self):
        pc = PitchCandidate(
            frequency=440.0,
            confidence=0.95,
            midi_note=69,
            source="aubio_yin",
            onset=True,
            onset_sample=48000,
        )
        assert pc.source == "aubio_yin"
        with pytest.raises(FrozenInstanceError):
            pc.source = "pitch_tracker"

    def test_multiple_detector_sources(self):
        pc_yin = PitchCandidate(440.0, 0.9, 69, "aubio_yin", False, None)
        pc_hybrid = PitchCandidate(440.0, 0.85, 69, "hybrid_f0", False, None)
        pc_chord = PitchCandidate(440.0, 0.8, 69, "chord_detector", False, None)
        assert pc_yin.source == "aubio_yin"
        assert pc_hybrid.source == "hybrid_f0"
        assert pc_chord.source == "chord_detector"
        assert pc_yin is not pc_hybrid
