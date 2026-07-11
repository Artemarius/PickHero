"""Integration tests for the architectural milestones.

Each test exercises a complete pipeline path, not a unit.
Stabilizer tests use synthetic DetectedNote frames synthesized with numpy.
AudioCapture tests mock sounddevice and feed synthetic audio.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from unittest import mock

import numpy as np
import pytest

aubio = pytest.importorskip("aubio")

from pickhero.audio.track_stabilizer import TrackStabilizer, StableNoteEvent
from pickhero.audio.detector import DetectedNote
from pickhero.audio.performance import PerformanceEvent
from pickhero.audio.event_types import EventKindSnapshot
from pickhero.audio.match_mode import MatchMode
from pickhero.audio.note_utils import midi_to_freq, midi_to_name
from pickhero.config import Config, apply_preset, JOSE_HIGH_ACCURACY_PRESET


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detected_note(
    midi: int,
    freq: float,
    confidence: float = 0.8,
    is_onset: bool = False,
    onset_sample: int | None = None,
    timestamp_ms: float = 0.0,
    performance: PerformanceEvent | None = None,
) -> DetectedNote:
    return DetectedNote(
        midi_note=midi,
        frequency=freq,
        confidence=confidence,
        name=midi_to_name(midi),
        is_onset=is_onset,
        onset_sample=onset_sample,
        performance=performance,
    )


_E2_FREQ = midi_to_freq(40)   # ~82.41 Hz
_E3_FREQ = midi_to_freq(52)   # ~164.81 Hz
_HOP_MS = 512 / 44100 * 1000  # ~11.6 ms per frame at the default hop


class _MockTimeInfo:
    """Stand-in for PortAudio's time_info struct."""

    def __init__(self, adc_time: float = 0.0):
        self.inputBufferAdcTime = adc_time
        self.currentTime = adc_time


# ---------------------------------------------------------------------------
# Stabilizer-level integration tests
# ---------------------------------------------------------------------------


class TestStabilizerIntegration:
    """Tests that exercise TrackStabilizer with synthetic DetectedNote frames."""

    def test_clean_e2_produces_one_stable_event(self):
        """A clean E2 played once produces exactly one StableNoteEvent with midi=40."""
        stab = TrackStabilizer(sample_rate=44100, hop_size=512, mode=MatchMode.ARCADE)
        f = _E2_FREQ

        # Onset frame alone should not emit yet
        ev = stab.process(
            _make_detected_note(40, f, confidence=0.9, is_onset=True, timestamp_ms=0),
            timestamp_ms=0,
        )
        assert len(ev) == 0, "onset alone should not emit"

        # Three sustain frames at the same pitch — establishes consensus
        all_events: list[StableNoteEvent] = []
        for i in range(1, 4):
            all_events.extend(
                stab.process(
                    _make_detected_note(40, f, confidence=0.9, timestamp_ms=i * _HOP_MS),
                    timestamp_ms=i * _HOP_MS,
                )
            )

        assert len(all_events) == 1, f"expected 1 event, got {len(all_events)}"
        assert all_events[0].midi_note == 40, f"expected midi=40, got {all_events[0].midi_note}"
        assert all_events[0].is_onset, "expected is_onset=True"
        assert all_events[0].confidence > 0.5
        assert all_events[0].consensus_frames >= 3

    def test_silence_produces_zero_events(self):
        """Feeding only None (silence) produces no events at all."""
        stab = TrackStabilizer(sample_rate=44100, hop_size=512, mode=MatchMode.ARCADE)

        for i in range(20):
            ev = stab.process(None, timestamp_ms=i * 10.0)
            assert len(ev) == 0

        # Even flush should produce nothing
        ev = stab.flush()
        assert len(ev) == 0

    def test_string_mute_does_not_match(self):
        """Very short burst with low confidence produces zero events.

        When confidence is below 0.3, the transient emission path is skipped
        and no consensus can be reached, so no event is emitted.
        """
        stab = TrackStabilizer(sample_rate=44100, hop_size=512, mode=MatchMode.ARCADE)
        f = _E2_FREQ

        # Onset with very low confidence — below transient threshold
        ev = stab.process(
            _make_detected_note(40, f, confidence=0.2, is_onset=True, timestamp_ms=0),
            timestamp_ms=0,
        )
        assert len(ev) == 0, "expected no events from low-confidence onset"

        # Send silence frames — the transient timeout path will fire but
        # won't emit because the onset confidence (< 0.3) falls below the
        # noise-floor threshold in transient emission logic.
        for i in range(1, 20):
            ev = stab.process(None, timestamp_ms=i * _HOP_MS)
            assert len(ev) == 0, (
                f"expected no events after silence timeout, got {len(ev)}"
            )

        assert stab._active is None, "stabilizer should have no active track"

    def test_e2_octave_glitch_resolves_to_e2(self):
        """Attack transient that locks onto E3 -> stabilizer resolves to E2 from sustain."""
        stab = TrackStabilizer(sample_rate=44100, hop_size=512, mode=MatchMode.ARCADE)
        e2 = _E2_FREQ
        e3 = _E3_FREQ

        # Onset at E3 (octave glitch — YIN locked onto the second harmonic)
        ev = stab.process(
            _make_detected_note(52, e3, confidence=0.8, is_onset=True, timestamp_ms=0),
            timestamp_ms=0,
        )
        assert len(ev) == 0

        # Three sustain frames at E2 — octave resolution kicks in
        all_events: list[StableNoteEvent] = []
        for i in range(1, 4):
            all_events.extend(
                stab.process(
                    _make_detected_note(40, e2, confidence=0.85, timestamp_ms=i * _HOP_MS),
                    timestamp_ms=i * _HOP_MS,
                )
            )

        assert len(all_events) == 1, f"expected 1 event, got {len(all_events)}"
        assert all_events[0].midi_note == 40, (
            f"expected E2 (midi=40), got {all_events[0].midi_note}"
        )
        assert abs(all_events[0].frequency - e2) < 1.0, (
            f"frequency off: {all_events[0].frequency} vs {e2}"
        )
        assert all_events[0].is_onset

    def test_legato_transition_survives_to_matcher(self):
        """A legato_transition PerformanceEvent preserves its event_kind
        through the stabilizer into the StableNoteEvent's event_snapshot.
        """
        stab = TrackStabilizer(sample_rate=44100, hop_size=512, mode=MatchMode.ARCADE)
        f = _E2_FREQ

        # PerformanceEvent tagged as legato_transition
        perf = PerformanceEvent(
            onset_ms=0.0,
            event_kind="legato_transition",
            midi_note=40,
            confidence=0.85,
        )

        # Onset with the performance reference
        ev = stab.process(
            _make_detected_note(
                40, f, confidence=0.85, is_onset=True, timestamp_ms=0, performance=perf
            ),
            timestamp_ms=0,
        )
        assert len(ev) == 0

        # Three sustain frames carrying the same performance object
        all_events: list[StableNoteEvent] = []
        for i in range(1, 4):
            all_events.extend(
                stab.process(
                    _make_detected_note(
                        40, f, confidence=0.85, timestamp_ms=i * _HOP_MS, performance=perf
                    ),
                    timestamp_ms=i * _HOP_MS,
                )
            )

        assert len(all_events) == 1, f"expected 1 event, got {len(all_events)}"
        snap = all_events[0].event_snapshot
        assert snap is not None, "expected event_snapshot to be populated"
        assert snap.event_kind == "legato_transition", (
            f"expected legato_transition, got {snap.event_kind}"
        )
        assert snap.midi_note == 40
        assert snap.confidence == 0.85

    def test_tab_prior_feeds_octave_resolution(self):
        """Tab context (tab_prior_midi=40) biases octave resolution to E2
        even when the onset harmonics suggest E3.
        """
        stab = TrackStabilizer(sample_rate=44100, hop_size=512, mode=MatchMode.ARCADE)
        e2 = _E2_FREQ
        e3 = _E3_FREQ

        # Onset at E3, but tab prior says this should be E2
        ev = stab.process(
            _make_detected_note(52, e3, confidence=0.8, is_onset=True, timestamp_ms=0),
            timestamp_ms=0,
            tab_prior_midi=40,
        )
        assert len(ev) == 0

        # Sustain frames at E2 — tab prior guides octave resolution
        all_events: list[StableNoteEvent] = []
        for i in range(1, 4):
            all_events.extend(
                stab.process(
                    _make_detected_note(40, e2, confidence=0.85, timestamp_ms=i * _HOP_MS),
                    timestamp_ms=i * _HOP_MS,
                    tab_prior_midi=40,
                )
            )

        assert len(all_events) == 1, f"expected 1 event, got {len(all_events)}"
        assert all_events[0].midi_note == 40, (
            f"expected E2 (midi=40) with tab_prior=40, got {all_events[0].midi_note}"
        )


# ---------------------------------------------------------------------------
# Audio-callback integration tests
# ---------------------------------------------------------------------------


class TestCallbackRealTimeSafety:
    """Verify the audio callback does no DSP work — only copies audio."""

    def test_callback_does_no_dsp(self):
        """The _audio_callback method must not call detector.process or any
        DSP function. All detection runs on the unified worker thread.
        """
        config = Config()
        config.audio.confidence_threshold = 0.3
        config.audio.noise_gate_db = -80.0
        config.audio.buf_size = 2048
        config.audio.hop_size = 512

        from pickhero.audio.input import AudioCapture

        capture = AudioCapture(config)

        # Patch detector.process to raise immediately if called.
        # The callback must NOT trigger this.
        dsp_mock = mock.MagicMock(side_effect=RuntimeError("DSP called from callback!"))
        with mock.patch.object(capture.detector, "process", dsp_mock):
            signal = np.zeros(2048, dtype=np.float32)
            indata = signal.reshape(-1, 1)
            # The callback only copies audio and enqueues blocks — no DSP.
            capture._audio_callback(indata, len(signal), _MockTimeInfo(0.0), 0)

            # detector.process must NOT have been called from the callback
            dsp_mock.assert_not_called()

        capture.stop()

    def test_callback_only_copies_to_ring_buffer(self):
        """The callback's only audio side-effect is writing to the ring buffer
        and enqueuing to the worker queue. Verify the ring advances.
        """
        config = Config()
        config.audio.confidence_threshold = 0.3
        config.audio.noise_gate_db = -80.0
        config.audio.buf_size = 2048
        config.audio.hop_size = 512

        from pickhero.audio.input import AudioCapture

        capture = AudioCapture(config)

        ring_before = int(np.sum(capture._audio_ring))

        signal = np.ones(2048, dtype=np.float32) * 0.5
        indata = signal.reshape(-1, 1)
        capture._audio_callback(indata, len(signal), _MockTimeInfo(0.0), 0)

        ring_after = int(np.sum(capture._audio_ring))
        # The ring should have absorbed the signal (nonzero entries)
        assert ring_after > ring_before, (
            "callback did not write signal to the audio ring"
        )

        capture.stop()


# ---------------------------------------------------------------------------
# Config preset tests
# ---------------------------------------------------------------------------


class TestConfigPreset:
    """Jose high-accuracy preset applies expected configuration values."""

    def test_jose_preset_applies(self):
        """Apply jose_high_accuracy preset and verify every field."""
        config = Config()
        apply_preset(config, "jose_high_accuracy")

        assert config.audio.profile == "high_accuracy"
        assert config.audio.sample_rate == 48000
        assert config.audio.hop_size == 256
        assert config.audio.buf_size == 4096
        assert config.match_mode == "judge"
        assert config.timing_judge_mode is True
        assert config.pitch_strict_mode is True
        assert config.offline_deep_analysis is True

        # Verify removals from the plan are absent
        assert "multi_label_techniques" not in config.preset_flags
        assert "after_take_analyzer" not in config.preset_flags
        assert "tone_profile_required" not in config.preset_flags

        # Known preset flags that should be present
        assert config.preset_flags.get("strict_chord_verification") is True
        assert config.preset_flags.get("chord_fft_size") == 16384


# ---------------------------------------------------------------------------
# Event immutability tests
# ---------------------------------------------------------------------------


class TestEventImmutability:
    """StableNoteEvent.event_snapshot must be unaffected by mutations to
    the original PerformanceEvent."""

    def test_event_snapshot_immutable(self):
        """Mutating a PerformanceEvent after snapshot creation must not
        alter the EventKindSnapshot."""
        from pickhero.audio.performance import TechniqueCandidate

        # Build a PerformanceEvent with known state
        perf = PerformanceEvent(
            onset_ms=100.0,
            event_kind="pick_onset",
            midi_note=40,
            confidence=0.9,
        )
        perf.technique_candidates.append(
            TechniqueCandidate(kind="normal", confidence=0.85)
        )

        # Create the immutable snapshot (as done in TrackStabilizer._try_emit)
        snap = EventKindSnapshot(
            event_kind=perf.event_kind,
            technique_candidates=tuple(perf.technique_candidates),
            onset_ms=perf.onset_ms,
            midi_note=perf.midi_note,
            confidence=perf.confidence,
        )

        # Mutate the original PerformanceEvent heavily
        perf.event_kind = "legato_transition"
        perf.midi_note = 52
        perf.confidence = 0.5
        perf.onset_ms = 999.0
        perf.technique_candidates.append(
            TechniqueCandidate(kind="bend", confidence=0.6)
        )

        # Snapshot must be completely unchanged
        assert snap.event_kind == "pick_onset", (
            f"expected pick_onset, got {snap.event_kind}"
        )
        assert snap.midi_note == 40, f"expected 40, got {snap.midi_note}"
        assert snap.confidence == 0.9, f"expected 0.9, got {snap.confidence}"
        assert snap.onset_ms == 100.0, f"expected 100.0, got {snap.onset_ms}"
        assert len(snap.technique_candidates) == 1, (
            f"expected 1 candidate, got {len(snap.technique_candidates)}"
        )
        assert snap.technique_candidates[0].kind == "normal"

    def test_stable_note_event_preserves_snapshot(self):
        """A StableNoteEvent created through the pipeline must carry an
        event_snapshot that reflects the PerformanceEvent at emission time,
        unaffected by subsequent mutations to the original.
        """
        stab = TrackStabilizer(sample_rate=44100, hop_size=512, mode=MatchMode.ARCADE)
        f = _E2_FREQ

        # Create a PerformanceEvent that will be referenced by all frames
        perf = PerformanceEvent(
            onset_ms=0.0,
            event_kind="pick_onset",
            midi_note=40,
            confidence=0.88,
        )

        all_events: list[StableNoteEvent] = []

        # Onset frame
        all_events.extend(
            stab.process(
                _make_detected_note(
                    40, f, confidence=0.88, is_onset=True, timestamp_ms=0, performance=perf
                ),
                timestamp_ms=0,
            )
        )

        # Three sustain frames — emission happens here (frame 3 of 4 total)
        for i in range(1, 4):
            all_events.extend(
                stab.process(
                    _make_detected_note(
                        40, f, confidence=0.88, timestamp_ms=i * _HOP_MS, performance=perf
                    ),
                    timestamp_ms=i * _HOP_MS,
                )
            )

        assert len(all_events) == 1, f"expected 1 event, got {len(all_events)}"

        # NOW mutate the original PerformanceEvent - after the snapshot was taken
        perf.event_kind = "legato_transition"
        perf.midi_note = 52
        perf.confidence = 0.5
        perf.onset_ms = 999.0

        # The snapshot must still reflect the state at emission time
        snap = all_events[0].event_snapshot
        assert snap is not None
        assert snap.event_kind == "pick_onset", (
            f"snapshot affected by mutation: {snap.event_kind}"
        )
        assert snap.midi_note == 40, f"snapshot midi affected: {snap.midi_note}"
        assert snap.confidence == 0.88, f"snapshot confidence affected: {snap.confidence}"
        assert snap.onset_ms == 0.0, f"snapshot onset_ms affected: {snap.onset_ms}"
