"""Regression corpus smoke tests.

These tests run against the dataset cache (real datasets are not bundled).
If no cache is present, a small synthetic corpus is generated in a temporary
cache so the test suite remains self-contained in CI.
"""

from __future__ import annotations

import tempfile

import numpy as np

from pickhero.audio.evidence import ExpectedNote
from pickhero.audio.match_mode import MatchMode
from pickhero.audio.note_utils import midi_to_freq
from pickhero.audio.verifier_composite import CompositeVerifier
from pickhero.datasets import ClipEvent, DatasetRegistry
from pickhero.datasets.schema import ClipExpectedNote


def _synthetic_corpus(registry: DatasetRegistry) -> None:
    """Write a tiny synthetic corpus to the registry cache."""
    events = [
        ClipEvent(
            clip_id="synthetic/c4/0",
            source="Synthetic",
            start_s=0.0,
            end_s=0.2,
            midi=60,
            technique="normal",
            confidence=1.0,
            audio_path="/tmp/synthetic_c4.wav",
            notes=(),
            string=3,
            fret=3,
        ),
        ClipEvent(
            clip_id="synthetic/e4/0",
            source="Synthetic",
            start_s=0.0,
            end_s=0.2,
            midi=64,
            technique="normal",
            confidence=1.0,
            audio_path="/tmp/synthetic_e4.wav",
            notes=(),
            string=2,
            fret=0,
        ),
        ClipEvent(
            clip_id="synthetic/chord/0",
            source="Synthetic",
            start_s=0.0,
            end_s=0.2,
            midi=None,
            technique="normal",
            confidence=1.0,
            audio_path="/tmp/synthetic_chord.wav",
            notes=(
                ClipExpectedNote(midi=60),
                ClipExpectedNote(midi=64),
            ),
            string=None,
            fret=None,
        ),
    ]
    registry._write_cache(events)


def _window_for_midi(
    midi: int, duration_ms: float, sample_rate: int, harmonics: list[float] | None = None
) -> np.ndarray:
    """Generate a sine-like window for a given MIDI note with optional harmonics."""
    freq = midi_to_freq(midi)
    samples = int(sample_rate * duration_ms / 1000.0)
    t = np.arange(samples) / sample_rate
    if harmonics is None:
        harmonics = [1.0]
    signal = sum(
        (amp / (idx + 1)) * np.sin(2 * np.pi * freq * (idx + 1) * t)
        for idx, amp in enumerate(harmonics)
    )
    return (0.5 * signal).astype(np.float32)


class TestCorpusVerifier:
    """Smoke-test the verifier against the synthetic corpus."""

    def test_synthetic_single_notes_verify(self):
        """Single-note synthetic events should verify as present."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = DatasetRegistry(cache_dir=tmp)
            _synthetic_corpus(registry)
            events = registry.load_events({"source": "Synthetic"})
            verifier = CompositeVerifier(sample_rate=48000)
            single_notes = [e for e in events if e.midi is not None]
            assert len(single_notes) >= 2
            for event in single_notes:
                window = _window_for_midi(event.midi, 200.0, 48000, harmonics=[1.0, 0.5, 0.3])
                result = verifier.verify_single_note(window, event.midi, MatchMode.ARCADE)
                assert result.is_pitch_present, f"{event.clip_id} should verify"

    def test_synthetic_chord_verifies_notes(self):
        """Synthetic chord events should verify constituent notes."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = DatasetRegistry(cache_dir=tmp)
            _synthetic_corpus(registry)
            events = registry.load_events({"source": "Synthetic"})
            chords = [e for e in events if e.midi_notes is not None]
            assert len(chords) >= 1
            verifier = CompositeVerifier(sample_rate=48000)
            for event in chords:
                window = sum(
                    _window_for_midi(m, 200.0, 48000, harmonics=[1.0, 0.5, 0.3])
                    for m in event.midi_notes
                )
                expected_notes = [
                    ExpectedNote(midi=m, event_id=f"corpus:{i}")
                    for i, m in enumerate(sorted(event.midi_notes or ()))
                ]
                result = verifier.verify_chord(window, expected_notes, MatchMode.ARCADE)
                assert len(result.notes) == len(event.midi_notes)
                present = sum(1 for n in result.notes if n.is_pitch_present)
                assert present >= len(event.midi_notes) / 2, "at least half chord notes present"
