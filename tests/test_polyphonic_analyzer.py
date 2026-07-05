"""Tests for pickhero.audio.polyphonic_analyzer — offline deep analysis.

Synthesizes raw audio + matched (event, note) pairs and verifies the
PolyphonicAnalyzer emits the expected verdicts for unison bends and rejects
false positives (single-F0, power chords).
"""

from __future__ import annotations

import numpy as np
import pytest

from pickhero.audio.performance import (
    PerformanceEvent,
    TechniqueSpec,
    TechniqueVerdict,
)
from pickhero.audio.polyphonic_analyzer import PolyphonicAnalyzer
from pickhero.tabs.timeline import NoteEvent


_SR = 48000


def _midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _note(timestamp_ms: float, midi: int, string: int, fret: int,
          techniques=()) -> NoteEvent:
    return NoteEvent(
        timestamp_ms=timestamp_ms, duration_ms=600.0,
        midi_note=midi, string=string, fret=fret, techniques=techniques,
    )


def _bend_event(onset_ms: float, release_ms: float, start_freq: float,
                end_freq: float) -> PerformanceEvent:
    """Build an event whose f0_curve ramps from start_freq to end_freq."""
    n = 50
    times = np.linspace(onset_ms, release_ms, n)
    freqs = np.linspace(start_freq, end_freq, n)
    f0_curve = [(float(t), float(f), 0.0) for t, f in zip(times, freqs)]
    return PerformanceEvent(
        onset_ms=onset_ms, release_ms=release_ms,
        f0_curve=f0_curve, midi_note=int(round(69 + 12 * np.log2(start_freq / 440.0))),
    )


class TestUnisonBendDetection:
    """Unison bend: a bent note + a simultaneously-sounding static note."""

    def test_unison_bend_fixture_requires_two_f0s(self):
        """D3 (146.83 Hz) static + a D3 bent to E3 (164.81 Hz) over 500ms →
        unison_bend verdict with time_to_unison_ms."""
        sr = _SR
        duration_ms = 600.0
        n = int(sr * duration_ms / 1000.0)
        t = np.arange(n) / sr
        # Static D3 sine
        static_freq = 146.83
        static = 0.4 * np.sin(2 * np.pi * static_freq * t)
        # Bent note: D3 → E3 over the first 500ms, then holds at E3
        bend_start = 146.83
        bend_end = 164.81
        bend_prog = np.minimum(t / 0.5, 1.0)
        bent_freq = bend_start + (bend_end - bend_start) * bend_prog
        bent_phase = 2 * np.pi * np.cumsum(bent_freq) / sr
        bent = 0.4 * np.sin(bent_phase)
        audio = (static + bent).astype(np.float32)

        static_note = _note(0.0, 50, 5, 0)  # D3, no bend
        bend_note = _note(0.0, 50, 4, 0,
                          techniques=(TechniqueSpec(kind="bend", target_cents=200.0),))
        bend_ev = _bend_event(0.0, duration_ms, bend_start, bend_end)
        static_ev = PerformanceEvent(onset_ms=0.0, release_ms=duration_ms,
                                     midi_note=50)
        pairs = [(bend_ev, bend_note), (static_ev, static_note)]

        analyzer = PolyphonicAnalyzer(audio, sr, pairs)
        verdicts = analyzer.analyze()
        unison = [v for v in verdicts if v.kind == "unison_bend"]
        assert unison, "expected a unison_bend verdict for two-F0 overlap"
        v = unison[0]
        assert "time_to_unison_ms" in v.metrics
        assert v.metrics["time_to_unison_ms"] >= 0.0

    def test_unison_bend_rejects_single_f0(self):
        """A single bent note (no static partner) → no unison_bend verdict."""
        sr = _SR
        duration_ms = 500.0
        n = int(sr * duration_ms / 1000.0)
        t = np.arange(n) / sr
        bend_start = 146.83
        bend_end = 164.81
        bend_prog = np.minimum(t / 0.4, 1.0)
        bent_freq = bend_start + (bend_end - bend_start) * bend_prog
        bent_phase = 2 * np.pi * np.cumsum(bent_freq) / sr
        audio = (0.4 * np.sin(bent_phase)).astype(np.float32)

        bend_note = _note(0.0, 50, 4, 0,
                          techniques=(TechniqueSpec(kind="bend", target_cents=200.0),))
        bend_ev = _bend_event(0.0, duration_ms, bend_start, bend_end)
        pairs = [(bend_ev, bend_note)]

        analyzer = PolyphonicAnalyzer(audio, sr, pairs)
        verdicts = analyzer.analyze()
        unison = [v for v in verdicts if v.kind == "unison_bend"]
        assert not unison, "single-F0 bent note must not produce a unison_bend verdict"


class TestPinchHarmonicRejection:
    """A power chord (root + fifth) must not trigger a pinch-harmonic verdict."""

    def test_power_chord_does_not_trigger_harmonic(self):
        sr = _SR
        duration_ms = 400.0
        n = int(sr * duration_ms / 1000.0)
        t = np.arange(n) / sr
        root = 82.41   # E2
        fifth = 123.47  # B2
        audio = (0.4 * np.sin(2 * np.pi * root * t)
                 + 0.4 * np.sin(2 * np.pi * fifth * t)).astype(np.float32)

        # Notes with pinch harmonic spec — but the audio is a clean power chord.
        root_note = _note(0.0, 40, 6, 0,
                          techniques=(TechniqueSpec(
                              kind="harmonic", subtype="pinch",
                              expected_sounding_midi=64,
                          ),))
        root_ev = PerformanceEvent(
            onset_ms=0.0, release_ms=duration_ms, midi_note=40,
            f0_curve=[(0.0, root, 0.0)],
        )
        pairs = [(root_ev, root_note)]

        analyzer = PolyphonicAnalyzer(audio, sr, pairs)
        verdicts = analyzer.analyze()
        pinch = [v for v in verdicts if v.kind == "pinch_harmonic"]
        # A clean power chord has strong fundamental, weak overtones → should
        # NOT produce a "good" pinch verdict (overtone ratio < 2.0).
        good_pinch = [v for v in pinch if v.grade == "good"]
        assert not good_pinch, (
            "power chord must not produce a good pinch_harmonic verdict"
        )
