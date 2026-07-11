"""Tests for pickhero.audio.analyzer — synthetic f0_curve → verdict grading.

Each test constructs a synthetic ``PerformanceEvent`` with a carefully-crafted
f0_curve (and supporting fields) and a matching ``NoteEvent`` with the
appropriate ``TechniqueSpec``.  The test exercises
``PerformanceAnalyzer().analyze()`` and asserts that the resulting
``TechniqueVerdict`` has ``grade == 'good'`` and an explanation containing
the expected coaching substring.
"""

from __future__ import annotations

import numpy as np
import pytest

from pickhero.audio.analyzer import PerformanceAnalyzer
from pickhero.audio.performance import (
    PerformanceEvent,
    TechniqueCandidate,
    TechniqueSpec,
)
from pickhero.tabs.timeline import NoteEvent


# Standard tuning: E2 = 82.41 Hz
_E2 = 82.41


def _hertz(base: float, cents: float) -> float:
    return base * 2.0 ** (cents / 1200.0)


# ─── 1. BendJudge ───────────────────────────────────────────────────────────

class TestBendJudge:
    def test_good_reaches_target(self):
        """100-cent monotonic rise → ``good``, explanation contains target."""
        n = 20
        times = np.linspace(0, 200, n)
        cents = np.minimum(times / 2.0, 100.0)

        f0_curve = [(t, _hertz(_E2, c), float(c)) for t, c in zip(times, cents)]
        energy = [(t, 0.5) for t in times]

        event = PerformanceEvent(
            onset_ms=0.0,
            f0_curve=f0_curve,
            energy_envelope=energy,
            midi_note=40,
        )
        note = NoteEvent(
            timestamp_ms=0, duration_ms=300, midi_note=40, string=6, fret=0,
            techniques=(TechniqueSpec(kind="bend", target_cents=100.0),),
        )

        results = PerformanceAnalyzer().analyze([(event, note)])
        v = results[0].verdicts[0]
        assert v.grade == "good"
        assert "reached 100 of 100 cents" in v.explanation


# ─── 2. VibratoJudge ────────────────────────────────────────────────────────

class TestVibratoJudge:
    def test_good_clean_5hz(self):
        """5 Hz, ±40 cents over ~464 ms → ``good`` (rate 4-8 Hz, depth 30-80 ¢)."""
        frame_ms = 11.6
        n_frames = 40
        freq_hz = 5.0
        amplitude = 40.0  # ±40 cents → depth 40 (inside 30-80 band)

        f0_curve: list[tuple[float, float, float]] = []
        for i in range(n_frames):
            t = i * frame_ms
            cents_offset = amplitude * np.sin(2 * np.pi * freq_hz * t / 1000.0)
            f0_curve.append((t, _hertz(_E2, cents_offset), float(cents_offset)))

        energy = [(t, 0.5) for t, _, _ in f0_curve]
        event = PerformanceEvent(
            onset_ms=0.0,
            f0_curve=f0_curve,
            energy_envelope=energy,
        )
        note = NoteEvent(
            timestamp_ms=0, duration_ms=500, midi_note=40, string=6, fret=0,
            techniques=(TechniqueSpec(kind="vibrato"),),
        )

        results = PerformanceAnalyzer().analyze([(event, note)])
        v = results[0].verdicts[0]
        assert v.grade == "good"


# ─── 3. SlideJudge ──────────────────────────────────────────────────────────

class TestSlideJudge:
    def test_good_landing_on_target(self):
        """Short rise + long fall to +210 ¢ → fall wins → ``good``.

        The SlideJudge picks the longest monotonic segment. A short rise (10
        frames, 0→200 ¢) followed by a long fall (40 frames, 200→210 ¢) means
        the fall segment is selected, and its landing value (+210 ¢) is within
        ±15 of the target (200 ¢ for a 0→2 fret slide) → ``good``.
        """
        n_rise = 10
        n_fall = 41  # must be significantly longer than n_rise
        rise_cents = np.linspace(0, 200, n_rise)
        fall_cents = np.linspace(200, 210, n_fall)
        # Drop the shared peak to avoid a zero-delta frame
        all_cents = np.concatenate([rise_cents, fall_cents[1:]])
        n = len(all_cents)
        times = np.linspace(0, 500, n)

        f0_curve = [(t, _hertz(_E2, c), float(c)) for t, c in zip(times, all_cents)]
        energy = [(t, 0.5) for t in times]

        event = PerformanceEvent(
            onset_ms=0.0,
            f0_curve=f0_curve,
            energy_envelope=energy,
        )
        note = NoteEvent(
            timestamp_ms=0, duration_ms=500, midi_note=40, string=6, fret=0,
            techniques=(TechniqueSpec(kind="slide", end_fret=2),),
        )

        results = PerformanceAnalyzer().analyze([(event, note)])
        v = results[0].verdicts[0]
        assert v.grade == "good"
        assert "of fret 2" in v.explanation


# ─── 4. LegatoJudge (hammer_on / pull_off) ──────────────────────────────────

class TestLegatoJudge:
    def test_good_no_pick(self):
        """Low pick transient + 0.5 volume + stable pitch → ``good``."""
        times = np.linspace(0, 100, 10)
        cents = np.zeros(10)  # perfectly stable

        f0_curve = [(t, _hertz(_E2, c), float(c)) for t, c in zip(times, cents)]
        energy = [(t, 0.3) for t in times]

        event = PerformanceEvent(
            onset_ms=0.0,
            f0_curve=f0_curve,
            energy_envelope=energy,
            onset_features={
                "pick_transient_strength": 0.2,
                "hammer_volume_ratio": 0.5,
                "transition_ms": 30.0,
            },
            midi_note=42,
        )
        note = NoteEvent(
            timestamp_ms=0, duration_ms=100, midi_note=42, string=5, fret=2,
            techniques=(TechniqueSpec(kind="hammer_on", tied_to_previous=True),),
        )

        results = PerformanceAnalyzer().analyze([(event, note)])
        v = results[0].verdicts[0]
        assert v.grade == "good"
        assert "stable" in v.explanation


# ─── 5. PalmMuteJudge ───────────────────────────────────────────────────────

class TestPalmMuteJudge:
    def test_good_short_halflife(self):
        """Short halflife (50 ms) + low centroid → ``good``, ``well-controlled``."""
        times = np.linspace(0, 200, 20)
        f0_curve = [(t, _hertz(_E2, 0.0), 0.0) for t in times]
        energy = [(t, 0.5) for t in times]

        pm_candidate = TechniqueCandidate(
            kind="palm_mute",
            confidence=0.9,
            metrics={"decay_halflife_ms": 50.0, "centroid_hz": 800.0},
        )

        event = PerformanceEvent(
            onset_ms=0.0,
            f0_curve=f0_curve,
            energy_envelope=energy,
            technique_candidates=[pm_candidate],
        )
        note = NoteEvent(
            timestamp_ms=0, duration_ms=200, midi_note=40, string=6, fret=0,
            techniques=(TechniqueSpec(kind="palm_mute"),),
        )

        results = PerformanceAnalyzer().analyze([(event, note)])
        v = results[0].verdicts[0]
        assert v.grade == "good"
        assert "well-controlled" in v.explanation


# ─── 6. HarmonicJudge ───────────────────────────────────────────────────────

class TestHarmonicJudge:
    def test_good_natural_octave(self):
        """f0 = 2× expected (12th-fret octave), hnr 0.8 → ``good``."""
        # fret=12: ratio=2.0; midi=40 → base=82.41 Hz; expected harmonic=E3=164.81
        E3 = 164.81

        f0_curve = [(0.0, E3, 0.0)]
        spectral = [
            {"centroid": 800.0, "flux": 0.1, "flatness": 0.0, "hnr": 0.8},
        ]

        event = PerformanceEvent(
            onset_ms=0.0,
            f0_curve=f0_curve,
            spectral_features=spectral,
        )
        note = NoteEvent(
            timestamp_ms=0, duration_ms=200, midi_note=40, string=6, fret=12,
            techniques=(TechniqueSpec(kind="harmonic", subtype="natural"),),
        )

        results = PerformanceAnalyzer().analyze([(event, note)])
        v = results[0].verdicts[0]
        assert v.grade == "good"
        assert "clear and in tune" in v.explanation


# ─── 7. DeadNoteJudge ───────────────────────────────────────────────────────

class TestDeadNoteJudge:
    def test_good_no_pitch(self):
        """Noise burst 0.8, empty f0 → ``good``, ``struck cleanly``."""
        event = PerformanceEvent(
            onset_ms=0.0,
            f0_curve=[],  # no pitched sustain
            onset_features={"noise_burst": 0.8},
        )
        note = NoteEvent(
            timestamp_ms=0, duration_ms=100, midi_note=0, string=1, fret=0,
            techniques=(TechniqueSpec(kind="dead_note"),),
        )

        results = PerformanceAnalyzer().analyze([(event, note)])
        v = results[0].verdicts[0]
        assert v.grade == "good"
        assert "struck cleanly" in v.explanation


# ─── Patch 4: new-behavior + regression tests ─────────────────────────────────


class TestPalmMuteLabelInversion:
    """Patch 4a: palm-mute labels were inverted. Very short halflife (very dead)
    must say 'too dead'; long sustain (open) must say 'too open'."""

    def test_very_short_halflife_is_too_dead(self):
        """halflife 10ms → tightness 0.95 (>0.9) → 'too dead — lighten muting'."""
        times = np.linspace(0, 200, 20)
        f0_curve = [(t, _hertz(_E2, 0.0), 0.0) for t in times]
        energy = [(t, 0.5) for t in times]
        pm_candidate = TechniqueCandidate(
            kind="palm_mute",
            confidence=0.9,
            metrics={"decay_halflife_ms": 10.0, "centroid_hz": 800.0},
        )
        event = PerformanceEvent(
            onset_ms=0.0, f0_curve=f0_curve, energy_envelope=energy,
            technique_candidates=[pm_candidate],
        )
        note = NoteEvent(
            timestamp_ms=0, duration_ms=200, midi_note=40, string=6, fret=0,
            techniques=(TechniqueSpec(kind="palm_mute"),),
        )
        results = PerformanceAnalyzer().analyze([(event, note)])
        v = results[0].verdicts[0]
        assert v.grade == "weak"
        assert "too dead" in v.explanation, f"expected 'too dead', got: {v.explanation}"

    def test_open_sustained_is_too_open(self):
        """halflife 250ms → tightness -0.25 (<0.4) → 'too open — mute more'."""
        times = np.linspace(0, 200, 20)
        f0_curve = [(t, _hertz(_E2, 0.0), 0.0) for t in times]
        energy = [(t, 0.5) for t in times]
        pm_candidate = TechniqueCandidate(
            kind="palm_mute",
            confidence=0.9,
            metrics={"decay_halflife_ms": 250.0, "centroid_hz": 800.0},
        )
        event = PerformanceEvent(
            onset_ms=0.0, f0_curve=f0_curve, energy_envelope=energy,
            technique_candidates=[pm_candidate],
        )
        note = NoteEvent(
            timestamp_ms=0, duration_ms=200, midi_note=40, string=6, fret=0,
            techniques=(TechniqueSpec(kind="palm_mute"),),
        )
        results = PerformanceAnalyzer().analyze([(event, note)])
        v = results[0].verdicts[0]
        assert v.grade == "weak"
        assert "too open" in v.explanation, f"expected 'too open', got: {v.explanation}"


class TestDeadNotePitchDetection:
    """Patch 4b: dead-note must detect pitch by frequency, not cents. A stable
    pitched note (cents==0) must FAIL dead-note grading."""

    def test_stable_pitch_must_fail_dead_note(self):
        """Constant 82.4 Hz (cents 0.0), noise_burst 0.8 → grade 'missed'."""
        times = np.linspace(0, 100, 10)
        # Stable pitch at 82.4 Hz, cents 0.0 relative to base
        f0_curve = [(t, 82.4, 0.0) for t in times]
        event = PerformanceEvent(
            onset_ms=0.0,
            f0_curve=f0_curve,
            onset_features={"noise_burst": 0.8},
        )
        note = NoteEvent(
            timestamp_ms=0, duration_ms=100, midi_note=0, string=1, fret=0,
            techniques=(TechniqueSpec(kind="dead_note"),),
        )
        results = PerformanceAnalyzer().analyze([(event, note)])
        v = results[0].verdicts[0]
        assert v.grade == "missed", (
            f"stable pitched note should fail dead-note (missed), got {v.grade}"
        )


class TestSlideTargetGrading:
    """Patch 4c: SlideJudge grades against spec.target_cents, not zero."""

    def test_slide_5_to_7_must_fail_if_landing_short(self):
        """target_cents=200 (5→7), landing at +50 → 150 cents off → 'weak'."""
        n_rise = 10
        rise_cents = np.linspace(0, 50, n_rise)
        times = np.linspace(0, 200, n_rise)
        f0_curve = [(t, _hertz(_E2, c), float(c)) for t, c in zip(times, rise_cents)]
        energy = [(t, 0.5) for t in times]
        event = PerformanceEvent(
            onset_ms=0.0, f0_curve=f0_curve, energy_envelope=energy,
        )
        note = NoteEvent(
            timestamp_ms=0, duration_ms=200, midi_note=40, string=6, fret=5,
            techniques=(TechniqueSpec(
                kind="slide", start_fret=5, end_fret=7, target_cents=200.0,
            ),),
        )
        results = PerformanceAnalyzer().analyze([(event, note)])
        v = results[0].verdicts[0]
        assert v.grade == "weak", (
            f"landing +50 vs target +200 should be weak, got {v.grade}"
        )

    def test_slide_with_target_on_land_is_good(self):
        """target_cents=200, landing at +205 → 5 cents off → 'good'."""
        n_rise = 10
        rise_cents = np.linspace(0, 205, n_rise)
        times = np.linspace(0, 200, n_rise)
        f0_curve = [(t, _hertz(_E2, c), float(c)) for t, c in zip(times, rise_cents)]
        energy = [(t, 0.5) for t in times]
        event = PerformanceEvent(
            onset_ms=0.0, f0_curve=f0_curve, energy_envelope=energy,
        )
        note = NoteEvent(
            timestamp_ms=0, duration_ms=200, midi_note=40, string=6, fret=5,
            techniques=(TechniqueSpec(
                kind="slide", start_fret=5, end_fret=7, target_cents=200.0,
            ),),
        )
        results = PerformanceAnalyzer().analyze([(event, note)])
        v = results[0].verdicts[0]
        assert v.grade == "good"


class TestHarmonicOpenStringPitch:
    """Patch 4d: HarmonicJudge uses spec.expected_sounding_midi (open string +
    node ratio), not the fretted midi × ratio."""

    def test_natural_harmonic_uses_open_string_not_fretted(self):
        """fret=12, string=6 (low E): expected_freq ≈ 164.81 Hz (open E2 × 2),
        NOT _midi_to_freq(40) × 2 (which is the same here, but with
        expected_sounding_midi=52 the judge must use that directly)."""
        # string=6 low E: open midi = 40 (E2). 12th-fret natural harmonic
        # sounds at E3 = midi 52. expected_freq = _midi_to_freq(52) = 164.81.
        E3 = 164.81
        f0_curve = [(0.0, E3, 0.0)]
        spectral = [{"centroid": 800.0, "flux": 0.1, "flatness": 0.0, "hnr": 0.8}]
        event = PerformanceEvent(
            onset_ms=0.0, f0_curve=f0_curve, spectral_features=spectral,
        )
        # Realistic spec with expected_sounding_midi populated by the loader.
        note = NoteEvent(
            timestamp_ms=0, duration_ms=200, midi_note=40, string=6, fret=12,
            techniques=(TechniqueSpec(
                kind="harmonic", subtype="natural",
                expected_sounding_midi=52, node_fret=12,
            ),),
        )
        results = PerformanceAnalyzer().analyze([(event, note)])
        v = results[0].verdicts[0]
        assert v.grade == "good"
        assert abs(v.metrics["expected_freq"] - 164.81) < 0.5, (
            f"expected_freq≈164.81 (open E2×2), got {v.metrics['expected_freq']}"
        )


class TestUnexpectedTechniqueJudge:
    """Patch 4f: a candidate present but not expected by any spec emits a
    'weak' unexpected penalty."""

    def test_normal_note_palm_muted_emits_penalty(self):
        """Empty techniques + palm_mute candidate → verdict kind='unexpected',
        grade='weak'."""
        times = np.linspace(0, 200, 20)
        f0_curve = [(t, _hertz(_E2, 0.0), 0.0) for t in times]
        energy = [(t, 0.5) for t in times]
        pm_candidate = TechniqueCandidate(
            kind="palm_mute",
            confidence=0.9,
            metrics={"decay_halflife_ms": 50.0, "centroid_hz": 800.0},
        )
        event = PerformanceEvent(
            onset_ms=0.0, f0_curve=f0_curve, energy_envelope=energy,
            technique_candidates=[pm_candidate],
        )
        # Normal note: NO techniques expected.
        note = NoteEvent(
            timestamp_ms=0, duration_ms=200, midi_note=40, string=6, fret=0,
            techniques=(),
        )
        results = PerformanceAnalyzer().analyze([(event, note)])
        unexpected = [v for v in results[0].verdicts if v.kind == "unexpected"]
        assert unexpected, "expected an 'unexpected' verdict for palm_mute on a clean note"
        assert unexpected[0].grade == "weak"
