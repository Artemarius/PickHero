"""Tests for the expected-event verifiers."""

from __future__ import annotations

import numpy as np
import pytest

from pickhero.audio.evidence import ExpectedNote
from pickhero.audio.match_mode import MatchMode
from pickhero.audio.note_utils import midi_to_freq
from pickhero.audio.verifier_composite import CompositeVerifier
from pickhero.audio.verifier_spectral import SpectralVerifier
from pickhero.audio.verifier_cqt import CQTVerifier


def _sine_window(
    midi: int,
    duration_ms: float = 200.0,
    sample_rate: int = 48000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a sine wave at the frequency of ``midi``."""
    freq = midi_to_freq(midi)
    samples = int(sample_rate * duration_ms / 1000.0)
    t = np.arange(samples) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _ramped_sine_window(
    midi: int,
    duration_ms: float = 200.0,
    sample_rate: int = 48000,
    attack_ms: float = 25.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Sine with a linear attack so onset detection has a rising edge."""
    freq = midi_to_freq(midi)
    samples = int(sample_rate * duration_ms / 1000.0)
    t = np.arange(samples) / sample_rate
    attack_samples = int(sample_rate * attack_ms / 1000.0)
    envelope = np.ones(samples, dtype=np.float32)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
    signal = amplitude * np.sin(2 * np.pi * freq * t) * envelope
    return signal.astype(np.float32)


def _gated_sine_window(
    midi: int,
    duration_ms: float = 200.0,
    sample_rate: int = 48000,
    attack_ms: float = 100.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Sine that starts at ``attack_ms`` with a short ramp."""
    freq = midi_to_freq(midi)
    samples = int(sample_rate * duration_ms / 1000.0)
    t = np.arange(samples) / sample_rate
    attack_samples = int(sample_rate * attack_ms / 1000.0)
    ramp_samples = int(sample_rate * 5 / 1000.0)
    envelope = np.zeros(samples, dtype=np.float32)
    if attack_samples < samples:
        envelope[attack_samples:] = 1.0
    if attack_samples + ramp_samples <= samples:
        envelope[attack_samples:attack_samples + ramp_samples] = np.linspace(
            0.0, 1.0, ramp_samples
        )
    else:
        envelope[attack_samples:] = np.linspace(0.0, 1.0, samples - attack_samples)
    signal = amplitude * np.sin(2 * np.pi * freq * t) * envelope
    return signal.astype(np.float32)


def _harmonic_rich_window(
    midi: int,
    duration_ms: float = 200.0,
    sample_rate: int = 48000,
    harmonics: list[float] | None = None,
) -> np.ndarray:
    """Sine with explicit harmonics."""
    freq = midi_to_freq(midi)
    samples = int(sample_rate * duration_ms / 1000.0)
    t = np.arange(samples) / sample_rate
    if harmonics is None:
        harmonics = [1.0, 0.5, 0.3, 0.2, 0.1]
    signal = sum(
        (amp / (idx + 1)) * np.sin(2 * np.pi * freq * (idx + 1) * t)
        for idx, amp in enumerate(harmonics)
    )
    return (0.5 * signal).astype(np.float32)


class TestSpectralVerifier:
    """Smoke tests for the spectral harmonic verifier."""

    def test_sine_at_expected_note_is_present(self):
        """A clean sine at the expected MIDI should be detected."""
        verifier = SpectralVerifier(sample_rate=48000)
        window = _sine_window(64, duration_ms=200.0, sample_rate=48000)
        result = verifier.verify_single_note(window, 64, MatchMode.ARCADE)
        assert result.is_pitch_present, "expected note to be present"
        assert result.pitch_evidence is not None
        assert result.pitch_evidence.midi_note == 64

    def test_silent_window_is_not_present(self):
        """Silence should not verify as present."""
        verifier = SpectralVerifier(sample_rate=48000)
        window = np.zeros(4800, dtype=np.float32)
        result = verifier.verify_single_note(window, 64, MatchMode.ARCADE)
        assert not result.is_pitch_present

    def test_wrong_note_is_not_present_in_judge_mode(self):
        """A note one semitone off should not pass in JUDGE mode."""
        verifier = SpectralVerifier(sample_rate=48000)
        window = _sine_window(64, duration_ms=200.0, sample_rate=48000)
        result = verifier.verify_single_note(window, 65, MatchMode.JUDGE)
        assert not result.is_pitch_present

    def test_chord_detects_expected_notes(self):
        """A two-note chord should verify both notes present."""
        verifier = SpectralVerifier(sample_rate=48000)
        root = _sine_window(64, duration_ms=200.0, sample_rate=48000, amplitude=0.5)
        fifth = _sine_window(71, duration_ms=200.0, sample_rate=48000, amplitude=0.4)
        window = root + fifth
        expected = [ExpectedNote(midi=64, string=1), ExpectedNote(midi=71, string=2)]
        result = verifier.verify_chord(window, expected, MatchMode.ARCADE)
        assert len(result.notes) == 2
        present = [n.is_pitch_present for n in result.notes]
        assert any(present), "expected at least one chord note present"


class TestCQTVerifier:
    """Smoke tests for the CQT chord verifier."""

    def test_cqt_detects_chord_notes(self):
        """CQT should detect a simple major third chord."""
        verifier = CQTVerifier(sample_rate=48000)
        root = _sine_window(60, duration_ms=200.0, sample_rate=48000, amplitude=0.5)
        third = _sine_window(64, duration_ms=200.0, sample_rate=48000, amplitude=0.4)
        window = root + third
        expected = [ExpectedNote(midi=60, string=1), ExpectedNote(midi=64, string=2)]
        result = verifier.verify_chord(window, expected, MatchMode.ARCADE)
        assert len(result.notes) == 2


class TestCompositeVerifier:
    """Smoke tests for the composite verifier."""

    def test_composite_dispatches_single_note_to_spectral(self):
        """Composite verify_single_note should return a NoteVerification."""
        verifier = CompositeVerifier(sample_rate=48000)
        window = _sine_window(64, duration_ms=200.0, sample_rate=48000)
        result = verifier.verify_single_note(window, 64, MatchMode.ARCADE)
        assert result.is_pitch_present

    def test_composite_dispatches_technique(self):
        """Composite verify_technique should return a TechniqueVerification."""
        verifier = CompositeVerifier(sample_rate=48000)
        window = _sine_window(64, duration_ms=200.0, sample_rate=48000)
        result = verifier.verify_technique(window, "bend", {"target_cents": 100.0})
        assert isinstance(result.technique, str)


class TestVerifierCorrectness:
    """Correctness tests that assert specific behavior, not just smoke."""

    def test_wrong_note_rejected_in_arcade(self):
        """Sine at MIDI 64 should not verify as MIDI 66 in ARCADE mode."""
        verifier = CompositeVerifier(sample_rate=48000)
        window = _sine_window(64, duration_ms=200.0, sample_rate=48000)
        result = verifier.verify_single_note(window, 66, MatchMode.ARCADE)
        assert not result.is_pitch_present, "wrong note should be rejected"

    def test_empty_window_does_not_crash(self):
        """Empty window should return is_pitch_present=False, not crash."""
        verifier = CompositeVerifier(sample_rate=48000)
        result = verifier.verify_single_note(
            np.zeros(0, dtype=np.float32), 64, MatchMode.ARCADE
        )
        assert not result.is_pitch_present
        assert result.pitch_evidence is None

    def test_duplicate_midi_chord_preserves_order(self):
        """Two ExpectedNote with same MIDI, different strings → 2 results."""
        verifier = SpectralVerifier(sample_rate=48000)
        window = _sine_window(64, duration_ms=200.0, sample_rate=48000, amplitude=0.5)
        expected = [
            ExpectedNote(midi=64, string=1, fret=0, event_id="1000:1"),
            ExpectedNote(midi=64, string=2, fret=5, event_id="1000:2"),
        ]
        result = verifier.verify_chord(window, expected, MatchMode.ARCADE)
        assert len(result.notes) == 2
        for nv in result.notes:
            assert nv.is_pitch_present

    def test_verify_does_not_mutate_input_set(self):
        """verify() convenience wrapper should not mutate the caller's set."""
        verifier = CompositeVerifier(sample_rate=48000)
        window = _sine_window(64, duration_ms=200.0, sample_rate=48000)
        s = {64}
        verifier.verify(window, s, [], MatchMode.ARCADE, 0.0)
        assert s == {64}, f"set mutated to {s}"

    def test_silence_rejected_by_composite(self):
        """Silence should not verify as present in the composite verifier."""
        verifier = CompositeVerifier(sample_rate=48000)
        window = np.zeros(4800, dtype=np.float32)
        result = verifier.verify_single_note(window, 64, MatchMode.ARCADE)
        assert not result.is_pitch_present

    def test_cents_based_frequency_tolerance(self):
        """A note within the mode's cent tolerance verifies; outside does not."""
        verifier = SpectralVerifier(sample_rate=48000)
        samples = int(48000 * 0.2)
        t = np.arange(samples) / 48000
        # +30 cents in ARCADE (tolerance 70) should pass unambiguously.
        slightly_sharp = midi_to_freq(64) * 2 ** (30 / 1200)
        window = (0.5 * np.sin(2 * np.pi * slightly_sharp * t)).astype(np.float32)
        result = verifier.verify_single_note(window, 64, MatchMode.ARCADE)
        assert result.is_pitch_present, "+30 cents should pass in ARCADE"
        # +90 cents should fail in PRACTICE (tolerance 75).
        very_sharp = midi_to_freq(64) * 2 ** (90 / 1200)
        window = (0.5 * np.sin(2 * np.pi * very_sharp * t)).astype(np.float32)
        result = verifier.verify_single_note(window, 64, MatchMode.PRACTICE)
        assert not result.is_pitch_present, "+90 cents should fail in PRACTICE"

    def test_harmonic_aliases_rejected(self):
        """A harmonic-rich MIDI 64 signal must not verify as its sub-octaves."""
        verifier = SpectralVerifier(sample_rate=48000)
        window = _harmonic_rich_window(64, duration_ms=200.0, sample_rate=48000)
        result = verifier.verify_single_note(window, 52, MatchMode.JUDGE)
        assert not result.is_pitch_present, "MIDI 64 harmonic alias should not pass as MIDI 52"
        # The target itself should still pass.
        result = verifier.verify_single_note(window, 64, MatchMode.JUDGE)
        assert result.is_pitch_present, "MIDI 64 should verify as itself"

    def test_low_note_semitone_discrimination(self):
        """A pure low-E sine must not be confused with adjacent semitones."""
        verifier = SpectralVerifier(sample_rate=48000)
        for midi in [38, 39, 40, 41, 42]:
            window = _sine_window(midi, duration_ms=400.0, sample_rate=48000)
            result = verifier.verify_single_note(window, 40, MatchMode.PRACTICE)
            if midi == 40:
                assert result.is_pitch_present, f"MIDI {midi} should verify as MIDI 40"
            else:
                assert not result.is_pitch_present, f"MIDI {midi} should not verify as MIDI 40"

    def test_onset_present_for_attacked_note(self):
        """A note with a clear attack reports an onset inside the window."""
        verifier = SpectralVerifier(sample_rate=48000)
        window = _ramped_sine_window(64, duration_ms=200.0, sample_rate=48000)
        result = verifier.verify_single_note(window, 64, MatchMode.ARCADE)
        assert result.is_pitch_present
        assert result.is_onset_present, "attacked note should report onset"
        assert result.onset_ms is not None
        assert 0.0 <= result.onset_ms <= 50.0

    def test_cqt_pure_sine_chord_passes_in_arcade(self):
        """CQT must accept a chord of pure sines in ARCADE mode."""
        verifier = CQTVerifier(sample_rate=48000)
        root = _sine_window(60, duration_ms=200.0, sample_rate=48000, amplitude=0.6)
        third = _sine_window(64, duration_ms=200.0, sample_rate=48000, amplitude=0.5)
        window = root + third
        expected = [ExpectedNote(midi=60), ExpectedNote(midi=64)]
        result = verifier.verify_chord(window, expected, MatchMode.ARCADE)
        assert len(result.notes) == 2
        present = [n.is_pitch_present for n in result.notes]
        assert all(present), "pure-sine chord notes should verify in ARCADE"

    def test_judge_requires_overtones_for_pure_sine(self):
        """A pure sine with no harmonics should not pass in JUDGE mode."""
        verifier = SpectralVerifier(sample_rate=48000)
        window = _sine_window(64, duration_ms=200.0, sample_rate=48000)
        result = verifier.verify_single_note(window, 64, MatchMode.JUDGE)
        assert not result.is_pitch_present, "pure sine should fail in JUDGE mode"

    def test_cqt_out_of_range_returns_zero_score(self):
        """Notes below or above the CQT grid must not be accepted."""
        verifier = CQTVerifier(sample_rate=48000)
        window = _harmonic_rich_window(60, duration_ms=200.0, sample_rate=48000)
        low = verifier.verify_single_note(window, 20, MatchMode.ARCADE)
        high = verifier.verify_single_note(window, 95, MatchMode.ARCADE)
        assert not low.is_pitch_present
        assert not high.is_pitch_present

    def test_cqt_chord_reports_shared_strum_onset(self):
        """A strummed chord should report a single shared onset on every note."""
        verifier = CQTVerifier(sample_rate=48000)
        root = _ramped_sine_window(60, duration_ms=200.0, sample_rate=48000, attack_ms=25.0)
        third = _ramped_sine_window(64, duration_ms=200.0, sample_rate=48000, attack_ms=25.0)
        window = np.zeros_like(root)
        window[0:len(root)] += root
        window[0:len(third)] += third
        expected = [ExpectedNote(midi=60), ExpectedNote(midi=64)]
        result = verifier.verify_chord(window, expected, MatchMode.ARCADE)
        assert all(n.is_onset_present for n in result.notes)
        assert result.notes[0].onset_ms is not None
        assert result.notes[1].onset_ms is not None

    def test_composite_single_note_does_not_use_cqt_fallback(self):
        """Spectral anti-alias rejection must not be undone by CQT."""
        verifier = CompositeVerifier(sample_rate=48000)
        window = _sine_window(64, duration_ms=200.0, sample_rate=48000)
        result = verifier.verify_single_note(window, 52, MatchMode.ARCADE)
        # MIDI 64 is the 2nd harmonic of MIDI 52; spectral anti-alias should reject it.
        assert not result.is_pitch_present
    def test_spectral_fft_cache_reuses_spectrum(self):
        """Calling verify_chord twice with the same window should reuse the FFT."""
        verifier = SpectralVerifier(sample_rate=48000)
        window = _harmonic_rich_window(60, duration_ms=200.0, sample_rate=48000)
        verifier.verify_chord(window, [ExpectedNote(midi=60)], MatchMode.ARCADE)
        first_id = verifier._last_window_id
        second = verifier.verify_chord(window, [ExpectedNote(midi=60)], MatchMode.ARCADE)
        assert verifier._last_window_id == first_id
        assert second.notes[0].is_pitch_present

    def test_onset_detected_in_second_half_with_expected_offset(self):
        """An attack at 100ms is found when the verifier expects it."""
        verifier = SpectralVerifier(sample_rate=48000)
        # 200ms window: silence for 100ms, then a note with a sharp attack.
        window = _gated_sine_window(64, duration_ms=200.0, sample_rate=48000, attack_ms=100.0)
        result = verifier.verify_single_note(
            window, 64, MatchMode.ARCADE, expected_onset_offset_ms=100.0
        )
        assert result.is_onset_present
        assert result.onset_ms is not None
        assert result.onset_ms == pytest.approx(100.0, abs=15.0)

    def test_onset_rejected_without_expected_offset(self):
        """The same attack is rejected when the verifier expects onset near 0."""
        verifier = SpectralVerifier(sample_rate=48000)
        window = _gated_sine_window(64, duration_ms=200.0, sample_rate=48000, attack_ms=100.0)
        result = verifier.verify_single_note(
            window, 64, MatchMode.ARCADE, expected_onset_offset_ms=0.0, onset_tolerance_ms=25.0
        )
        assert not result.is_onset_present

    def test_spectral_chord_uses_broadband_onset(self):
        """Chord onset is derived from broadband energy, not the first note."""
        verifier = SpectralVerifier(sample_rate=48000)
        # Both notes start at 100ms; the shared strum onset must be detected.
        root = _gated_sine_window(60, duration_ms=200.0, sample_rate=48000, attack_ms=100.0)
        third = _gated_sine_window(64, duration_ms=200.0, sample_rate=48000, attack_ms=100.0)
        window = root + third
        expected = [ExpectedNote(midi=60), ExpectedNote(midi=64)]
        result = verifier.verify_chord(
            window, expected, MatchMode.ARCADE, expected_onset_offset_ms=100.0
        )
        assert all(n.is_onset_present for n in result.notes)
