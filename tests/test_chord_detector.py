"""Tests for pickhero.audio.chord_detector.

Generates synthetic audio signals (sine waves, summed sines, white noise)
at 48 kHz and verifies ChordDetector correctly identifies or rejects
power-chord note sets. No audio hardware required.
"""

from __future__ import annotations

import numpy as np
import pytest

from pickhero.audio.chord_detector import ChordDetector
from pickhero.audio.note_utils import midi_to_freq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48000
FFT_SIZE = 16384
HOP_SIZE = 512  # push_audio chunk size used in the real pipeline

# MIDI note numbers used throughout the tests
E2 = 40  # low E open   ~82.41 Hz
B2 = 47  # perfect fifth above E2  ~123.47 Hz
D3 = 50  # unrelated pitch  ~146.83 Hz


# ---------------------------------------------------------------------------
# Synthetic signal helpers
# ---------------------------------------------------------------------------

def generate_sine(
    freq: float,
    duration_s: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Return a pure sine wave as float32."""
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def generate_white_noise(
    duration_s: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Return uniform white noise as float32."""
    n = int(sample_rate * duration_s)
    return np.random.uniform(-amplitude, amplitude, n).astype(np.float32)


def push_in_chunks(detector: ChordDetector, signal: np.ndarray, hop: int = HOP_SIZE) -> None:
    """Feed *signal* into *detector* in *hop*-sample chunks."""
    for start in range(0, len(signal), hop):
        end = start + hop
        detector.push_audio(signal[start:end])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector() -> ChordDetector:
    """Fresh ChordDetector at 48 kHz with default 16384-point FFT."""
    return ChordDetector(sample_rate=SAMPLE_RATE, fft_size=FFT_SIZE)


# ---------------------------------------------------------------------------
# Helpers used by tests
# ---------------------------------------------------------------------------

def _push_and_verify(
    det: ChordDetector,
    signal: np.ndarray,
    expected_midi: list[int],
    *,
    hop: int = HOP_SIZE,
) -> list[bool]:
    """Push *signal* through *det* then return verify_chord()."""
    push_in_chunks(det, signal, hop)
    return det.verify_chord(expected_midi)


# ===================================================================
# Power chord: both notes present
# ===================================================================

class TestPowerChord:
    """An E5 power chord (E2 + B2) should register both notes."""

    def test_both_notes_present(self, detector: ChordDetector) -> None:
        """E2 and B2 played together → [True, True]."""
        e2 = midi_to_freq(E2)
        b2 = midi_to_freq(B2)
        signal = generate_sine(e2) + generate_sine(b2)
        signal = (signal / np.max(np.abs(signal)) * 0.5).astype(np.float32)

        results = _push_and_verify(detector, signal, [E2, B2])
        assert results == [True, True], f"Expected [True, True], got {results}"

    def test_buffer_not_yet_full_returns_false(self, detector: ChordDetector) -> None:
        """verify_chord with too few samples → all False."""
        tiny = np.zeros(256, dtype=np.float32)
        detector.push_audio(tiny)
        results = detector.verify_chord([E2, B2])
        assert results == [False, False], f"Expected [False, False], got {results}"

    def test_silence_returns_all_false(self, detector: ChordDetector) -> None:
        """Zero-filled buffer → all False."""
        e2 = midi_to_freq(E2)
        sig = generate_sine(e2, duration_s=0.5) * 0.0  # silence
        results = _push_and_verify(detector, sig, [E2, B2])
        assert results == [False, False], f"Expected [False, False], got {results}"


# ===================================================================
# Single root: fifth should NOT be reported
# ===================================================================

class TestSingleRoot:
    """A lone root note must not falsely report the fifth."""

    def test_e2_only_no_fifth(self, detector: ChordDetector) -> None:
        """E2 alone → verify_chord([40, 47]) → [True, False]."""
        e2 = midi_to_freq(E2)
        signal = generate_sine(e2)

        results = _push_and_verify(detector, signal, [E2, B2])
        assert results[0] is True, f"Root (E2) should be detected, got {results}"
        assert results[1] is False, f"Fifth (B2) should NOT be detected, got {results}"

    def test_b2_only_no_root(self, detector: ChordDetector) -> None:
        """B2 alone → verify_chord([40, 47]) → [False, True]."""
        b2 = midi_to_freq(B2)
        signal = generate_sine(b2)

        results = _push_and_verify(detector, signal, [E2, B2])
        assert results[0] is False, f"Root (E2) should NOT be detected, got {results}"
        assert results[1] is True, f"Fifth (B2) should be detected, got {results}"


# ===================================================================
# Missing fundamental: 2nd harmonic drives detection
# ===================================================================

class TestMissingFundamental:
    """Guitar distortion often cancels the fundamental; the 2nd harmonic
    should be sufficient for ChordDetector to identify the note."""

    def test_second_harmonic_detects_root(self, detector: ChordDetector) -> None:
        """Signal at E2's 2nd harmonic (164.81 Hz) → E2 detected."""
        e2_h2 = midi_to_freq(E2) * 2.0  # 2nd harmonic of E2
        signal = generate_sine(e2_h2)

        results = _push_and_verify(detector, signal, [E2])
        assert results[0] is True, (
            f"E2 should be detected from its 2nd harmonic ({e2_h2:.2f} Hz), "
            f"got {results}"
        )

    def test_harmonic_does_not_falsely_report_fifth(self, detector: ChordDetector) -> None:
        """2nd harmonic of E2 should not trigger B2 detection."""
        e2_h2 = midi_to_freq(E2) * 2.0
        signal = generate_sine(e2_h2)

        results = _push_and_verify(detector, signal, [B2])
        assert results[0] is False, (
            f"B2 should NOT be detected from E2's 2nd harmonic, got {results}"
        )

    def test_chord_with_missing_fundamental_both_notes(self, detector: ChordDetector) -> None:
        """E2 + B2 via 2nd harmonics only → both detected."""
        e2_h2 = midi_to_freq(E2) * 2.0
        b2_h2 = midi_to_freq(B2) * 2.0
        signal = generate_sine(e2_h2) + generate_sine(b2_h2)
        signal = (signal / np.max(np.abs(signal)) * 0.5).astype(np.float32)

        results = _push_and_verify(detector, signal, [E2, B2])
        assert results == [True, True], f"Expected [True, True], got {results}"


# ===================================================================
# Noise rejection
# ===================================================================

class TestNoiseRejection:
    """White noise must not trigger false positive detections."""

    def test_white_noise_no_false_positives(self, detector: ChordDetector) -> None:
        """Uniform noise → verify_chord returns all False for E2+B2."""
        noise = generate_white_noise(duration_s=1.0)

        results = _push_and_verify(detector, noise, [E2, B2])
        assert results == [False, False], (
            f"Noise should not trigger detection, got {results}"
        )

    def test_white_noise_no_false_positives_wide_span(self, detector: ChordDetector) -> None:
        """Noise should not trigger any of several unrelated notes."""
        noise = generate_white_noise(duration_s=1.0)
        # Notes spanning the guitar range
        notes = [40, 45, 50, 55, 60, 64]
        results = _push_and_verify(detector, noise, notes)
        assert all(r is False for r in results), (
            f"Noise should not trigger any note, got {results}"
        )


# ===================================================================
# Off-frequency rejection
# ===================================================================

class TestUnrelatedFrequencies:
    """A sine wave at a completely unrelated frequency must not match."""

    def test_unrelated_note_does_not_match_power_chord(self, detector: ChordDetector) -> None:
        """D3 (146.83 Hz) → E2 + B2 both False."""
        d3_freq = midi_to_freq(D3)
        signal = generate_sine(d3_freq)

        results = _push_and_verify(detector, signal, [E2, B2])
        assert results == [False, False], (
            f"D3 sine should not match E2 or B2, got {results}"
        )

    def test_distant_note_rejected(self, detector: ChordDetector) -> None:
        """E2 sine must not trigger a note 3+ semitones away (G#2)."""
        e2_freq = midi_to_freq(E2)
        gsharp2 = E2 + 4  # G#2 = MIDI 44
        signal = generate_sine(e2_freq)

        results = _push_and_verify(detector, signal, [E2, gsharp2])
        assert results == [True, False], (
            f"E2 should be detected but G#2 should not, got {results}"
        )

    def test_nearby_semitones_limitation(self, detector: ChordDetector) -> None:
        """At low frequencies, the ±15 Hz tolerance band makes semitones
        indistinguishable — E2 (82.41 Hz) also triggers F2 (87.31 Hz, +1 st).
        This is a documented design tradeoff: wide tolerance needed for
        real guitar tuning drift vs. resolution at low pitches.
        """
        e2_freq = midi_to_freq(E2)
        f2 = E2 + 1  # F2 = MIDI 41
        signal = generate_sine(e2_freq)

        results = _push_and_verify(detector, signal, [E2, f2])
        # E2 detected, but F2 is ALSO detected due to overlapping tolerance bands
        assert results == [True, True], (
            f"With current ±15 Hz tolerance, F2 bleeds from E2 at low freqs. "
            f"Got {results}"
        )


# ===================================================================
# Reset behaviour
# ===================================================================

class TestReset:
    """After reset the detector must not retain stale audio."""

    def test_reset_clears_buffer(self, detector: ChordDetector) -> None:
        """Previous signal forgotten after reset → verify returns False."""
        e2 = midi_to_freq(E2)
        signal = generate_sine(e2)
        push_in_chunks(detector, signal)
        assert detector.verify_chord([E2]) == [True]

        detector.reset()
        assert detector.verify_chord([E2]) == [False], (
            "After reset, buffer should be empty"
        )


# ===================================================================
# Threshold sanity
# ===================================================================

class TestThresholdSanity:
    """Ensure detection thresholds are reasonable for real guitar power chords.
    These tests probe boundary conditions rather than checking specific values.
    """

    def test_low_amplitude_still_detected(self, detector: ChordDetector) -> None:
        """Very quiet but clean E2 sine should still be detected."""
        e2 = midi_to_freq(E2)
        # Amplitude 0.01 — quiet but well above noise gate
        signal = generate_sine(e2, amplitude=0.01)

        results = _push_and_verify(detector, signal, [E2])
        assert results[0] is True, (
            f"Low-amplitude E2 should still be detected, got {results}"
        )

    def test_equal_amplitude_mixed_signal(self, detector: ChordDetector) -> None:
        """Equal amplitude E2 and B2 — both detected (balanced voicing)."""
        e2 = midi_to_freq(E2)
        b2 = midi_to_freq(B2)
        e2_sig = generate_sine(e2, amplitude=0.5)
        b2_sig = generate_sine(b2, amplitude=0.5)
        signal = (e2_sig + b2_sig).astype(np.float32)

        results = _push_and_verify(detector, signal, [E2, B2])
        assert results == [True, True], f"Expected [True, True], got {results}"

    def test_moderately_unbalanced_voicing(self, detector: ChordDetector) -> None:
        """One note 5× louder than the other — this ratio should pass."""
        e2 = midi_to_freq(E2)
        b2 = midi_to_freq(B2)
        e2_sig = generate_sine(e2, amplitude=0.5)
        b2_sig = generate_sine(b2, amplitude=0.1)  # 5× quieter
        signal = (e2_sig + b2_sig).astype(np.float32)

        results = _push_and_verify(detector, signal, [E2, B2])
        assert results == [True, True], f"Expected [True, True], got {results}"

    def test_extreme_imbalance_limitation(self, detector: ChordDetector) -> None:
        """A note 10× quieter than the peak fails the 15%-of-peak threshold.
        This is expected: at 20 dB imbalance the quieter note is below the
        detection floor. Real distortion-heavy power chords rarely exceed
        ~6-8 dB imbalance between root and fifth.
        """
        e2 = midi_to_freq(E2)
        b2 = midi_to_freq(B2)
        e2_sig = generate_sine(e2, amplitude=0.5)
        b2_sig = generate_sine(b2, amplitude=0.05)  # 10× quieter
        signal = (e2_sig + b2_sig).astype(np.float32)

        results = _push_and_verify(detector, signal, [E2, B2])
        # Root definitely present; fifth may be missed
        assert results[0] is True, f"Root should be detected, got {results}"
        assert results[1] is False, (
            f"Fifth at 10× imbalance falls below 15%-of-peak threshold, got {results}"
        )
