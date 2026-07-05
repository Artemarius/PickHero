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


# ===================================================================
# Onset-gated verification
# ===================================================================

class TestOnsetGating:
    """verify_chord_with_onset caches results between onsets."""

    def test_fresh_analysis_on_onset(self, detector: ChordDetector) -> None:
        """has_onset=True triggers a fresh FFT analysis."""
        e2 = midi_to_freq(E2)
        b2 = midi_to_freq(B2)
        signal = generate_sine(e2) + generate_sine(b2)
        signal = (signal / np.max(np.abs(signal)) * 0.5).astype(np.float32)
        push_in_chunks(detector, signal)

        results = detector.verify_chord_with_onset([E2, B2], has_onset=True)
        assert results == [True, True], f"Expected [True, True], got {results}"

    def test_cached_result_without_onset(self, detector: ChordDetector) -> None:
        """Without onset, cached result is returned (no re-analysis)."""
        e2 = midi_to_freq(E2)
        b2 = midi_to_freq(B2)
        signal = generate_sine(e2) + generate_sine(b2)
        signal = (signal / np.max(np.abs(signal)) * 0.5).astype(np.float32)
        push_in_chunks(detector, signal)

        # First call with onset populates the cache.
        r1 = detector.verify_chord_with_onset([E2, B2], has_onset=True)
        # Second call without onset returns the cached result.
        r2 = detector.verify_chord_with_onset([E2, B2], has_onset=False)
        assert r1 == r2, f"Cached result should match: {r1} vs {r2}"

    def test_verify_chord_uses_cache(self, detector: ChordDetector) -> None:
        """verify_chord (no onset arg) also uses the cache."""
        e2 = midi_to_freq(E2)
        b2 = midi_to_freq(B2)
        signal = generate_sine(e2) + generate_sine(b2)
        signal = (signal / np.max(np.abs(signal)) * 0.5).astype(np.float32)
        push_in_chunks(detector, signal)

        # Populate cache via verify_chord_with_onset.
        r1 = detector.verify_chord_with_onset([E2, B2], has_onset=True)
        # verify_chord should return the cached result.
        r2 = detector.verify_chord([E2, B2])
        assert r1 == r2, f"verify_chord should return cached: {r1} vs {r2}"

    def test_reset_clears_cache(self, detector: ChordDetector) -> None:
        """After reset, the onset cache is cleared."""
        e2 = midi_to_freq(E2)
        signal = generate_sine(e2)
        push_in_chunks(detector, signal)
        detector.verify_chord_with_onset([E2], has_onset=True)

        detector.reset()
        # After reset, no audio and no cache → all False.
        assert detector.verify_chord([E2]) == [False]


# ===================================================================
# Chroma verification
# ===================================================================

class TestChromaVerification:
    """The chroma pre-check rejects clearly unrelated audio."""

    def test_unrelated_note_rejected_by_chroma(self, detector: ChordDetector) -> None:
        """D3 (pitch class 2) should not match an E+B (4+11) chord template."""
        d3 = midi_to_freq(D3)
        signal = generate_sine(d3)
        results = _push_and_verify(detector, signal, [E2, B2])
        assert results == [False, False], (
            f"D3 should not match E+B chord, got {results}"
        )

    def test_chorda_passes_matching_chord(self, detector: ChordDetector) -> None:
        """A real E+B power chord passes the chroma pre-check."""
        e2 = midi_to_freq(E2)
        b2 = midi_to_freq(B2)
        signal = generate_sine(e2) + generate_sine(b2)
        signal = (signal / np.max(np.abs(signal)) * 0.5).astype(np.float32)
        results = _push_and_verify(detector, signal, [E2, B2])
        assert results == [True, True], f"E+B chord should pass, got {results}"

    def test_octave_error_still_detected(self, detector: ChordDetector) -> None:
        """E3 (one octave above E2) should still partially match E2 in chroma."""
        e3_freq = midi_to_freq(E2 + 12)  # E3 = MIDI 52
        signal = generate_sine(e3_freq)
        results = _push_and_verify(detector, signal, [E2])
        # E3's pitch class (4) matches E2's pitch class (4) in chroma,
        # and the strong fundamental should be detected.
        assert results[0] is True, f"E3 should match E2 (same pitch class), got {results}"


# ===================================================================
# Energy-ratio threshold
# ===================================================================

class TestEnergyRatio:
    """Each note must contribute a minimum fraction of the chord's total energy."""

    def test_balanced_voicing_passes(self, detector: ChordDetector) -> None:
        """Equal-amplitude E2+B2 → both detected (each is 50% of total)."""
        e2 = midi_to_freq(E2)
        b2 = midi_to_freq(B2)
        signal = generate_sine(e2, amplitude=0.5) + generate_sine(b2, amplitude=0.5)
        signal = (signal / np.max(np.abs(signal)) * 0.5).astype(np.float32)
        results = _push_and_verify(detector, signal, [E2, B2])
        assert results == [True, True], f"Balanced chord should pass, got {results}"

    def test_moderate_imbalance_passes(self, detector: ChordDetector) -> None:
        """5× imbalance → both still pass (quieter note is ~17% of total)."""
        e2 = midi_to_freq(E2)
        b2 = midi_to_freq(B2)
        signal = generate_sine(e2, amplitude=0.5) + generate_sine(b2, amplitude=0.1)
        signal = (signal / np.max(np.abs(signal)) * 0.5).astype(np.float32)
        results = _push_and_verify(detector, signal, [E2, B2])
        assert results == [True, True], f"5× imbalance should pass, got {results}"


# ===================================================================
# Per-string calibration templates
# ===================================================================

class TestStringTemplates:
    """Per-string harmonic templates from calibration improve discrimination."""

    def test_set_string_templates_accepted(self, detector: ChordDetector) -> None:
        """set_string_templates stores templates without error."""
        import numpy as np
        templates = {
            40: np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32),
            47: np.array([1.0, 0.7, 0.5, 0.3, 0.1], dtype=np.float32),
        }
        detector.set_string_templates(templates)
        # Verify detection still works with custom templates.
        e2 = midi_to_freq(E2)
        b2 = midi_to_freq(B2)
        signal = generate_sine(e2) + generate_sine(b2)
        signal = (signal / np.max(np.abs(signal)) * 0.5).astype(np.float32)
        results = _push_and_verify(detector, signal, [E2, B2])
        assert results == [True, True], f"Custom templates should work, got {results}"

    def test_short_template_padded(self, detector: ChordDetector) -> None:
        """A template shorter than _NUM_HARMONICS is padded with defaults."""
        import numpy as np
        templates = {40: np.array([1.0, 0.8], dtype=np.float32)}
        detector.set_string_templates(templates)
        # Should not crash — padding happens internally.
        e2 = midi_to_freq(E2)
        signal = generate_sine(e2)
        results = _push_and_verify(detector, signal, [E2])
        assert results == [True], f"Padded template should work, got {results}"


class TestOnsetAnchoredWindow:
    """Onset-anchored window freezes the transient on has_onset=True."""

    def test_onset_anchored_window_freezes_transient(self, detector: ChordDetector) -> None:
        """When has_onset=True, the ring buffer is frozen for the FFT.

        Push audio, call verify_chord_with_onset(has_onset=True), then push
        DIFFERENT audio (a new note), and call verify_chord_with_onset(has_onset=False).
        The second result should use the FROZEN window (the original chord),
        not the new audio — because the frozen window captures the strike transient.
        """
        # Push a power chord (E2 + B2)
        e2 = midi_to_freq(E2)
        b2 = midi_to_freq(B2)
        chord_signal = generate_sine(e2) + generate_sine(b2)
        chord_signal = (chord_signal / np.max(np.abs(chord_signal)) * 0.5).astype(np.float32)
        detector.push_audio(chord_signal)

        # Verify with onset — should freeze the window and detect both notes
        results1 = detector.verify_chord_with_onset([E2, B2], has_onset=True)
        assert results1 == [True, True], f"Both notes should be present, got {results1}"

        # Push DIFFERENT audio — an unrelated note (D3)
        d3 = midi_to_freq(D3)
        unrelated_signal = generate_sine(d3)
        unrelated_signal = (unrelated_signal / np.max(np.abs(unrelated_signal)) * 0.5).astype(np.float32)
        detector.push_audio(unrelated_signal)

        # Verify WITHOUT onset — should use the frozen window (E2+B2), not D3
        results2 = detector.verify_chord_with_onset([E2, B2], has_onset=False)
        # The frozen window still has the chord, so both should still be present.
        # If the detector used the live buffer (D3), it would NOT find E2+B2.
        assert results2 == [True, True], (
            f"Frozen window should still show E2+B2, got {results2}. "
            f"If this fails, the onset-anchored window is not being used."
        )

    def test_frozen_window_cleared_on_reset(self, detector: ChordDetector) -> None:
        """reset() clears the frozen window so the next call uses the live buffer."""
        e2 = midi_to_freq(E2)
        signal = generate_sine(e2)
        detector.push_audio(signal)
        detector.verify_chord_with_onset([E2], has_onset=True)
        assert detector._frozen_buffer is not None

        detector.reset()
        assert detector._frozen_buffer is None
