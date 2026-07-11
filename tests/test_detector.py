"""Tests for pickhero.audio.detector.

Generates synthetic sine waves and feeds them to the pitch detector
to verify correct note identification. No audio hardware required.
"""

import numpy as np
import pytest

from pickhero.audio.note_utils import midi_to_freq, freq_to_midi


def generate_sine(freq: float, duration_s: float = 0.5, sample_rate: int = 44100,
                  amplitude: float = 0.5) -> np.ndarray:
    """Generate a sine wave at the given frequency."""
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# Skip all tests if aubio is not installed
aubio = pytest.importorskip("aubio")

from pickhero.audio.detector import PitchDetector  # noqa: E402


class TestPitchDetector:
    """Feed synthetic sine waves and verify detection."""

    @pytest.fixture
    def detector(self):
        return PitchDetector(
            buf_size=2048,
            hop_size=512,
            sample_rate=44100,
            confidence_threshold=0.3,  # lower for synthetic signals
            onset_threshold=0.3,
            noise_gate_db=-80.0,  # very low gate for clean sine
        )

    def _detect_note_from_sine(self, detector: PitchDetector, freq: float) -> int | None:
        """Generate a sine wave and return the most common detected MIDI note."""
        signal = generate_sine(freq, duration_s=0.3, amplitude=0.5)
        hop = detector.hop_size
        detected = []

        for i in range(0, len(signal) - hop + 1, hop):
            chunk = signal[i:i + hop]
            result = detector.process(chunk)
            if result is not None:
                detected.append(result.midi_note)

        if not detected:
            return None
        # Return most common detection
        from collections import Counter
        return Counter(detected).most_common(1)[0][0]

    def test_a4_440hz(self, detector):
        """A4 = 440 Hz = MIDI 69"""
        note = self._detect_note_from_sine(detector, 440.0)
        assert note is not None
        assert note == 69

    def test_a3_220hz(self, detector):
        """A3 = 220 Hz = MIDI 57"""
        note = self._detect_note_from_sine(detector, 220.0)
        assert note is not None
        assert note == 57

    def test_e2_low_e_string(self, detector):
        """E2 = ~82.41 Hz = MIDI 40 (low E string open)"""
        note = self._detect_note_from_sine(detector, 82.41)
        assert note is not None
        assert note == 40

    def test_e4_high_e_string(self, detector):
        """E4 = ~329.63 Hz = MIDI 64 (high E string open)"""
        note = self._detect_note_from_sine(detector, 329.63)
        assert note is not None
        assert note == 64

    def test_d3_open_d_string(self, detector):
        """D3 = ~146.83 Hz = MIDI 50"""
        note = self._detect_note_from_sine(detector, 146.83)
        assert note is not None
        assert note == 50

    def test_silence_returns_none(self, detector):
        """Silent input should not produce detections."""
        silence = np.zeros(detector.hop_size, dtype=np.float32)
        result = detector.process(silence)
        assert result is None

    def test_very_quiet_below_noise_gate(self, detector):
        """Extremely quiet signal should be caught by noise gate."""
        signal = generate_sine(440.0, duration_s=0.05, amplitude=0.00001)
        hop = detector.hop_size
        results = []
        for i in range(0, len(signal) - hop + 1, hop):
            result = detector.process(signal[i:i + hop])
            if result is not None:
                results.append(result)
        assert len(results) == 0

    def test_all_open_strings(self, detector):
        """Verify detection of all 6 open string frequencies."""
        open_strings = {
            40: 82.41,   # E2
            45: 110.00,  # A2
            50: 146.83,  # D3
            55: 196.00,  # G3
            59: 246.94,  # B3
            64: 329.63,  # E4
        }
        for expected_midi, freq in open_strings.items():
            note = self._detect_note_from_sine(detector, freq)
            assert note is not None, f"Failed to detect {freq}Hz (expected MIDI {expected_midi})"
            assert note == expected_midi, \
                f"Detected MIDI {note} instead of {expected_midi} for {freq}Hz"

    def test_reset_clears_state(self, detector):
        """After reset, detector should work as if freshly created."""
        # Process some audio
        signal = generate_sine(440.0, duration_s=0.1)
        for i in range(0, len(signal) - detector.hop_size + 1, detector.hop_size):
            detector.process(signal[i:i + detector.hop_size])

        # Reset
        detector.reset()

        # Should still work
        note = self._detect_note_from_sine(detector, 440.0)
        assert note == 69

class TestOnsetSample:
    """Verify onset_sample field is populated correctly."""

    @pytest.fixture
    def detector(self):
        return PitchDetector(
            buf_size=2048,
            hop_size=512,
            sample_rate=44100,
            confidence_threshold=0.3,
            onset_threshold=0.3,
            noise_gate_db=-80.0,
        )

    def test_onset_sample_is_none_for_non_onset(self, detector):
        """Non-onset detections should have onset_sample=None."""
        signal = generate_sine(440.0, duration_s=0.2, amplitude=0.5)
        hop = detector.hop_size
        non_onset_results = []
        for i in range(0, len(signal) - hop + 1, hop):
            result = detector.process(signal[i:i + hop])
            if result is not None and not result.is_onset:
                non_onset_results.append(result)
        for r in non_onset_results:
            assert r.onset_sample is None

    def test_onset_sample_is_int_when_onset(self, detector):
        """When an onset fires, onset_sample should be a non-negative int."""
        sr = 44100
        hop = detector.hop_size
        total = hop * 20
        t = np.arange(total, dtype=np.float32) / sr
        sig = np.zeros(total, dtype=np.float32)
        start = hop * 4
        sig[start:] = 0.5 * np.sin(2 * np.pi * 440 * t[:total - start]) * np.exp(-3.0 * t[:total - start])
        onset_results = []
        for i in range(0, total - hop + 1, hop):
            result = detector.process(sig[i:i + hop])
            if result is not None and result.is_onset:
                onset_results.append(result)
        assert len(onset_results) >= 1, "Expected at least one onset detection"
        for r in onset_results:
            assert isinstance(r.onset_sample, int)
            assert r.onset_sample >= 0

    def test_onset_sample_increases_monotonically(self, detector):
        """Successive onsets should have non-decreasing sample positions."""
        sr = 44100
        hop = detector.hop_size
        burst_len = int(sr * 0.1)
        silence_len = int(sr * 0.1)
        burst1 = (0.5 * np.sin(2 * np.pi * 440 * np.arange(burst_len) / sr)).astype(np.float32)
        silence = np.zeros(silence_len, dtype=np.float32)
        burst2 = (0.5 * np.sin(2 * np.pi * 330 * np.arange(burst_len) / sr)).astype(np.float32)
        signal = np.concatenate([burst1, silence, burst2])
        onset_samples = []
        for i in range(0, len(signal) - hop + 1, hop):
            result = detector.process(signal[i:i + hop])
            if result is not None and result.is_onset:
                onset_samples.append(result.onset_sample)
        for j in range(1, len(onset_samples)):
            assert onset_samples[j] >= onset_samples[j - 1], \
                f"onset_sample decreased: {onset_samples[j]} < {onset_samples[j - 1]}"