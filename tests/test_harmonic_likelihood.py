"""Tests for the harmonic likelihood spectral validator (M3.3).

Validates that _spectral_check scores real harmonic signals high and
noise/cable resonances low, and that more harmonics score higher.
"""
from __future__ import annotations

import numpy as np
import pytest

from pickhero.audio.pitch_engine import PitchEngine


@pytest.fixture
def engine() -> PitchEngine:
    """PitchEngine configured for testing with a populated spectral buffer."""
    eng = PitchEngine(
        sample_rate=44100,
        hop_size=512,
        buf_size=4096,
        confidence_threshold=0.3,
        onset_threshold=0.5,
        noise_gate_db=-80.0,
        calibration=None,
        profile="high_accuracy",
    )
    return eng


def _fill_spectral(eng: PitchEngine, signal: np.ndarray) -> None:
    """Fill the engine's rolling spectral buffer with the given signal."""
    buf_size = eng._SPECTRAL_BUF_SIZE
    # Tile or truncate to exactly buf_size samples
    if len(signal) < buf_size:
        reps = (buf_size // len(signal)) + 1
        signal = np.tile(signal, reps)
    signal = signal[:buf_size].astype(np.float32)
    eng._push_spectral(signal)


def _harmonic_signal(freq: float, sr: int, n_harmonics: int, duration: float = 0.1) -> np.ndarray:
    """Generate a signal with the given fundamental and N harmonics."""
    t = np.arange(int(sr * duration)) / sr
    signal = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        signal += (1.0 / h) * np.sin(2 * np.pi * freq * h * t)
    return signal.astype(np.float32)


def _noise_signal(sr: int, duration: float = 0.1) -> np.ndarray:
    """Generate white noise."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(sr * duration)).astype(np.float32)


class TestHarmonicLikelihood:
    """M3.3: _spectral_check returns harmonic likelihood in [0, 1]."""

    def test_harmonic_signal_scores_high(self, engine: PitchEngine) -> None:
        """A signal with 5 harmonics should score > 0.15."""
        signal = _harmonic_signal(220.0, 44100, n_harmonics=5)
        _fill_spectral(engine, signal)
        score = engine._spectral_check(220.0)
        assert score > 0.15, f"Harmonic signal scored {score:.3f}, expected > 0.15"

    def test_noise_scores_low(self, engine: PitchEngine) -> None:
        """White noise should score < 0.05 (no harmonic structure)."""
        signal = _noise_signal(44100)
        _fill_spectral(engine, signal)
        score = engine._spectral_check(220.0)
        assert score < 0.05, f"Noise scored {score:.3f}, expected < 0.05"

    def test_harmonic_plus_noise_still_scores_high(self, engine: PitchEngine) -> None:
        """Harmonic signal mixed with noise should still score > 0.15."""
        harmonic = _harmonic_signal(220.0, 44100, n_harmonics=5)
        noise = _noise_signal(44100, duration=len(harmonic) / 44100)
        signal = harmonic + 0.5 * noise
        _fill_spectral(engine, signal)
        score = engine._spectral_check(220.0)
        assert score > 0.15, f"Harmonic+noise scored {score:.3f}, expected > 0.15"

    def test_harmonic_scores_higher_than_noise(self, engine: PitchEngine) -> None:
        """Harmonic signal must score higher than pure noise."""
        harmonic = _harmonic_signal(220.0, 44100, n_harmonics=5)
        _fill_spectral(engine, harmonic)
        harmonic_score = engine._spectral_check(220.0)

        noise = _noise_signal(44100)
        _fill_spectral(engine, noise)
        noise_score = engine._spectral_check(220.0)

        assert harmonic_score > noise_score, (
            f"Harmonic ({harmonic_score:.3f}) should > noise ({noise_score:.3f})"
        )

    def test_more_harmonics_score_higher(self, engine: PitchEngine) -> None:
        """A signal with 5 harmonics should score higher than 3 harmonics."""
        signal_5 = _harmonic_signal(220.0, 44100, n_harmonics=5)
        _fill_spectral(engine, signal_5)
        score_5 = engine._spectral_check(220.0)

        signal_3 = _harmonic_signal(220.0, 44100, n_harmonics=3)
        _fill_spectral(engine, signal_3)
        score_3 = engine._spectral_check(220.0)

        assert score_5 > score_3, (
            f"5 harmonics ({score_5:.3f}) should > 3 harmonics ({score_3:.3f})"
        )

    def test_zero_frequency_returns_zero(self, engine: PitchEngine) -> None:
        """freq=0 should return 0.0 (no valid fundamental)."""
        signal = _harmonic_signal(220.0, 44100, n_harmonics=5)
        _fill_spectral(engine, signal)
        assert engine._spectral_check(0.0) == 0.0

    def test_empty_buffer_returns_zero(self, engine: PitchEngine) -> None:
        """With no spectral data, score should be 0.0."""
        # Fresh engine — no data pushed
        assert engine._spectral_check(220.0) == 0.0

    def test_score_in_unit_interval(self, engine: PitchEngine) -> None:
        """Score must be in [0, 1] for any input."""
        signal = _harmonic_signal(110.0, 44100, n_harmonics=5)
        _fill_spectral(engine, signal)
        score = engine._spectral_check(110.0)
        assert 0.0 <= score <= 1.0
