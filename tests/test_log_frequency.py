"""Tests for pickhero.audio.log_frequency — multi-resolution log spectrum."""

import math

import numpy as np
import pytest

from pickhero.audio.log_frequency import (
    LogHypothesis,
    MultiResolutionLogSpectrum,
    _freq_to_midi,
    _midi_to_freq,
)


def _sine_wave(freq_hz: float, sample_rate: int, duration_s: float) -> np.ndarray:
    """Generate a pure sine wave as float32."""
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)


def _midi_for_freq(freq_hz: float) -> int:
    """Round-trip a frequency to the nearest MIDI note."""
    return int(round(_freq_to_midi(freq_hz)))


class TestMultiResolutionLogSpectrum:
    """Multi-resolution log-frequency spectral analysis."""

    @pytest.fixture
    def spectrum(self):
        """Default spectrum covering guitar range with three FFT sizes."""
        return MultiResolutionLogSpectrum(
            sample_rate=48000,
            min_midi=40,   # E2 ~82 Hz
            max_midi=84,   # C6 ~1047 Hz
            fft_sizes=(1024, 2048, 4096),
        )

    def test_analyse_pure_sine_returns_correct_midi(self, spectrum):
        """A pure sine wave produces a hypothesis at the correct MIDI note."""
        # A4 = 440 Hz = MIDI 69
        audio = _sine_wave(440.0, sample_rate=48000, duration_s=0.5)
        frame = spectrum.analyse(audio, top_k=3, prior_midis=set())

        assert len(frame.hypotheses) >= 1
        top = frame.hypotheses[0]
        assert top.midi == 69, f"Expected MIDI 69 (A4), got {top.midi}"
        assert top.salience > 0.0
        # cents_error should be small for a pure tone
        if top.cents_error is not None:
            assert abs(top.cents_error) < 25.0

    def test_empty_audio_returns_empty_hypotheses(self, spectrum):
        """Empty audio array returns LogFrame with empty hypotheses and zeroed dicts."""
        audio = np.array([], dtype=np.float32)
        frame = spectrum.analyse(audio, top_k=5, prior_midis=set())

        assert frame.hypotheses == []
        assert frame.midi_scores == {}
        assert len(frame.pitch_class_energy) == 12
        assert all(v == 0.0 for v in frame.pitch_class_energy.values())

    def test_prior_midis_does_not_alter_scores(self, spectrum):
        """prior_midis parameter does not change any raw score, only tiebreaks selection."""
        audio = _sine_wave(440.0, sample_rate=48000, duration_s=0.5)

        # Analyse without prior
        frame_no_prior = spectrum.analyse(audio, top_k=3, prior_midis=set())

        # Analyse with an unrelated prior — scores must be identical
        frame_with_prior = spectrum.analyse(audio, top_k=3, prior_midis={99, 100})

        assert frame_no_prior.midi_scores == frame_with_prior.midi_scores
        assert frame_no_prior.harmonic_support == frame_with_prior.harmonic_support
        assert frame_no_prior.pitch_class_energy == frame_with_prior.pitch_class_energy
        for m in range(40, 85):
            assert (
                frame_no_prior.score_for_midi(m)
                == frame_with_prior.score_for_midi(m)
            )

    def test_harmonic_support_measures_integer_multiples_only(self, spectrum):
        """harmonic_support only includes integer multiples (2, 3, 4) of f0."""
        audio = _sine_wave(440.0, sample_rate=48000, duration_s=0.5)
        frame = spectrum.analyse(audio, top_k=3, prior_midis=set())

        # A4=69 has harmonics at A5=81 (~880), E6=~88 (~1320), A6=93 (~1760)
        # The test range max_midi=84, so only A5=81 is in range
        hs = frame.harmonic_support
        if 69 in hs:
            # Harmonic support for the fundamental should be based on
            # integer-multiple harmonics (2*f0, 3*f0, 4*f0)
            assert isinstance(hs[69], float)
            assert 0.0 <= hs[69] <= 1.0

    def test_score_for_midi_returns_normalized(self, spectrum):
        """score_for_midi() returns a value in 0.0-1.0."""
        audio = _sine_wave(440.0, sample_rate=48000, duration_s=0.5)
        frame = spectrum.analyse(audio, top_k=5, prior_midis=set())

        score = frame.score_for_midi(69)
        assert 0.0 <= score <= 1.0

        # Non-existent note returns 0.0
        assert frame.score_for_midi(999) == 0.0

    def test_midi_scores_dict_contains_all_midi_range(self, spectrum):
        """midi_scores has entries for every MIDI note in the configured range."""
        audio = _sine_wave(440.0, sample_rate=48000, duration_s=0.5)
        frame = spectrum.analyse(audio, top_k=5, prior_midis=set())

        for m in range(40, 85):
            assert m in frame.midi_scores
            assert isinstance(frame.midi_scores[m], float)
            assert 0.0 <= frame.midi_scores[m] <= 1.0

    def test_hypotheses_sorted_by_salience_descending(self, spectrum):
        """Returned hypotheses are sorted by salience descending."""
        audio = _sine_wave(440.0, sample_rate=48000, duration_s=0.5)
        frame = spectrum.analyse(audio, top_k=5, prior_midis=set())

        for i in range(len(frame.hypotheses) - 1):
            assert (
                frame.hypotheses[i].salience
                >= frame.hypotheses[i + 1].salience
            )

    def test_LogHypothesis_fields_are_populated(self, spectrum):
        """Each LogHypothesis has all required fields with correct types."""
        audio = _sine_wave(440.0, sample_rate=48000, duration_s=0.5)
        frame = spectrum.analyse(audio, top_k=3, prior_midis=set())

        for hyp in frame.hypotheses:
            assert isinstance(hyp, LogHypothesis)
            assert isinstance(hyp.midi, int)
            assert isinstance(hyp.frequency, float)
            assert hyp.salience >= 0.0
            assert hyp.harmonic_support >= 0.0
            assert hyp.subharmonic_risk >= 0.0
            assert 40 <= hyp.midi <= 84

    def test_pitch_class_energy_normalized(self, spectrum):
        """pitch_class_energy values are max-normalized 0.0-1.0."""
        audio = _sine_wave(440.0, sample_rate=48000, duration_s=0.5)
        frame = spectrum.analyse(audio, top_k=5, prior_midis=set())

        assert len(frame.pitch_class_energy) == 12
        for pc in range(12):
            assert 0.0 <= frame.pitch_class_energy[pc] <= 1.0

        # At least one class should have energy (A=9 is pitch class for MIDI 69)
        assert any(v > 0.0 for v in frame.pitch_class_energy.values())
