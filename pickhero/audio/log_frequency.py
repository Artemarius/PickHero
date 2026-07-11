"""Multi-resolution log-frequency spectral analysis.

Provides a constant-Q-style front end that combines several FFT sizes so that
low fundamentals get long windows (good frequency resolution) while high
fundamentals keep shorter windows (good time resolution).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _midi_to_freq(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _freq_to_midi(freq: float) -> float:
    return 69.0 + 12.0 * np.log2(freq / 440.0 + 1e-15)


@dataclass(frozen=True)
class LogHypothesis:
    midi: int
    frequency: float
    cents_error: float | None
    salience: float
    harmonic_support: float
    subharmonic_risk: float


@dataclass(frozen=True)
class LogFrame:
    hypotheses: list[LogHypothesis]
    _midi_salience: dict[int, float]
    _harmonic_support: dict[int, float]
    _subharmonic_risk: dict[int, float]
    _pitch_class_energy: dict[int, float]
    _midi_scores: dict[int, float]

    @property
    def harmonic_support(self) -> dict[int, float]:
        return self._harmonic_support

    @property
    def pitch_class_energy(self) -> dict[int, float]:
        return self._pitch_class_energy

    @property
    def midi_scores(self) -> dict[int, float]:
        return self._midi_scores

    def score_for_midi(self, midi: int) -> float:
        return self._midi_salience.get(int(midi), 0.0)

    def alias_risk_for_midi(self, midi: int) -> float:
        return self._subharmonic_risk.get(int(midi), 0.0)


class MultiResolutionLogSpectrum:
    """Log-spaced MIDI energy from several FFT resolutions."""

    def __init__(
        self,
        sample_rate: int,
        min_midi: int,
        max_midi: int,
        fft_sizes: tuple[int, ...],
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.min_midi = int(min_midi)
        self.max_midi = int(max_midi)
        self.fft_sizes = tuple(sorted(set(int(s) for s in fft_sizes)))
        if not self.fft_sizes:
            raise ValueError("fft_sizes must not be empty")

        self._midi_range = list(range(self.min_midi, self.max_midi + 1))
        self._midi_freqs = np.array(
            [_midi_to_freq(m) for m in self._midi_range], dtype=np.float64
        )

    def analyse(
        self,
        audio: np.ndarray,
        top_k: int,
        prior_midis: set[int],
    ) -> LogFrame:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return LogFrame(
                hypotheses=[],
                _midi_salience={},
                _harmonic_support={},
                _subharmonic_risk={},
                _pitch_class_energy={i: 0.0 for i in range(12)},
                _midi_scores={},
            )

        energy_by_size: list[np.ndarray] = []
        peak_freq_by_size: list[np.ndarray] = []

        for size in self.fft_sizes:
            if len(audio) < size:
                padded = np.zeros(size, dtype=np.float32)
                padded[: len(audio)] = audio
            else:
                # Use the most recent `size` samples for time-local analysis.
                padded = audio[-size:].astype(np.float32)

            # Center / window to reduce spectral leakage.
            windowed = padded * np.hanning(len(padded))
            spectrum = np.abs(np.fft.rfft(windowed))
            freqs = np.fft.rfftfreq(len(padded), 1.0 / self.sample_rate)
            energy, peak_freq = self._midi_energy(spectrum, freqs)
            energy_by_size.append(energy)
            peak_freq_by_size.append(peak_freq)

        # Combine resolutions: each MIDI note keeps the strongest response.
        energy_stack = np.stack(energy_by_size, axis=0)
        best_resolution = np.argmax(energy_stack, axis=0)
        combined = np.max(energy_stack, axis=0)

        # Peak frequency from the resolution that contributed the energy.
        peak_freqs = np.array(
            [
                peak_freq_by_size[best_resolution[i]][i]
                for i in range(len(self._midi_range))
            ]
        )

        total_energy = float(np.sum(combined)) + 1e-15
        midi_salience = {
            m: float(e / total_energy) for m, e in zip(self._midi_range, combined)
        }

        # Harmonic support: energy at integer-multiple harmonics of a candidate
        # relative to the frame's total energy.
        harmonic_support: dict[int, float] = {}
        for idx, m in enumerate(self._midi_range):
            harmonic_midis = [round(m + 12.0 * np.log2(h)) for h in range(2, 5)]
            harmonic_midis = [h for h in harmonic_midis if self.min_midi <= h <= self.max_midi]
            harmonic_indices = [self._midi_range.index(h) for h in harmonic_midis]
            harmonic_energy = sum(combined[i] for i in harmonic_indices)
            harmonic_support[m] = float(min(1.0, harmonic_energy / total_energy))

        # Subharmonic / alias risk: competing energy at octave/sub-octave below.
        subharmonic_risk: dict[int, float] = {}
        for idx, m in enumerate(self._midi_range):
            risks = []
            for delta in (12, 19):  # octave, ~f/3
                sub = m - delta
                if sub >= self.min_midi:
                    sub_idx = self._midi_range.index(sub)
                    denominator = combined[idx] + combined[sub_idx] + 1e-15
                    risks.append(combined[sub_idx] / denominator)
            subharmonic_risk[m] = float(max(risks, default=0.0))

        # Pitch-class energy: sum per class, then max-normalize.
        pc_raw = {pc: 0.0 for pc in range(12)}
        for m, e in zip(self._midi_range, combined):
            pc_raw[m % 12] += float(e)
        pc_max = max(pc_raw.values()) or 1e-15
        pitch_class_energy = {pc: v / pc_max for pc, v in pc_raw.items()}

        # Select hypotheses using prior_midis as a tiebreaker when salience is
        # nearly equal (within TIE_TOLERANCE). Returned scores are unchanged;
        # the final hypotheses are always ordered by true salience descending.
        top_k = max(0, int(top_k))
        prior = prior_midis or set()
        tie_tolerance = 0.02
        indexed = list(enumerate(self._midi_range))

        # Stage 1: tolerant ranking decides which midis make the cut.
        indexed.sort(
            key=lambda item: (
                -(round(midi_salience[item[1]] / tie_tolerance) * tie_tolerance),
                0 if item[1] in prior else 1,
                item[1],
            )
        )
        selected = indexed[:top_k]

        # Stage 2: output order is strictly by true salience descending.
        selected.sort(key=lambda item: (-midi_salience[item[1]], item[1]))

        hypotheses: list[LogHypothesis] = []
        for idx, m in selected:
            freq = peak_freqs[idx]
            cents = self._cents_error(freq, m)
            hypotheses.append(
                LogHypothesis(
                    midi=m,
                    frequency=float(freq),
                    cents_error=cents,
                    salience=midi_salience[m],
                    harmonic_support=harmonic_support[m],
                    subharmonic_risk=subharmonic_risk[m],
                )
            )

        return LogFrame(
            hypotheses=hypotheses,
            _midi_salience=midi_salience,
            _harmonic_support=harmonic_support,
            _subharmonic_risk=subharmonic_risk,
            _pitch_class_energy=pitch_class_energy,
            _midi_scores=dict(midi_salience),
        )

    def _midi_energy(
        self, spectrum: np.ndarray, freqs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-MIDI energy and per-MIDI peak frequency estimate."""
        energy = np.zeros(len(self._midi_range), dtype=np.float64)
        peak_freqs = np.zeros(len(self._midi_range), dtype=np.float64)

        for i, m in enumerate(self._midi_range):
            center = self._midi_freqs[i]
            lower = _midi_to_freq(m - 0.5)
            upper = _midi_to_freq(m + 0.5)

            lo_bin = int(np.floor(lower / (self.sample_rate / (2 * (len(spectrum) - 1)))))
            hi_bin = int(np.ceil(upper / (self.sample_rate / (2 * (len(spectrum) - 1)))))
            # Robust bin clamping.
            lo_bin = max(1, min(lo_bin, len(freqs) - 1))
            hi_bin = max(lo_bin, min(hi_bin, len(freqs) - 1))

            band_freqs = freqs[lo_bin : hi_bin + 1]
            band_mag = spectrum[lo_bin : hi_bin + 1].astype(np.float64)
            if band_mag.size == 0:
                continue

            # Triangular weighting peaking at the center frequency.
            weights = 1.0 - np.abs(band_freqs - center) / max(center - lower, upper - center)
            weights = np.maximum(weights, 0.0)
            weight_sum = np.sum(weights) + 1e-15
            energy[i] = np.sum(band_mag * weights) / weight_sum

            # Peak frequency for cents error: bin of maximum magnitude in band.
            peak_bin = lo_bin + int(np.argmax(band_mag))
            peak_freqs[i] = freqs[peak_bin]

        return energy, peak_freqs

    def _cents_error(self, freq: float, midi: int) -> float | None:
        if freq <= 0.0:
            return None
        target = _midi_to_freq(midi)
        cents = 1200.0 * np.log2(freq / target)
        return float(np.clip(cents, -100.0, 100.0))
