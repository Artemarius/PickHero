"""FFT-based chord verification.

When the matcher sees multiple notes at the same timestamp (a chord),
this module verifies the expected frequencies are present in the audio
spectrum. Unlike YIN (monophonic), FFT can detect multiple simultaneous
pitches by checking for spectral energy at each expected frequency.

Works alongside the existing YIN detector — YIN handles single notes,
this handles chords.
"""

from __future__ import annotations

import numpy as np

from pickhero.audio.note_utils import midi_to_freq


class ChordDetector:
    """Detects chords via FFT spectral energy verification.

    Accumulates audio into a ring buffer. When asked to verify a chord,
    runs an FFT and checks if each expected note's frequency has
    significant energy relative to the spectral peak.
    """

    def __init__(self, sample_rate: int = 48000, fft_size: int = 8192):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self._buffer = np.zeros(fft_size, dtype=np.float32)
        self._buffer_fill = 0

    def set_sample_rate(self, sr: int) -> None:
        if sr != self.sample_rate:
            self.sample_rate = sr
            self._buffer = np.zeros(self.fft_size, dtype=np.float32)
            self._buffer_fill = 0

    def push_audio(self, samples: np.ndarray) -> None:
        """Add audio samples to the ring buffer."""
        samples = samples.astype(np.float32, copy=False)
        n = len(samples)
        if n == 0:
            return

        if self._buffer_fill + n <= self.fft_size:
            self._buffer[self._buffer_fill:self._buffer_fill + n] = samples
            self._buffer_fill += n
        else:
            # Shift buffer left and append new samples
            keep = self.fft_size - n
            self._buffer[:keep] = self._buffer[n:self.fft_size]
            self._buffer[keep:] = samples[:self.fft_size - keep]
            self._buffer_fill = self.fft_size

    def verify_chord(self, expected_midi_notes: list[int]) -> list[bool]:
        """Check which expected notes are present in the current spectrum.

        Args:
            expected_midi_notes: List of MIDI note numbers for each string
                                 in the chord.

        Returns:
            List of booleans, one per input note — True if the note's
            frequency has significant spectral energy.
        """
        if self._buffer_fill < self.fft_size // 2:
            return [False] * len(expected_midi_notes)

        # FFT with Hann window
        buf = self._buffer[:self.fft_size] * np.hanning(self.fft_size)
        spectrum = np.abs(np.fft.rfft(buf))
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)

        # Find overall peak for relative threshold
        peak_energy = np.max(spectrum)
        if peak_energy < 1e-6:
            return [False] * len(expected_midi_notes)

        # For each expected note, check energy at its fundamental and
        # first harmonic (2nd harmonic is strongest on guitar)
        results = []
        for midi_note in expected_midi_notes:
            freq = midi_to_freq(midi_note)
            energy = self._freq_energy(spectrum, freqs, freq)

            # Also check 2nd harmonic — on guitar, the 2nd harmonic is
            # often stronger than the fundamental, especially on distortion
            harm_energy = self._freq_energy(spectrum, freqs, freq * 2.0)
            total = max(energy, harm_energy * 0.7)

            # Note is present if energy is at least 15% of peak
            results.append(total >= peak_energy * 0.15)

        return results

    def _freq_energy(
        self,
        spectrum: np.ndarray,
        freqs: np.ndarray,
        target_freq: float,
        tolerance_hz: float = 15.0,
    ) -> float:
        """Get spectral energy near a target frequency.

        Sums energy in a ±tolerance band around the target frequency.
        """
        mask = (freqs >= target_freq - tolerance_hz) & (freqs <= target_freq + tolerance_hz)
        if not np.any(mask):
            return 0.0
        return float(np.max(spectrum[mask]))
