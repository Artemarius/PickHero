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

    def __init__(self, sample_rate: int = 48000, fft_size: int = 16384):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self._buffer = np.zeros(fft_size, dtype=np.float32)
        self._buffer_fill = 0
        # Precompute Hann window — avoids reallocating on every verify_chord call.
        self._window = np.hanning(fft_size).astype(np.float32)

    def set_sample_rate(self, sr: int) -> None:
        if sr != self.sample_rate:
            self.sample_rate = sr
            self._buffer = np.zeros(self.fft_size, dtype=np.float32)
            self._buffer_fill = 0

    def reset(self) -> None:
        """Clear the audio ring buffer.

        Call on seek/restart so a chord check immediately after seeking doesn't
        verify against stale audio captured before the seek point.
        """
        self._buffer[:] = 0
        self._buffer_fill = 0

    def push_audio(self, samples: np.ndarray) -> None:
        """Add audio samples to the ring buffer, keeping the most recent fft_size samples."""
        samples = samples.astype(np.float32, copy=False)
        n = len(samples)
        if n == 0:
            return

        if n >= self.fft_size:
            # Input larger than buffer: keep only the most recent fft_size samples.
            self._buffer[:] = samples[-self.fft_size:]
            self._buffer_fill = self.fft_size
        elif self._buffer_fill + n <= self.fft_size:
            self._buffer[self._buffer_fill:self._buffer_fill + n] = samples
            self._buffer_fill += n
        else:
            # Shift buffer left and append new samples.
            keep = self.fft_size - n
            self._buffer[:keep] = self._buffer[self._buffer_fill - keep:self._buffer_fill]
            self._buffer[keep:] = samples
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

        # FFT with precomputed Hann window
        buf = self._buffer[:self.fft_size] * self._window
        spectrum = np.abs(np.fft.rfft(buf))
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)

        # Global noise floor: a note counts as present only if it rises above
        # the spectrum's 75th percentile, not just the median. This is more
        # robust against random peaks in broadband noise.
        global_peak = float(np.max(spectrum))
        if global_peak < 1e-6:
            return [False] * len(expected_midi_notes)
        noise_floor = float(np.percentile(spectrum, 75))

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

            # Note is present if it clears the noise floor by a healthy margin
            # AND is at least 15% of the global peak. The floor check prevents
            # a single loud fundamental from drowning out quieter chord tones
            # (the old global-peak-only threshold failed on spread voicings).
            results.append(total >= global_peak * 0.15 and total > noise_floor * 3.0)

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
