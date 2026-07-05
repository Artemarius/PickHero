"""FFT-based chord verification with harmonic-bank matching.

When the matcher sees multiple notes at the same timestamp (a chord),
this module verifies the expected frequencies are present in the audio
spectrum. Unlike YIN (monophonic), FFT can detect multiple simultaneous
pitches by checking for spectral energy at each expected frequency.

Works alongside the existing YIN detector — YIN handles single notes,
this handles chords.

Improvements over naive single-bin FFT presence checks:
  - **Harmonic-bank matching**: checks fundamental + harmonics 2-5,
    rejecting sympathetic resonance (a coincidental frequency without a
    full harmonic series).
  - **Onset-gating**: only re-runs the FFT when an onset fires; between
    onsets, returns the cached result. Avoids flutter during sustain.
  - **Energy-ratio threshold**: each note must contribute a minimum
    fraction of the chord's total harmonic energy, handling unbalanced
    voicings where one string is struck louder than another.
  - **Chroma verification**: 12-bin chromagram + cosine similarity as a
    fast pre-check; robust to octave errors that plague YIN.
  - **Per-string calibration templates**: optional harmonic-weight
    templates learned during calibration (à la Rocksmith), improving
    discrimination between same-pitch notes on different strings.
"""

from __future__ import annotations

import time as _time

import numpy as np

from pickhero.audio.note_utils import midi_to_freq


# Number of harmonics to check per note (1st=fundamental, 2nd, ..., Nth).
_NUM_HARMONICS = 5

# Harmonic amplitude weights — real guitar strings decay roughly as 1/h,
# but pickups and distortion reshape this. These weights are intentionally
# generous to avoid rejecting real notes on darker-sounding guitars.
_HARMONIC_WEIGHTS = np.array([1.0, 0.85, 0.7, 0.55, 0.45], dtype=np.float32)

# How many harmonics must clear the salience threshold for a note to count.
_MIN_HARMONICS_PRESENT = 2

# Each note must contribute at least this fraction of the chord's total
# harmonic-bank energy.
_MIN_ENERGY_RATIO = 0.08

# Chroma cosine-similarity threshold for the chord-level pre-check.
_CHROMA_SIMILARITY_THRESHOLD = 0.3

# How long (in seconds) to cache a verification result between onsets.
_ONSET_CACHE_TTL_S = 0.5


class ChordDetector:
    """Detects chords via FFT harmonic-bank verification.

    Accumulates audio into a ring buffer. When asked to verify a chord,
    runs an FFT and checks if each expected note's fundamental and
    harmonics have significant energy relative to the spectral peak.
    """

    def __init__(self, sample_rate: int = 48000, fft_size: int = 16384):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self._buffer = np.zeros(fft_size, dtype=np.float32)
        self._buffer_fill = 0
        self._window = np.hanning(fft_size).astype(np.float32)
        self._freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        # Precompute pitch-class for each FFT bin (vectorized chroma).
        self._rebuild_bin_tables()
        # Onset-gating cache
        self._cached_result: list[bool] | None = None
        self._cache_time: float = 0.0
        self._cached_notes: tuple[int, ...] | None = None
        # Per-string harmonic templates (from calibration). Optional.
        self._string_templates: dict[int, np.ndarray] = {}

    def _rebuild_bin_tables(self) -> None:
        """Precompute per-bin MIDI note and pitch-class arrays for vectorized chroma."""
        # Mask out sub-audio bins (< 20 Hz) to avoid log2 issues.
        valid = self._freqs >= 20.0
        safe_freqs = np.where(valid, self._freqs, 440.0)
        midi_per_bin = np.round(12 * np.log2(safe_freqs / 440.0) + 69).astype(np.int32)
        self._bin_valid = valid
        self._bin_midi = midi_per_bin
        self._bin_pc = np.where(valid, midi_per_bin % 12, -1)

    def set_sample_rate(self, sr: int) -> None:
        if sr != self.sample_rate:
            self.sample_rate = sr
            self._buffer = np.zeros(self.fft_size, dtype=np.float32)
            self._buffer_fill = 0
            self._freqs = np.fft.rfftfreq(self.fft_size, 1.0 / sr)
            self._rebuild_bin_tables()
            self._cached_result = None
            self._cached_notes = None

    def reset(self) -> None:
        """Clear the audio ring buffer and onset cache."""
        self._buffer[:] = 0
        self._buffer_fill = 0
        self._cached_result = None
        self._cached_notes = None

    def set_string_templates(self, templates: dict[int, np.ndarray]) -> None:
        """Set per-string harmonic templates from calibration."""
        self._string_templates = {
            k: np.asarray(v, dtype=np.float32)[:_NUM_HARMONICS]
            for k, v in templates.items()
        }

    def push_audio(self, samples: np.ndarray) -> None:
        """Add audio samples to the ring buffer, keeping the most recent fft_size samples."""
        samples = samples.astype(np.float32, copy=False)
        n = len(samples)
        if n == 0:
            return
        if n >= self.fft_size:
            self._buffer[:] = samples[-self.fft_size:]
            self._buffer_fill = self.fft_size
        elif self._buffer_fill + n <= self.fft_size:
            self._buffer[self._buffer_fill:self._buffer_fill + n] = samples
            self._buffer_fill += n
        else:
            keep = self.fft_size - n
            self._buffer[:keep] = self._buffer[self._buffer_fill - keep:self._buffer_fill]
            self._buffer[keep:] = samples
            self._buffer_fill = self.fft_size

    def verify_chord(self, expected_midi_notes: list[int]) -> list[bool]:
        """Check which expected notes are present in the current spectrum."""
        return self._verify(expected_midi_notes, has_onset=False)

    def verify_chord_with_onset(
        self, expected_midi_notes: list[int], has_onset: bool
    ) -> list[bool]:
        """Onset-gated chord verification."""
        return self._verify(expected_midi_notes, has_onset=has_onset)

    # ------------------------------------------------------------------
    # Core verification
    # ------------------------------------------------------------------

    def _verify(
        self, expected_midi_notes: list[int], has_onset: bool
    ) -> list[bool]:
        n = len(expected_midi_notes)
        if n == 0:
            return []

        notes_key = tuple(expected_midi_notes)

        # Onset-gating: return cached result if fresh AND same chord notes.
        if not has_onset and self._cached_result is not None:
            age = _time.monotonic() - self._cache_time
            if age < _ONSET_CACHE_TTL_S and self._cached_notes == notes_key:
                return list(self._cached_result)

        if self._buffer_fill < self.fft_size // 2:
            return [False] * n

        buf = self._buffer[:self.fft_size] * self._window
        spectrum = np.abs(np.fft.rfft(buf))

        global_peak = float(np.max(spectrum))
        if global_peak < 1e-6:
            self._cached_result = [False] * n
            self._cached_notes = notes_key
            self._cache_time = _time.monotonic()
            return self._cached_result

        noise_floor = float(np.percentile(spectrum, 75))

        # Chroma pre-check: reject if the whole chord doesn't match.
        if n >= 2 and not self._chroma_matches(spectrum, expected_midi_notes):
            self._cached_result = [False] * n
            self._cached_notes = notes_key
            self._cache_time = _time.monotonic()
            return self._cached_result

        # Per-note harmonic-bank scoring.
        # Compute harmonic energies and harmonic-present counts in a single pass.
        note_energies = np.zeros(n, dtype=np.float32)
        fundamental_energies = np.zeros(n, dtype=np.float32)
        harmonics_present = np.zeros(n, dtype=np.int32)
        for i, midi_note in enumerate(expected_midi_notes):
            freq = midi_to_freq(midi_note)
            weights = self._string_templates.get(midi_note, _HARMONIC_WEIGHTS)
            if len(weights) < _NUM_HARMONICS:
                padded = np.full(_NUM_HARMONICS, 0.5, dtype=np.float32)
                padded[:len(weights)] = weights
                weights = padded

            fundamental_energy = 0.0
            salience = 0.0
            for h in range(_NUM_HARMONICS):
                harmonic_freq = freq * (h + 1)
                if harmonic_freq > self.sample_rate / 2:
                    break
                e = self._freq_energy(spectrum, self._freqs, harmonic_freq)
                salience += e * weights[h]
                if h == 0:
                    fundamental_energy = e

            note_energies[i] = salience
            fundamental_energies[i] = fundamental_energy
            harmonic_threshold = max(noise_floor * 3.0, fundamental_energy * 0.10)
            count = 0
            for h in range(_NUM_HARMONICS):
                harmonic_freq = freq * (h + 1)
                if harmonic_freq > self.sample_rate / 2:
                    break
                e = self._freq_energy(spectrum, self._freqs, harmonic_freq)
                if e > harmonic_threshold:
                    count += 1
            harmonics_present[i] = count

        total_energy = float(np.sum(note_energies))

        results: list[bool] = []
        for i in range(n):
            present = self._score_note(
                harmonics_present[i],
                note_energies[i],
                fundamental_energies[i],
                total_energy,
                noise_floor,
                global_peak,
            )
            results.append(present)

        self._cached_result = results
        self._cached_notes = notes_key
        self._cache_time = _time.monotonic()
        return results

    # ------------------------------------------------------------------
    # Note scoring
    # ------------------------------------------------------------------
    def _score_note(
        self,
        harmonics_present: int,
        energy: float,
        fundamental_energy: float,
        total_energy: float,
        noise_floor: float,
        global_peak: float,
    ) -> bool:
        """Decide if a note is present given its harmonic-bank salience.

        Two paths to "present" (either suffices):
          A. Harmonic-bank: ≥ _MIN_HARMONICS_PRESENT harmonics clear threshold.
          B. Strong-fundamental: fundamental ≥ 15% of global peak AND > 3× noise floor.
        Additionally, the energy-ratio check always applies.
        """
        harmonic_path = harmonics_present >= _MIN_HARMONICS_PRESENT
        strong_fundamental = (
            fundamental_energy >= global_peak * 0.15
            and fundamental_energy > noise_floor * 3.0
        )
        if not (harmonic_path or strong_fundamental):
            return False
        if total_energy > 0:
            if energy / total_energy < _MIN_ENERGY_RATIO:
                return False
        return True

    # ------------------------------------------------------------------
    # Chroma verification (vectorized)
    # ------------------------------------------------------------------

    def _chroma_matches(
        self, spectrum: np.ndarray, expected_notes: list[int]
    ) -> bool:
        """Chroma-level cosine similarity pre-check."""
        chroma = self._compute_chroma(spectrum)
        expected_chroma = np.zeros(12, dtype=np.float32)
        for note in expected_notes:
            expected_chroma[note % 12] += 1.0

        norm_c = float(np.linalg.norm(chroma)) or 1.0
        norm_e = float(np.linalg.norm(expected_chroma)) or 1.0
        similarity = float(np.dot(chroma / norm_c, expected_chroma / norm_e))
        return similarity >= _CHROMA_SIMILARITY_THRESHOLD

    def _compute_chroma(self, spectrum: np.ndarray) -> np.ndarray:
        """Compute a 12-bin chromagram from the magnitude spectrum (vectorized)."""
        chroma = np.zeros(12, dtype=np.float32)
        valid = self._bin_valid
        pc = self._bin_pc
        for pc_val in range(12):
            mask = valid & (pc == pc_val)
            if np.any(mask):
                chroma[pc_val] = float(np.sum(spectrum[mask]))
        return chroma

    # ------------------------------------------------------------------
    # Low-level spectral energy
    # ------------------------------------------------------------------

    def _freq_energy(
        self,
        spectrum: np.ndarray,
        freqs: np.ndarray,
        target_freq: float,
        tolerance_hz: float = 15.0,
    ) -> float:
        """Get spectral energy near a target frequency (max in ±tolerance band)."""
        lo = target_freq - tolerance_hz
        hi = target_freq + tolerance_hz
        # Binary search for the band edges — O(log n) instead of boolean mask.
        left = int(np.searchsorted(freqs, lo, side="left"))
        right = int(np.searchsorted(freqs, hi, side="right"))
        if left >= right:
            return 0.0
        return float(np.max(spectrum[left:right]))
