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

import numpy as np

from pickhero.audio.note_utils import midi_to_freq


# Number of harmonics to check per note (1st=fundamental, 2nd, ..., Nth).
# A plucked guitar string produces significant energy in the first 5-8
# harmonics; we check the first 5 to keep compute bounded.
_NUM_HARMONICS = 5

# Harmonic amplitude weights — real guitar strings decay roughly as 1/h,
# but pickups and distortion reshape this. These weights are intentionally
# generous to avoid rejecting real notes on darker-sounding guitars.
_HARMONIC_WEIGHTS = np.array([1.0, 0.85, 0.7, 0.55, 0.45], dtype=np.float32)

# How many harmonics must clear the salience threshold for a note to count.
# 2-of-5 is lenient: handles distortion that cancels the fundamental
# (missing fundamental effect) while still rejecting sympathetic resonance
# (which typically produces only 1 spuriously-aligned peak).
# A pure sine wave (no harmonics) can still pass if its fundamental is
# strong enough (criterion 3, absolute floor), so synthetic test signals
# and clean DI recordings still work.
_MIN_HARMONICS_PRESENT = 2

# Each note must contribute at least this fraction of the chord's total
# harmonic-bank energy. Prevents a single loud string from masking a
# missing one. 8% is generous: a 6-note chord with one missing note
# leaves 5 notes sharing ~92%, so each present note is ~18%.
_MIN_ENERGY_RATIO = 0.08

# Chroma cosine-similarity threshold for the chord-level pre-check.
# Below this, the whole chord is rejected without per-note scoring.
# 0.3 is deliberately lenient: a single note from a chord will still
# partially match (its pitch class is in the template), so we only reject
# when the audio is clearly unrelated to the expected chord.
_CHROMA_SIMILARITY_THRESHOLD = 0.3

# How long (in seconds) to cache a verification result between onsets.
# The audio callback runs ~every 10ms; we hold the last result until a
# new onset arrives or this timeout expires (safety net for missed onsets).
_ONSET_CACHE_TTL_S = 0.5


class ChordDetector:
    """Detects chords via FFT harmonic-bank verification.

    Accumulates audio into a ring buffer. When asked to verify a chord,
    runs an FFT and checks if each expected note's fundamental and
    harmonics have significant energy relative to the spectral peak.

    Supports onset-gating: pass ``has_onset=True`` on a fresh note strike
    to trigger a new FFT analysis. Between onsets, the cached result is
    returned (avoids re-analysis during sustain).
    """

    def __init__(self, sample_rate: int = 48000, fft_size: int = 16384):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self._buffer = np.zeros(fft_size, dtype=np.float32)
        self._buffer_fill = 0
        # Precompute Hann window — avoids reallocating on every verify_chord call.
        self._window = np.hanning(fft_size).astype(np.float32)
        # Precompute FFT frequency axis (only positive frequencies).
        self._freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        # Onset-gating cache: (result, wall_time, notes_verified)
        self._cached_result: list[bool] | None = None
        self._cache_time: float = 0.0
        self._cached_notes: tuple[int, ...] | None = None
        # Per-string harmonic templates (from calibration). Optional.
        # {midi_note: np.array of length _NUM_HARMONICS}
        self._string_templates: dict[int, np.ndarray] = {}

    def set_sample_rate(self, sr: int) -> None:
        if sr != self.sample_rate:
            self.sample_rate = sr
            self._buffer = np.zeros(self.fft_size, dtype=np.float32)
            self._buffer_fill = 0
            self._freqs = np.fft.rfftfreq(self.fft_size, 1.0 / sr)
            self._cached_result = None

    def reset(self) -> None:
        """Clear the audio ring buffer and onset cache.

        Call on seek/restart so a chord check immediately after seeking doesn't
        verify against stale audio captured before the seek point.
        """
        self._buffer[:] = 0
        self._buffer_fill = 0
        self._cached_result = None
        self._cached_notes = None

    def set_string_templates(self, templates: dict[int, np.ndarray]) -> None:
        """Set per-string harmonic templates from calibration.

        Each value is a length-N array of harmonic amplitude weights
        (1st=fundamental, 2nd, ..., Nth). When set, these replace the
        default ``_HARMONIC_WEIGHTS`` for the corresponding MIDI note.
        """
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
        return self._verify(expected_midi_notes, has_onset=False)

    def verify_chord_with_onset(
        self, expected_midi_notes: list[int], has_onset: bool
    ) -> list[bool]:
        """Onset-gated chord verification.

        When ``has_onset`` is True, runs a fresh FFT analysis and caches
        the result. When False, returns the cached result if still within
        the TTL, avoiding redundant FFT computation during sustain.
        """
        return self._verify(expected_midi_notes, has_onset=has_onset)

    # ------------------------------------------------------------------
    # Core verification
    # ------------------------------------------------------------------

    def _verify(
        self, expected_midi_notes: list[int], has_onset: bool
    ) -> list[bool]:
        import time

        n = len(expected_midi_notes)
        if n == 0:
            return []
        # Onset-gating: if no onset, return cached result (if still fresh
        # AND for the same chord notes — different chords must re-analyze).
        if not has_onset and self._cached_result is not None:
            age = time.monotonic() - self._cache_time
            if age < _ONSET_CACHE_TTL_S and self._cached_notes == tuple(expected_midi_notes):
                return list(self._cached_result)
                # Different chord shape — fall through to fresh analysis.

        if self._buffer_fill < self.fft_size // 2:
            return [False] * n

        # FFT with precomputed Hann window
        buf = self._buffer[:self.fft_size] * self._window
        spectrum = np.abs(np.fft.rfft(buf))

        global_peak = float(np.max(spectrum))
        if global_peak < 1e-6:
            self._cached_result = [False] * n
            self._cached_notes = tuple(expected_midi_notes)
            self._cache_time = time.monotonic()
            return self._cached_result

        noise_floor = float(np.percentile(spectrum, 75))

        # --- Chroma pre-check: reject if the whole chord doesn't match ---
        if n >= 2 and not self._chroma_matches(spectrum, expected_midi_notes):
            self._cached_result = [False] * n
            self._cached_notes = tuple(expected_midi_notes)
            self._cache_time = time.monotonic()
            return self._cached_result

        # --- Per-note harmonic-bank scoring ---
        note_energies = []
        for midi_note in expected_midi_notes:
            energy = self._harmonic_salience(spectrum, midi_note)
            note_energies.append(energy)

        total_energy = sum(note_energies)
        results = []
        for i, (midi_note, energy) in enumerate(zip(expected_midi_notes, note_energies)):
            present = self._score_note(
                spectrum, midi_note, energy, total_energy, noise_floor, global_peak
            )
            results.append(present)

        self._cached_result = results
        self._cached_notes = tuple(expected_midi_notes)
        self._cache_time = time.monotonic()
        return results

    # ------------------------------------------------------------------
    # Harmonic-bank salience (Klapuri-inspired, score-informed)
    # ------------------------------------------------------------------

    def _harmonic_salience(self, spectrum: np.ndarray, midi_note: int) -> float:
        """Sum weighted energy across the note's harmonic series.

        This is a score-informed mini-Klapuri: we know the candidate note,
        so we check whether its harmonics (2f, 3f, 4f, 5f) align with
        spectral peaks. A real plucked string produces energy across
        multiple harmonics; sympathetic resonance at a coincidental
        frequency does not.
        """
        freq = midi_to_freq(midi_note)
        weights = self._string_templates.get(midi_note, _HARMONIC_WEIGHTS)
        if len(weights) < _NUM_HARMONICS:
            # Pad short templates with default weights.
            padded = np.full(_NUM_HARMONICS, 0.5, dtype=np.float32)
            padded[:len(weights)] = weights
            weights = padded

        salience = 0.0
        for h in range(_NUM_HARMONICS):
            harmonic_freq = freq * (h + 1)
            if harmonic_freq > self.sample_rate / 2:
                break
            e = self._freq_energy(spectrum, self._freqs, harmonic_freq)
            salience += e * weights[h]
        return salience

    def _score_note(
        self,
        spectrum: np.ndarray,
        midi_note: int,
        energy: float,
        total_energy: float,
        noise_floor: float,
        global_peak: float,
    ) -> bool:
        """Decide if a note is present given its harmonic-bank salience.

        Two paths to "present" (either suffices):
          A. **Harmonic-bank path**: ≥ ``_MIN_HARMONICS_PRESENT`` harmonics
             clear a meaningful threshold (relative to the note's own
             fundamental energy). A real plucked string produces 3-5
             harmonics; sympathetic resonance produces only 1 spurious peak.
          B. **Strong-fundamental path**: the fundamental is strong in
             absolute terms (≥ 15% of global peak AND ≥ 3× noise floor).
             This lets pure sine waves and clean DI signals pass without
             requiring harmonics, while still rejecting noise-floor blips.

        Additionally, all notes must pass the **energy ratio** check:
        each note's salience must be ≥ ``_MIN_ENERGY_RATIO`` of the chord's
        total harmonic-bank energy. Handles unbalanced voicings.
        """
        freq = midi_to_freq(midi_note)

        # --- Path A: harmonic-bank count ---
        fundamental_energy = self._freq_energy(spectrum, self._freqs, freq)
        harmonic_threshold = max(noise_floor * 3.0, fundamental_energy * 0.10)
        harmonics_present = 0
        for h in range(_NUM_HARMONICS):
            harmonic_freq = freq * (h + 1)
            if harmonic_freq > self.sample_rate / 2:
                break
            e = self._freq_energy(spectrum, self._freqs, harmonic_freq)
            if e > harmonic_threshold:
                harmonics_present += 1
        harmonic_path = harmonics_present >= _MIN_HARMONICS_PRESENT

        # --- Path B: strong fundamental (absolute) ---
        strong_fundamental = (
            fundamental_energy >= global_peak * 0.15
            and fundamental_energy > noise_floor * 3.0
        )

        if not (harmonic_path or strong_fundamental):
            return False

        # --- Energy ratio (always applies) ---
        if total_energy > 0:
            ratio = energy / total_energy
            if ratio < _MIN_ENERGY_RATIO:
                return False

        return True

    # ------------------------------------------------------------------
    # Chroma verification
    # ------------------------------------------------------------------

    def _chroma_matches(
        self, spectrum: np.ndarray, expected_notes: list[int]
    ) -> bool:
        """Chroma-level cosine similarity pre-check.

        Folds the spectrum into a 12-bin chromagram and compares against
        the expected chord's chroma template. Robust to octave errors.
        Below ``_CHROMA_SIMILARITY_THRESHOLD``, the whole chord is rejected.
        """
        chroma = self._compute_chroma(spectrum)
        expected_chroma = np.zeros(12, dtype=np.float32)
        for note in expected_notes:
            pitch_class = note % 12
            expected_chroma[pitch_class] += 1.0

        # Normalize both vectors (guard against zero-division).
        norm_c = float(np.linalg.norm(chroma)) or 1.0
        norm_e = float(np.linalg.norm(expected_chroma)) or 1.0
        chroma_norm = chroma / norm_c
        expected_norm = expected_chroma / norm_e

        similarity = float(np.dot(chroma_norm, expected_norm))
        return similarity >= _CHROMA_SIMILARITY_THRESHOLD

    def _compute_chroma(self, spectrum: np.ndarray) -> np.ndarray:
        """Compute a 12-bin chromagram from the magnitude spectrum.

        Sums spectral energy into 12 pitch-class bins (C, C#, ..., B),
        folding across all octaves. Uses the precomputed frequency axis.
        """
        chroma = np.zeros(12, dtype=np.float32)
        for i, freq in enumerate(self._freqs):
            if freq < 20.0:
                continue
            # MIDI note number for this frequency bin.
            midi = int(round(12 * np.log2(freq / 440.0) + 69))
            if 0 <= midi <= 127:
                pitch_class = midi % 12
                chroma[pitch_class] += spectrum[i]
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
        """Get spectral energy near a target frequency.

        Sums energy in a ±tolerance band around the target frequency.
        """
        mask = (freqs >= target_freq - tolerance_hz) & (freqs <= target_freq + tolerance_hz)
        if not np.any(mask):
            return 0.0
        return float(np.max(spectrum[mask]))
