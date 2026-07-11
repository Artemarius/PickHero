"""Spectral expected-event verifier.

Implements :class:`~pickhero.audio.verifier.ExpectedEventVerifier` using
FFT harmonic-bank verification.  Works on the raw audio window supplied by
:class:`~pickhero.audio.input.AudioCapture`, independent of the hop-sized
YIN detector.
"""

from __future__ import annotations

import numpy as np

from pickhero.audio.evidence import (
    ChordVerification,
    ExpectedNote,
    NoteVerification,
    PitchEvidence,
    TechniqueVerification,
)
from pickhero.audio.match_mode import MatchMode
from pickhero.audio.note_utils import cents_band, freq_to_midi, midi_to_freq
from pickhero.audio.verification_policy import VerificationPolicy
from pickhero.audio.verifier import ExpectedEventVerifier


# Number of harmonics to check per note (fundamental + overtones).
_NUM_HARMONICS = 5

# Harmonic amplitude weights — decay roughly as 1/h.
_HARMONIC_WEIGHTS = np.array([1.0, 0.85, 0.7, 0.55, 0.45], dtype=np.float32)

# Minimum harmonics that must clear the salience threshold, by mode.
# ARCADE: fundamental-only is enough for DI/clean audio (GuitarSet hexaphonic).
# PRACTICE: require at least one overtone for confidence.
# JUDGE: require strong harmonic series.
_MIN_HARMONICS_PRESENT = {
    MatchMode.ARCADE: 1,
    MatchMode.PRACTICE: 2,
    MatchMode.JUDGE: 3,
}

# Frequency tolerance is cents-based, sourced from VerificationPolicy.

# Single-note presence threshold by mode.
# ARCADE uses a lower bar to catch DI audio where overall energy is lower.
_HARMONIC_SCORE_THRESHOLDS = {
    MatchMode.ARCADE: 0.10,
    MatchMode.PRACTICE: 0.25,
    MatchMode.JUDGE: 0.5,
}

# Per-note minimum energy contribution for chords.
_MIN_ENERGY_RATIO = {
    MatchMode.ARCADE: 0.06,
    MatchMode.PRACTICE: 0.08,
    MatchMode.JUDGE: 0.12,
}

# Anti-alias competitor offsets in semitones.
_COMPETITOR_OFFSETS = (-19, -12, -2, -1, 1, 2, 12, 19)

# Mode-specific margin the target score must beat the best competitor by.
_ANTI_ALIAS_MARGINS = {
    MatchMode.JUDGE: 0.15,
    MatchMode.PRACTICE: 0.10,
    MatchMode.ARCADE: 0.05,
}

# Minimum fundamental score relative to the spectral peak.
_MIN_F0_SCORE = {
    MatchMode.ARCADE: 0.02,
    MatchMode.PRACTICE: 0.05,
    MatchMode.JUDGE: 0.10,
}


class SpectralVerifier(ExpectedEventVerifier):
    """Verifier using FFT harmonic-bank detection.

    Stateless with respect to audio history: each verification call runs a
    fresh FFT on the supplied window.  This makes it suitable for the
    expected-event architecture where the matcher asks "is this note in
    *this* window?".
    """

    def __init__(self, fft_size: int = 8192, sample_rate: int = 48000):
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self._window = np.hanning(fft_size).astype(np.float32)
        self._freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        self._last_window_id: int | None = None
        self._last_spectrum: np.ndarray | None = None

    def _spectrum(self, audio_window: np.ndarray) -> np.ndarray:
        """Return magnitude spectrum of the window, with a one-slot cache."""
        # Same array object commonly passed twice (chord notes, onset checks).
        window_id = id(audio_window)
        if self._last_window_id == window_id and self._last_spectrum is not None:
            return self._last_spectrum
        n = len(audio_window)
        if n >= self.fft_size:
            buf = audio_window[-self.fft_size:]
        else:
            buf = np.zeros(self.fft_size, dtype=audio_window.dtype)
            buf[:n] = audio_window
        buf = buf * self._window
        spectrum = np.abs(np.fft.rfft(buf))
        self._last_window_id = window_id
        self._last_spectrum = spectrum
        return spectrum

    def _freq_energy(
        self, spectrum: np.ndarray, target_freq: float, cents_tolerance: float
    ) -> float:
        """Energy in a band around target_freq using cents tolerance."""
        if target_freq <= 0 or target_freq >= self.sample_rate / 2:
            return 0.0
        lo, hi = cents_band(target_freq, cents_tolerance)
        left = int(np.searchsorted(self._freqs, lo, side="left"))
        right = int(np.searchsorted(self._freqs, hi, side="right"))
        if left >= right:
            return 0.0
        return float(np.max(spectrum[left:right]))

    def _score_note_raw(
        self,
        spectrum: np.ndarray,
        midi_note: int,
        cents_tolerance: float,
    ) -> tuple[float, float | None, float]:
        """Return (harmonic_score, cents_error, fundamental_score). No penalty."""
        fundamental = midi_to_freq(midi_note)
        if fundamental <= 0 or fundamental >= self.sample_rate / 2:
            return 0.0, None, 0.0
        harmonic_energies = [
            self._freq_energy(spectrum, fundamental * h, cents_tolerance)
            for h in range(1, _NUM_HARMONICS + 1)
        ]
        weighted = np.array(harmonic_energies, dtype=np.float32) * _HARMONIC_WEIGHTS
        total = float(np.sum(weighted))
        peak = float(np.max(spectrum)) if np.max(spectrum) > 0 else 1.0
        score = total / peak if peak > 0 else 0.0
        fundamental_score = harmonic_energies[0] / peak if peak > 0 else 0.0
        cents_error = self._cents_error(spectrum, fundamental, cents_tolerance)
        return score, cents_error, fundamental_score

    def _score_note(
        self,
        spectrum: np.ndarray,
        midi_note: int,
        cents_tolerance: float,
        mode: MatchMode = MatchMode.ARCADE,
    ) -> tuple[float, float]:
        """Return (harmonic_score, cents_error) for an expected note."""
        score, cents_error, _ = self._score_note_raw(
            spectrum, midi_note, cents_tolerance
        )
        if score <= 0.0:
            return score, cents_error

        fundamental = midi_to_freq(midi_note)
        harmonic_energies = [
            self._freq_energy(spectrum, fundamental * h, cents_tolerance)
            for h in range(1, _NUM_HARMONICS + 1)
        ]
        peak = float(np.max(spectrum)) if np.max(spectrum) > 0 else 1.0
        above = sum(1 for e, w in zip(harmonic_energies, _HARMONIC_WEIGHTS)
                    if e > 1e-6 and e * w > 0.01 * peak)
        min_needed = _MIN_HARMONICS_PRESENT.get(mode, 2)
        if above < min_needed:
            # JUDGE requires a real harmonic series; lower modes just get penalized.
            score = 0.0 if mode == MatchMode.JUDGE else score * 0.5

        return score, cents_error

    def _anti_alias_check(
        self,
        spectrum: np.ndarray,
        midi_note: int,
        target_score: float,
        cents_tolerance: float,
        mode: MatchMode,
    ) -> tuple[bool, float]:
        """Return (passes, best_competitor_score).

        A competitor is only dangerous if it has a credible fundamental. This
        prevents a note whose 2nd harmonic lands on a competitor from causing
        an ambiguity rejection.
        """
        min_f0 = _MIN_F0_SCORE.get(mode, 0.08)
        best_competitor = 0.0
        target_f0 = self._score_note_raw(
            spectrum, midi_note, cents_tolerance
        )[2]
        for offset in _COMPETITOR_OFFSETS:
            competitor_midi = midi_note + offset
            if competitor_midi < 23 or competitor_midi > 96:
                continue
            comp_score, _, comp_f0 = self._score_note_raw(
                spectrum, competitor_midi, cents_tolerance
            )
            if comp_f0 < min_f0:
                continue
            # Near-neighbor competitors (±1, ±2) can pick up the target's
            # own F0 through spectral leakage when the target's cents band
            # overlaps the competitor's. Skip only when both have very strong
            # F0 AND the target's F0 is at least as strong as the competitor's
            # (indicating a shared peak, not an independent fundamental).
            if abs(offset) <= 2 and target_f0 > 0.5 and comp_f0 > 0.5 and target_f0 >= comp_f0:
                continue
            best_competitor = max(best_competitor, comp_score)
        margin = _ANTI_ALIAS_MARGINS.get(mode, 0.08)
        return target_score - best_competitor >= margin, best_competitor

    def _cents_error(
        self, spectrum: np.ndarray, fundamental: float, cents_tolerance: float
    ) -> float | None:
        """Estimate pitch deviation from the nearest FFT bin peak."""
        if fundamental <= 0 or fundamental >= self.sample_rate / 2:
            return None
        lo, hi = cents_band(fundamental, cents_tolerance)
        left = int(np.searchsorted(self._freqs, lo, side="left"))
        right = int(np.searchsorted(self._freqs, hi, side="right"))
        if left >= right:
            return None
        band = spectrum[left:right]
        if np.max(band) <= 0:
            return None
        peak_idx = left + int(np.argmax(band))
        measured = self._freqs[peak_idx]
        if measured <= 0:
            return None
        return 1200.0 * np.log2(measured / fundamental)

    def _near_detected_peak(self, spectrum: np.ndarray, midi_note: int) -> bool:
        """True if the strongest spectral peak is within +/-2 semitones of midi_note."""
        if np.max(spectrum) <= 0:
            return False
        peak_idx = int(np.argmax(spectrum))
        peak_freq = self._freqs[peak_idx]
        if peak_freq <= 0:
            return False
        peak_midi = freq_to_midi(peak_freq)
        return abs(peak_midi - midi_note) <= 2

    def _detect_onset(
        self,
        audio_window: np.ndarray,
        target_freq: float,
        cents_tolerance: float,
        expected_onset_offset_ms: float | None = None,
        onset_tolerance_ms: float | None = None,
    ) -> tuple[bool, float | None]:
        """Return (is_onset, onset_ms) using spectral flux in the target band."""
        if len(audio_window) < int(self.sample_rate * 0.06):
            return False, None
        hop = int(self.sample_rate * 0.01)
        win = int(self.sample_rate * 0.02)
        energies: list[float] = []
        for start in range(0, len(audio_window) - win + 1, hop):
            frame = audio_window[start:start + win]
            spec = np.abs(np.fft.rfft(frame * np.hanning(win)))
            freqs = np.fft.rfftfreq(win, 1.0 / self.sample_rate)
            lo, hi = cents_band(target_freq, cents_tolerance)
            left = int(np.searchsorted(freqs, lo, side="left"))
            right = int(np.searchsorted(freqs, hi, side="right"))
            # Ensure at least the nearest bin is included (short windows can skip narrow bands).
            if left >= right:
                left = max(0, left - 1)
                right = min(len(freqs), right + 1)
            if left < right:
                energies.append(float(np.max(spec[left:right])))
            else:
                energies.append(0.0)
        if len(energies) < 3:
            return False, None
        flux = [max(0.0, energies[i] - energies[i - 1]) for i in range(1, len(energies))]
        if not flux:
            return False, None
        peak_flux = max(flux)
        peak_idx = flux.index(peak_flux)
        hop_ms = hop / self.sample_rate * 1000.0
        if expected_onset_offset_ms is not None:
            expected_idx = int(expected_onset_offset_ms / hop_ms)
            tolerance = int((onset_tolerance_ms or 50.0) / hop_ms)
            is_onset = (
                abs(peak_idx - expected_idx) <= tolerance
                and peak_flux > 0.1 * max(energies)
            )
        else:
            is_onset = peak_idx < len(flux) / 2 and peak_flux > 0.1 * max(energies)
        onset_ms = (peak_idx * hop / self.sample_rate * 1000.0) if is_onset else None
        return is_onset, onset_ms

    def _detect_broadband_onset(
        self,
        audio_window: np.ndarray,
        expected_onset_offset_ms: float | None = None,
        onset_tolerance_ms: float | None = None,
    ) -> tuple[bool, float | None]:
        """Return (is_onset, onset_ms) using broadband RMS flux.

        Shared chord strum detection: a strum produces a sharp rise in total
        energy across all frequencies, not just the lowest chord note.
        """
        if len(audio_window) < int(self.sample_rate * 0.06):
            return False, None
        hop = int(self.sample_rate * 0.01)
        win = int(self.sample_rate * 0.02)
        rms_values: list[float] = []
        for start in range(0, len(audio_window) - win + 1, hop):
            frame = audio_window[start:start + win]
            rms = float(np.sqrt(np.mean(frame * frame)))
            rms_values.append(rms)
        if len(rms_values) < 3:
            return False, None
        flux = [max(0.0, rms_values[i] - rms_values[i - 1])
                for i in range(1, len(rms_values))]
        if not flux:
            return False, None
        peak_flux = max(flux)
        peak_idx = flux.index(peak_flux)
        peak_rms = max(rms_values)
        hop_ms = hop / self.sample_rate * 1000.0
        if expected_onset_offset_ms is not None:
            expected_idx = int(expected_onset_offset_ms / hop_ms)
            tolerance = int((onset_tolerance_ms or 50.0) / hop_ms)
            is_onset = (
                abs(peak_idx - expected_idx) <= tolerance
                and peak_flux > 0.1 * peak_rms
            )
        else:
            is_onset = peak_idx < len(flux) / 2 and peak_flux > 0.1 * peak_rms
        onset_ms = (peak_idx * hop / self.sample_rate * 1000.0) if is_onset else None
        return is_onset, onset_ms

    def verify_single_note(
        self,
        audio_window: np.ndarray,
        expected_midi: int,
        mode: MatchMode,
        expected_onset_offset_ms: float | None = None,
        onset_tolerance_ms: float | None = None,
    ) -> NoteVerification:
        """Verify a single expected note via harmonic-bank detection."""
        if len(audio_window) == 0:
            return NoteVerification(
                is_pitch_present=False,
                is_onset_present=False,
                pitch_evidence=None,
                onset_ms=None,
                harmonic_score=0.0,
                timing_error_ms=None,
                alias_risk=0.0,
            )

        policy = VerificationPolicy.from_mode(mode)
        spectrum = self._spectrum(audio_window)
        cents_tolerance = policy.pitch_cents_tolerance

        score, cents_error = self._score_note(
            spectrum, expected_midi, cents_tolerance, mode
        )
        raw_score, _, fundamental_score = self._score_note_raw(
            spectrum, expected_midi, cents_tolerance
        )
        anti_alias_passes, best_competitor = self._anti_alias_check(
            spectrum, expected_midi, raw_score, cents_tolerance, mode
        )
        threshold = _HARMONIC_SCORE_THRESHOLDS.get(mode, 0.35)

        min_f0 = _MIN_F0_SCORE.get(mode, 0.08)
        f0_ok = fundamental_score >= min_f0
        if not f0_ok and policy.allow_semitone_fallback:
            f0_ok = self._near_detected_peak(spectrum, expected_midi)

        is_pitch_present = (
            score > threshold
            and anti_alias_passes
            and f0_ok
        )
        alias_risk = (
            best_competitor / raw_score
            if raw_score > 0
            else (0.0 if best_competitor <= 0 else 1.0)
        )

        fundamental = midi_to_freq(expected_midi)
        is_onset_present, onset_ms = self._detect_onset(
            audio_window, fundamental, cents_tolerance,
            expected_onset_offset_ms=expected_onset_offset_ms,
            onset_tolerance_ms=onset_tolerance_ms,
        )

        return NoteVerification(
            is_pitch_present=is_pitch_present,
            is_onset_present=is_onset_present,
            pitch_evidence=PitchEvidence(
                midi_note=expected_midi,
                cents_error=cents_error,
                confidence=min(1.0, score),
                source="spectral",
            ),
            onset_ms=onset_ms,
            harmonic_score=score,
            timing_error_ms=None,
            alias_risk=alias_risk,
        )

    def verify_chord(
        self,
        audio_window: np.ndarray,
        expected_notes: list[ExpectedNote],
        mode: MatchMode,
        expected_onset_offset_ms: float | None = None,
        onset_tolerance_ms: float | None = None,
    ) -> ChordVerification:
        """Verify an expected chord via per-note harmonic-bank detection."""
        notes: list[NoteVerification] = []
        if len(audio_window) == 0:
            return ChordVerification(
                notes=notes, partial=False, total_harmonic_energy=0.0
            )

        policy = VerificationPolicy.from_mode(mode)
        spectrum = self._spectrum(audio_window)
        peak = float(np.max(spectrum)) if np.max(spectrum) > 0 else 1.0
        total_energy = 0.0
        shared_onset_present, shared_onset_ms = self._detect_broadband_onset(
            audio_window, expected_onset_offset_ms, onset_tolerance_ms
        )
        for idx, en in enumerate(expected_notes):
            score, cents_error = self._score_note(
                spectrum, en.midi, policy.pitch_cents_tolerance, mode
            )
            note_energy = score * peak
            total_energy += note_energy
            threshold = _HARMONIC_SCORE_THRESHOLDS.get(mode, 0.35)
            notes.append(NoteVerification(
                is_pitch_present=score >= threshold,
                is_onset_present=shared_onset_present,
                pitch_evidence=PitchEvidence(
                    midi_note=en.midi,
                    cents_error=cents_error,
                    confidence=min(1.0, score),
                    source="spectral",
                ),
                onset_ms=shared_onset_ms,
                harmonic_score=score,
                timing_error_ms=None,
                alias_risk=0.0,
            ))

        min_ratio = _MIN_ENERGY_RATIO.get(mode, 0.08)
        if total_energy > 0:
            for note in notes:
                note_score = note.harmonic_score * peak
                if note_score / total_energy < min_ratio:
                    note.is_pitch_present = False

        partial = any(n.is_pitch_present for n in notes) and not all(n.is_pitch_present for n in notes)
        return ChordVerification(
            notes=notes,
            partial=partial,
            total_harmonic_energy=total_energy,
        )

    def verify_technique(
        self,
        audio_window: np.ndarray,
        expected: str,
        context: dict,
    ) -> TechniqueVerification:
        """Technique verification is delegated to technique-specific verifiers.

        The spectral verifier does not handle techniques; it returns a
        non-present verdict so that a dedicated technique verifier can
        override this later.
        """
        return TechniqueVerification(
            technique=expected,
            is_present=False,
            confidence=0.0,
            details={"reason": "spectral verifier does not detect techniques"},
        )

    def verify_silence(self, audio_window: np.ndarray, threshold_db: float) -> bool:
        """Return True if the window's RMS is below threshold_db."""
        if len(audio_window) == 0:
            return True
        rms = float(np.sqrt(np.mean(audio_window**2)))
        db = 20.0 * np.log10(rms + 1e-10)
        return db < threshold_db
