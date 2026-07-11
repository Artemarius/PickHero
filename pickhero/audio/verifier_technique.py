"""Technique verifiers for expected-event verification.

Each verifier receives a raw audio window and expected technique context,
then returns a :class:`~pickhero.audio.evidence.TechniqueVerification`.

These are second-look verifiers: they run on the full audio window after the
fact, independent of the real-time articulation detector.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pickhero.audio.evidence import TechniqueVerification
from pickhero.audio.note_utils import midi_to_freq
from pickhero.audio.trajectory import (
    direction_consistency,
    estimate_pitch_trajectory,
    landing_stability,
    normalized_curve_error,
)


def _pitch_contour(
    audio_window: np.ndarray,
    sample_rate: int,
    *,
    expected_midis: tuple[float, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper around the robust articulation trajectory."""
    trajectory = estimate_pitch_trajectory(
        audio_window, sample_rate, expected_midis=expected_midis
    )
    return trajectory.times_ms, trajectory.midi


def _rms(audio_window: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio_window**2)))


def _spectral_decay_slope(audio_window: np.ndarray, sample_rate: int) -> float:
    """Return slope of spectral envelope (more negative = faster decay)."""
    n = len(audio_window)
    if n < 256:
        return 0.0
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(audio_window * window))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    valid = freqs > 80
    if np.sum(valid) < 2:
        return 0.0
    log_freqs = np.log(freqs[valid] + 1e-6)
    log_mag = np.log(spectrum[valid] + 1e-10)
    slope, _ = np.polyfit(log_freqs, log_mag, 1)
    return float(slope)


def _harmonic_ratio(audio_window: np.ndarray, sample_rate: int, fundamental_midi: int | None) -> float:
    """Ratio of harmonic energy to total energy."""
    n = len(audio_window)
    if n < 256 or fundamental_midi is None:
        return 0.0
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(audio_window * window))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    f0 = midi_to_freq(fundamental_midi)
    if f0 <= 0:
        return 0.0
    total = float(np.sum(spectrum)) + 1e-10
    harmonic = 0.0
    for h in range(1, 6):
        hf = f0 * h
        if hf >= sample_rate / 2:
            break
        idx = int(np.argmin(np.abs(freqs - hf)))
        harmonic += float(spectrum[idx])
    return harmonic / total


class TechniqueVerifier:
    """Dispatcher for technique-specific verification."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def verify(
        self, audio_window: np.ndarray, technique: str, context: dict
    ) -> TechniqueVerification:
        """Dispatch and convert binary heuristics into tri-state evidence.

        A weak negative is uncertainty, not proof that the player omitted the
        technique. The base-note scorer consumes technique quality separately.
        """
        method = getattr(self, f"_verify_{technique}", self._verify_unknown)
        result = method(audio_window, context)
        confidence = max(0.0, min(1.0, float(result.confidence)))
        if result.is_present:
            return replace(
                result,
                confidence=confidence,
                uncertain=False,
                quality=result.quality if result.quality is not None else confidence,
            )
        if result.uncertain:
            # The verifier explicitly reported insufficient evidence. Preserve
            # that distinction rather than converting a zero-confidence result
            # into a failed articulation grade.
            return replace(
                result,
                confidence=confidence,
                uncertain=True,
                quality=None,
            )
        uncertain_floor = float(context.get("uncertain_floor", 0.28))
        return replace(
            result,
            confidence=confidence,
            uncertain=confidence >= uncertain_floor,
            quality=result.quality if result.quality is not None else confidence,
        )

    def _verify_bend(
        self, audio_window: np.ndarray, context: dict
    ) -> TechniqueVerification:
        """Grade bend target, direction, hold and authored trajectory shape."""
        base_midi = float(context.get("midi_note", 0.0) or 0.0)
        subtype = str(context.get("subtype") or "bend")
        curve = tuple(context.get("curve") or ())
        target_cents = float(context.get("target_cents") or 0.0)
        if curve:
            curve_values = [float(point[1]) for point in curve]
            if abs(target_cents) < 1.0:
                target_cents = max(curve_values, key=abs)
        target_cents = abs(target_cents) if abs(target_cents) >= 1.0 else 100.0

        expected = (base_midi, base_midi + target_cents / 100.0)
        trajectory = estimate_pitch_trajectory(
            audio_window, self.sample_rate, expected_midis=expected
        )
        if not trajectory.valid or base_midi <= 0.0:
            return TechniqueVerification(
                "bend",
                False,
                0.0,
                {"reason": "insufficient pitch trajectory"},
                uncertain=True,
                quality=None,
            )

        cents = trajectory.cents_from(base_midi)
        head_count = max(2, int(round(len(cents) * 0.18)))
        tail_count = max(3, int(round(len(cents) * 0.24)))
        start_cents = float(np.median(cents[:head_count]))
        end_cents = float(np.median(cents[-tail_count:]))
        peak_cents = float(np.percentile(cents, 95))
        target_error = float(np.min(np.abs(cents - target_cents)))
        target_accuracy = max(0.0, min(1.0, 1.0 - target_error / 65.0))
        curve_quality = normalized_curve_error(trajectory.times_ms, cents, curve)

        if subtype == "release":
            start_accuracy = max(0.0, min(1.0, 1.0 - abs(start_cents - target_cents) / 70.0))
            release_accuracy = max(0.0, min(1.0, 1.0 - abs(end_cents) / 70.0))
            direction = direction_consistency(cents, -1)
            quality = 0.34 * start_accuracy + 0.36 * release_accuracy + 0.22 * direction
            if curve_quality is not None:
                quality += 0.08 * curve_quality
            else:
                quality += 0.08 * min(start_accuracy, release_accuracy)
        elif subtype == "pre":
            start_accuracy = max(0.0, min(1.0, 1.0 - abs(start_cents - target_cents) / 65.0))
            hold_error = float(np.median(np.abs(cents - target_cents)))
            hold_quality = max(0.0, min(1.0, 1.0 - hold_error / 65.0))
            quality = 0.46 * start_accuracy + 0.46 * hold_quality
            quality += 0.08 * (curve_quality if curve_quality is not None else hold_quality)
            direction = 1.0
            release_accuracy = None
        else:
            coverage = max(0.0, min(1.0, peak_cents / max(target_cents, 1.0)))
            direction = direction_consistency(cents, 1)
            near_target = float(np.mean(np.abs(cents[-tail_count:] - target_cents) <= 55.0))
            quality = 0.34 * target_accuracy + 0.25 * coverage + 0.20 * direction + 0.13 * near_target
            quality += 0.08 * (curve_quality if curve_quality is not None else target_accuracy)
            start_accuracy = max(0.0, min(1.0, 1.0 - abs(start_cents) / 80.0))
            release_accuracy = None

        quality = max(0.0, min(1.0, quality))
        return TechniqueVerification(
            "bend",
            quality >= 0.62,
            quality,
            {
                "subtype": subtype,
                "target_cents": target_cents,
                "start_cents": start_cents,
                "end_cents": end_cents,
                "peak_cents": peak_cents,
                "target_error_cents": target_error,
                "target_accuracy": target_accuracy,
                "direction_consistency": direction,
                "start_accuracy": start_accuracy,
                "release_accuracy": release_accuracy,
                "curve_quality": curve_quality,
            },
            quality=quality,
        )

    def _verify_slide(
        self, audio_window: np.ndarray, context: dict
    ) -> TechniqueVerification:
        """Grade slide direction, travel, continuity and destination landing."""
        start_midi = context.get("start_midi")
        end_midi = context.get("end_midi")
        subtype = str(context.get("subtype") or "shift")
        target_cents = context.get("target_cents")
        if start_midi is None:
            return TechniqueVerification(
                "slide",
                False,
                0.0,
                {"reason": "missing authored start pitch"},
                uncertain=True,
                quality=None,
            )
        start_midi = float(start_midi)
        if end_midi is None and target_cents is not None:
            end_midi = start_midi + float(target_cents) / 100.0
        expected = (start_midi,) if end_midi is None else (start_midi, float(end_midi))
        trajectory = estimate_pitch_trajectory(
            audio_window, self.sample_rate, expected_midis=expected
        )
        if not trajectory.valid:
            return TechniqueVerification(
                "slide",
                False,
                0.0,
                {"reason": "insufficient pitch trajectory"},
                uncertain=True,
                quality=None,
            )

        cents = trajectory.cents_from(start_midi)
        end_segment = max(3, int(round(len(cents) * 0.22)))
        landing_cents = float(np.median(cents[-end_segment:]))
        moved_cents = float(np.percentile(cents, 95) - np.percentile(cents, 5))
        if end_midi is not None:
            expected_delta = (float(end_midi) - start_midi) * 100.0
        elif target_cents is not None:
            expected_delta = float(target_cents)
        else:
            expected_delta = landing_cents

        direction_sign = 1 if expected_delta > 0 else (-1 if expected_delta < 0 else 0)
        direction = direction_consistency(cents, direction_sign)
        coverage = (
            max(0.0, min(1.0, moved_cents / max(abs(expected_delta), 1.0)))
            if abs(expected_delta) >= 25.0 else 0.0
        )
        deltas = np.abs(np.diff(cents))
        continuity = float(np.mean(deltas <= 220.0)) if len(deltas) else 0.0
        stability = landing_stability(cents)

        if end_midi is not None and subtype != "slide_out":
            landing_error = abs(landing_cents - expected_delta)
            landing_accuracy = max(0.0, min(1.0, 1.0 - landing_error / 85.0))
            quality = (
                0.30 * landing_accuracy
                + 0.25 * coverage
                + 0.22 * direction
                + 0.13 * continuity
                + 0.10 * stability
            )
        else:
            landing_error = None
            landing_accuracy = None
            motion = max(0.0, min(1.0, moved_cents / 180.0))
            quality = 0.42 * motion + 0.32 * direction + 0.16 * continuity + 0.10 * stability

        quality = max(0.0, min(1.0, quality))
        return TechniqueVerification(
            "slide",
            quality >= 0.60,
            quality,
            {
                "subtype": subtype,
                "expected_delta_cents": expected_delta,
                "moved_cents": moved_cents,
                "landing_cents": landing_cents,
                "landing_error_cents": landing_error,
                "landing_accuracy": landing_accuracy,
                "coverage": coverage,
                "direction_consistency": direction,
                "continuity": continuity,
                "landing_stability": stability,
            },
            quality=quality,
        )

    def _verify_vibrato(
        self, audio_window: np.ndarray, context: dict
    ) -> TechniqueVerification:
        """Grade vibrato rate, depth, periodicity and pitch-centre stability."""
        expected_midi = float(context.get("midi_note", 0.0) or 0.0)
        trajectory = estimate_pitch_trajectory(
            audio_window,
            self.sample_rate,
            expected_midis=(expected_midi,) if expected_midi > 0.0 else (),
        )
        if not trajectory.valid or trajectory.duration_ms < 260.0:
            return TechniqueVerification(
                "vibrato",
                False,
                0.0,
                {"reason": "trajectory too short"},
                uncertain=True,
                quality=None,
            )

        times = trajectory.times_ms
        midis = trajectory.midi
        slope, intercept = np.polyfit(times, midis, 1)
        detrended_cents = (midis - (slope * times + intercept)) * 100.0
        depth_cents = float(
            (np.percentile(detrended_cents, 90) - np.percentile(detrended_cents, 10)) * 0.5
        )

        # Interpolate missing frames to a regular 10 ms grid before spectral
        # rate estimation. The previous implementation treated the first
        # autocorrelation bump as the vibrato period, which often measured
        # detector jitter instead of hand motion.
        step_ms = 10.0
        regular_times = np.arange(times[0], times[-1] + step_ms * 0.5, step_ms)
        regular = np.interp(regular_times, times, detrended_cents)
        regular -= float(np.mean(regular))
        spectrum = np.abs(np.fft.rfft(regular * np.hanning(len(regular)))) ** 2
        frequencies = np.fft.rfftfreq(len(regular), d=step_ms / 1000.0)
        band = (frequencies >= 3.0) & (frequencies <= 8.5)
        if not np.any(band) or float(np.sum(spectrum)) <= 1e-12:
            return TechniqueVerification(
                "vibrato", False, 0.0, {"reason": "no periodic modulation"}
            )
        band_indices = np.flatnonzero(band)
        peak_index = int(band_indices[np.argmax(spectrum[band])])
        rate_hz = float(frequencies[peak_index])
        periodicity = float(spectrum[peak_index] / (np.sum(spectrum[1:]) + 1e-12))
        periodicity_quality = max(0.0, min(1.0, periodicity / 0.42))
        depth_quality = max(0.0, min(1.0, (depth_cents - 7.0) / 28.0))
        rate_quality = max(0.0, min(1.0, 1.0 - abs(rate_hz - 5.5) / 3.0))
        centre_drift_cents = abs(float(slope)) * trajectory.duration_ms * 100.0
        centre_quality = max(0.0, min(1.0, 1.0 - centre_drift_cents / 95.0))
        quality = (
            0.34 * periodicity_quality
            + 0.30 * depth_quality
            + 0.22 * rate_quality
            + 0.14 * centre_quality
        )
        quality = max(0.0, min(1.0, quality))
        return TechniqueVerification(
            "vibrato",
            quality >= 0.58,
            quality,
            {
                "rate_hz": rate_hz,
                "depth_cents": depth_cents,
                "periodicity": periodicity,
                "centre_drift_cents": centre_drift_cents,
                "periodicity_quality": periodicity_quality,
                "depth_quality": depth_quality,
                "rate_quality": rate_quality,
                "centre_quality": centre_quality,
            },
            quality=quality,
        )

    def _verify_hammer_on(
        self, audio_window: np.ndarray, context: dict
    ) -> TechniqueVerification:
        """Look for pitch step up without a strong new onset."""
        return self._verify_legato(
            audio_window, context, technique="hammer_on", direction="up"
        )

    def _verify_pull_off(
        self, audio_window: np.ndarray, context: dict
    ) -> TechniqueVerification:
        """Look for pitch step down without a strong new onset."""
        return self._verify_legato(
            audio_window, context, technique="pull_off", direction="down"
        )

    def _verify_legato(
        self,
        audio_window: np.ndarray,
        context: dict,
        *,
        technique: str,
        direction: str,
    ) -> TechniqueVerification:
        """Detect a directional pitch transition without a second pick."""
        start_midi = context.get("start_midi")
        end_midi = context.get("end_midi")
        expected = tuple(
            float(value) for value in (start_midi, end_midi) if value is not None
        )
        times, midis = _pitch_contour(
            audio_window, self.sample_rate, expected_midis=expected
        )
        if len(midis) < 4:
            return TechniqueVerification(
                technique,
                False,
                0.0,
                {"reason": "insufficient transition contour"},
                uncertain=True,
                quality=None,
            )
        # Split window in half and compare median pitch.
        half = len(midis) // 2
        first = float(np.median(midis[:half]))
        second = float(np.median(midis[half:]))
        delta = second - first
        moved = abs(delta) >= 0.5
        direction_ok = (direction == "up" and delta > 0) or (direction == "down" and delta < 0)
        # Estimate onset strength from RMS ratio of halves (rough proxy).
        mid = len(audio_window) // 2
        rms_first = _rms(audio_window[:mid])
        rms_second = _rms(audio_window[mid:])
        onset_ratio = (rms_second / (rms_first + 1e-10)) if rms_first > 0 else 1.0
        low_onset = onset_ratio < 2.0
        transition_quality = min(1.0, abs(delta) / 2.0) if direction_ok else 0.0
        onset_quality = max(0.0, min(1.0, 1.0 - max(0.0, onset_ratio - 1.15) / 1.25))
        quality = transition_quality * 0.68 + onset_quality * 0.32
        reached = moved and direction_ok and low_onset and quality >= 0.55
        return TechniqueVerification(
            technique,
            reached,
            quality,
            {
                "direction": direction,
                "delta_semitones": delta,
                "onset_ratio": onset_ratio,
                "transition_quality": transition_quality,
                "onset_quality": onset_quality,
            },
            quality=quality,
        )

    def _verify_palm_mute(
        self, audio_window: np.ndarray, context: dict
    ) -> TechniqueVerification:
        """Fast spectral decay (negative slope) indicates palm mute."""
        slope = _spectral_decay_slope(audio_window, self.sample_rate)
        # Slope more negative than -1.5 is a strong palm-mute signature.
        is_mute = slope < -1.5
        conf = min(1.0, abs(slope) / 3.0) if is_mute else 0.0
        return TechniqueVerification("palm_mute", is_mute, conf, {"spectral_slope": slope})

    def _verify_harmonic(
        self, audio_window: np.ndarray, context: dict
    ) -> TechniqueVerification:
        """High harmonic energy relative to fundamental."""
        midi = context.get("midi_note")
        ratio = _harmonic_ratio(audio_window, self.sample_rate, midi)
        is_harmonic = ratio > 0.6
        conf = min(1.0, ratio)
        return TechniqueVerification("harmonic", is_harmonic, conf, {"harmonic_ratio": ratio})

    def _verify_dead_note(
        self, audio_window: np.ndarray, context: dict
    ) -> TechniqueVerification:
        """Percussive, low-pitch energy, short sustain."""
        rms = _rms(audio_window)
        # Dead notes are quiet and noisy after the initial attack.
        is_dead = rms > 0.001 and _spectral_decay_slope(audio_window, self.sample_rate) < -1.0
        conf = min(1.0, rms * 100) if is_dead else 0.0
        return TechniqueVerification("dead_note", is_dead, conf, {"rms": rms})

    def _verify_unknown(
        self, audio_window: np.ndarray, context: dict
    ) -> TechniqueVerification:
        """Unknown technique — leave it ungraded rather than rejecting it."""
        return TechniqueVerification(
            "unknown",
            False,
            0.0,
            {"reason": "no verifier for authored technique"},
            uncertain=True,
            quality=None,
        )
