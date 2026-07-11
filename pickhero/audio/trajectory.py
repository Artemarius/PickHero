"""Robust monophonic pitch trajectories for articulation grading.

This module is intentionally separate from the real-time detector.  It performs
bounded, after-the-fact analysis on the authored note window and exposes a
smoothed pitch path suitable for bend, slide and vibrato quality scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pickhero.audio.note_utils import midi_to_freq


@dataclass(frozen=True)
class PitchTrajectory:
    times_ms: np.ndarray
    midi: np.ndarray
    periodicity: np.ndarray

    @property
    def valid(self) -> bool:
        return len(self.midi) >= 3

    @property
    def duration_ms(self) -> float:
        if len(self.times_ms) < 2:
            return 0.0
        return float(self.times_ms[-1] - self.times_ms[0])

    def cents_from(self, midi_reference: float) -> np.ndarray:
        return (self.midi - float(midi_reference)) * 100.0


def _parabolic_peak(values: np.ndarray, index: int) -> float:
    """Sub-sample peak interpolation, returning a float array index."""
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left = float(values[index - 1])
    center = float(values[index])
    right = float(values[index + 1])
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return float(index)
    return float(index) + 0.5 * (left - right) / denominator


def _median_smooth(values: np.ndarray, radius: int = 1) -> np.ndarray:
    if len(values) < 3 or radius <= 0:
        return values.copy()
    output = np.empty_like(values)
    for index in range(len(values)):
        lo = max(0, index - radius)
        hi = min(len(values), index + radius + 1)
        output[index] = np.median(values[lo:hi])
    return output


def _expected_frequency_bounds(expected_midis: tuple[float, ...]) -> tuple[float, float]:
    if not expected_midis:
        return 55.0, 1500.0
    lo = min(expected_midis) - 5.0
    hi = max(expected_midis) + 5.0
    return max(25.0, midi_to_freq(lo)), min(1800.0, midi_to_freq(hi))


def estimate_pitch_trajectory(
    audio_window: np.ndarray,
    sample_rate: int,
    *,
    expected_midis: tuple[float, ...] = (),
    hop_ms: float = 10.0,
    frame_ms: float = 42.0,
    minimum_periodicity: float = 0.28,
) -> PitchTrajectory:
    """Estimate a smoothed pitch contour using normalized autocorrelation.

    The search range is constrained by the authored pitches when available.
    This is important for guitar: unconstrained autocorrelation frequently
    chooses an octave or a strong harmonic during bends and distorted notes.
    """
    samples = np.asarray(audio_window, dtype=np.float32).reshape(-1)
    if sample_rate <= 0:
        empty = np.zeros(0, dtype=np.float64)
        return PitchTrajectory(empty, empty, empty)

    min_freq, max_freq = _expected_frequency_bounds(expected_midis)
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    requested_frame = int(sample_rate * frame_ms / 1000.0)
    # Low guitar and bass notes need enough cycles for stable autocorrelation.
    # Keep the authored 42 ms response for normal guitar while extending only
    # as far as needed for low tunings, capped to avoid smearing transitions.
    pitch_frame = int(np.ceil(sample_rate * 2.25 / max(min_freq, 1.0)))
    frame_size = max(64, requested_frame, pitch_frame)
    frame_size = min(frame_size, max(64, int(sample_rate * 0.12)))
    if len(samples) < frame_size:
        empty = np.zeros(0, dtype=np.float64)
        return PitchTrajectory(empty, empty, empty)

    min_lag = max(1, int(sample_rate / max_freq))
    max_lag = min(frame_size - 2, int(sample_rate / min_freq))
    if max_lag <= min_lag:
        empty = np.zeros(0, dtype=np.float64)
        return PitchTrajectory(empty, empty, empty)

    window = np.hanning(frame_size).astype(np.float32)
    fft_size = 1 << (frame_size * 2 - 1).bit_length()
    times: list[float] = []
    midis: list[float] = []
    periodicities: list[float] = []
    for start in range(0, len(samples) - frame_size + 1, hop):
        frame = samples[start:start + frame_size].astype(np.float64, copy=False)
        frame = frame - float(np.mean(frame))
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms < 1e-5:
            continue
        frame = frame * window
        # FFT autocorrelation keeps long bend/vibrato windows practical. The
        # direct O(N²) correlation used previously became a frame-time spike
        # when finalizing sustained techniques.
        spectrum = np.fft.rfft(frame, n=fft_size)
        correlation = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[:frame_size]
        zero = float(correlation[0])
        if zero <= 1e-12:
            continue
        normalized = correlation / zero
        search = normalized[min_lag:max_lag + 1]
        if len(search) == 0:
            continue
        local_index = int(np.argmax(search))
        peak_index = local_index + min_lag
        periodicity = float(normalized[peak_index])
        if periodicity < minimum_periodicity:
            continue
        lag = _parabolic_peak(normalized, peak_index)
        if lag <= 0.0:
            continue
        frequency = sample_rate / lag
        midi = float(69.0 + 12.0 * np.log2(frequency / 440.0))
        if not np.isfinite(midi) or midi < 20.0 or midi > 110.0:
            continue
        times.append((start + frame_size * 0.5) / sample_rate * 1000.0)
        midis.append(midi)
        periodicities.append(periodicity)

    if not midis:
        empty = np.zeros(0, dtype=np.float64)
        return PitchTrajectory(empty, empty, empty)

    midi_array = np.asarray(midis, dtype=np.float64)
    midi_array = _median_smooth(midi_array, radius=1)

    # Reject isolated octave/harmonic jumps without erasing genuine slides.
    if len(midi_array) >= 3:
        cleaned = midi_array.copy()
        for index in range(1, len(cleaned) - 1):
            neighbours = 0.5 * (cleaned[index - 1] + cleaned[index + 1])
            if abs(cleaned[index] - neighbours) > 5.5:
                cleaned[index] = neighbours
        midi_array = _median_smooth(cleaned, radius=1)

    return PitchTrajectory(
        times_ms=np.asarray(times, dtype=np.float64),
        midi=midi_array,
        periodicity=np.asarray(periodicities, dtype=np.float64),
    )


def direction_consistency(values: np.ndarray, expected_direction: int) -> float:
    """Fraction of meaningful contour steps moving in the expected direction."""
    if len(values) < 2 or expected_direction == 0:
        return 0.0
    deltas = np.diff(values)
    meaningful = np.abs(deltas) >= 2.0  # cents; ignore detector jitter
    if not np.any(meaningful):
        return 0.0
    signed = deltas[meaningful] * float(expected_direction)
    return float(np.mean(signed >= -3.0))


def landing_stability(values: np.ndarray, fraction: float = 0.22) -> float:
    """Return 0..1 stability of the final trajectory segment."""
    if len(values) < 3:
        return 0.0
    count = max(3, int(round(len(values) * fraction)))
    spread = float(np.percentile(values[-count:], 90) - np.percentile(values[-count:], 10))
    return max(0.0, min(1.0, 1.0 - spread / 55.0))


def normalized_curve_error(
    times_ms: np.ndarray,
    observed_cents: np.ndarray,
    authored_curve: tuple[tuple[float, float], ...],
) -> float | None:
    """Mean absolute curve error normalized to 0..1.

    Guitar Pro bend-point x coordinates vary by source format. They are treated
    as relative positions and normalized to the observed trajectory duration.
    """
    if len(times_ms) < 3 or len(authored_curve) < 2:
        return None
    points = sorted((float(x), float(y)) for x, y in authored_curve)
    xs = np.asarray([point[0] for point in points], dtype=np.float64)
    ys = np.asarray([point[1] for point in points], dtype=np.float64)
    span = float(xs[-1] - xs[0])
    if span <= 1e-9:
        return None
    authored_position = (xs - xs[0]) / span
    observed_span = float(times_ms[-1] - times_ms[0])
    if observed_span <= 1e-9:
        return None
    observed_position = (times_ms - times_ms[0]) / observed_span
    expected = np.interp(observed_position, authored_position, ys)
    mae = float(np.mean(np.abs(observed_cents - expected)))
    return max(0.0, min(1.0, 1.0 - mae / 80.0))
