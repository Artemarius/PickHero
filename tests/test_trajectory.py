"""Tests for pickhero.audio.trajectory — pitch contour analysis."""

import numpy as np
import pytest

from pickhero.audio.trajectory import (
    PitchTrajectory,
    _median_smooth,
    _parabolic_peak,
    direction_consistency,
    estimate_pitch_trajectory,
    landing_stability,
    normalized_curve_error,
)


# ---------------------------------------------------------------------------
# PitchTrajectory dataclass
# ---------------------------------------------------------------------------


class TestPitchTrajectoryValid:
    def test_true_when_three_or_more_midi_values(self):
        t = PitchTrajectory(
            times_ms=np.array([0.0, 1.0, 2.0]),
            midi=np.array([60.0, 61.0, 62.0]),
            periodicity=np.array([0.9, 0.8, 0.7]),
        )
        assert t.valid is True

    def test_true_with_exactly_three_values(self):
        t = PitchTrajectory(
            times_ms=np.array([0.0, 1.0, 2.0]),
            midi=np.array([60.0, 61.0, 62.0]),
            periodicity=np.array([0.9, 0.8, 0.7]),
        )
        assert t.valid is True

    def test_false_with_two_midi_values(self):
        t = PitchTrajectory(
            times_ms=np.array([0.0, 1.0]),
            midi=np.array([60.0, 61.0]),
            periodicity=np.array([0.9, 0.8]),
        )
        assert t.valid is False

    def test_false_with_one_midi_value(self):
        t = PitchTrajectory(
            times_ms=np.array([0.0]),
            midi=np.array([60.0]),
            periodicity=np.array([0.9]),
        )
        assert t.valid is False

    def test_false_when_empty(self):
        empty = np.zeros(0, dtype=np.float64)
        t = PitchTrajectory(empty, empty, empty)
        assert t.valid is False


class TestPitchTrajectoryDurationMs:
    def test_normal_case(self):
        t = PitchTrajectory(
            times_ms=np.array([10.0, 50.0, 100.0]),
            midi=np.array([60.0, 61.0, 62.0]),
            periodicity=np.array([0.9, 0.8, 0.7]),
        )
        assert t.duration_ms == 90.0

    def test_single_point_returns_zero(self):
        t = PitchTrajectory(
            times_ms=np.array([42.0]),
            midi=np.array([60.0]),
            periodicity=np.array([0.9]),
        )
        assert t.duration_ms == 0.0

    def test_empty_returns_zero(self):
        empty = np.zeros(0, dtype=np.float64)
        t = PitchTrajectory(empty, empty, empty)
        assert t.duration_ms == 0.0

    def test_two_points_calculates_correctly(self):
        t = PitchTrajectory(
            times_ms=np.array([5.0, 95.0]),
            midi=np.array([60.0, 61.0]),
            periodicity=np.array([0.9, 0.8]),
        )
        assert t.duration_ms == 90.0


class TestPitchTrajectoryCentsFrom:
    def test_basic_reference(self):
        t = PitchTrajectory(
            times_ms=np.array([0.0, 1.0, 2.0]),
            midi=np.array([60.0, 62.0, 65.0]),
            periodicity=np.array([0.9, 0.8, 0.7]),
        )
        result = t.cents_from(60.0)
        expected = np.array([0.0, 200.0, 500.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_returns_ndarray(self):
        t = PitchTrajectory(
            times_ms=np.array([0.0]),
            midi=np.array([60.0]),
            periodicity=np.array([0.9]),
        )
        result = t.cents_from(60.0)
        assert isinstance(result, np.ndarray)

    def test_with_negative_reference(self):
        t = PitchTrajectory(
            times_ms=np.array([0.0, 1.0]),
            midi=np.array([60.0, 61.0]),
            periodicity=np.array([0.9, 0.8]),
        )
        result = t.cents_from(60.0)
        expected = np.array([0.0, 100.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_fractional_reference(self):
        t = PitchTrajectory(
            times_ms=np.array([0.0, 1.0, 2.0]),
            midi=np.array([60.0, 61.0, 62.0]),
            periodicity=np.array([0.9, 0.8, 0.7]),
        )
        result = t.cents_from(60.5)
        expected = np.array([-50.0, 50.0, 150.0])
        np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# estimate_pitch_trajectory
# ---------------------------------------------------------------------------


class TestEstimatePitchTrajectory:
    def test_empty_audio_returns_empty(self):
        audio = np.array([], dtype=np.float32)
        result = estimate_pitch_trajectory(audio, sample_rate=44100)
        assert result.valid is False
        assert len(result.midi) == 0
        assert len(result.times_ms) == 0
        assert len(result.periodicity) == 0

    def test_silent_audio_returns_empty(self):
        audio = np.zeros(44100, dtype=np.float32)
        result = estimate_pitch_trajectory(audio, sample_rate=44100)
        assert result.valid is False
        assert len(result.midi) == 0

    def test_zero_sample_rate_returns_empty(self):
        audio = np.random.randn(100).astype(np.float32)
        result = estimate_pitch_trajectory(audio, sample_rate=0)
        assert result.valid is False
        assert len(result.midi) == 0

    def test_negative_sample_rate_returns_empty(self):
        audio = np.random.randn(100).astype(np.float32)
        result = estimate_pitch_trajectory(audio, sample_rate=-1)
        assert result.valid is False
        assert len(result.midi) == 0

    def test_too_short_audio_returns_empty(self):
        """Audio shorter than the computed frame_size returns empty."""
        n = 100  # well below the ~1852 frame minimum at 44100 Hz
        t = np.arange(n, dtype=np.float64) / 44100.0
        audio = (np.sin(2.0 * np.pi * 440.0 * t) * 0.3).astype(np.float32)
        result = estimate_pitch_trajectory(audio, sample_rate=44100)
        assert result.valid is False
        assert len(result.midi) == 0

    def test_pure_sine_estimates_near_e2(self):
        """A pure 82.41 Hz (E2) sine wave yields midi values near 40
        when expected_midis constrains the lag search range.
        """
        sample_rate = 44100
        frequency = 82.41  # E2  —  midi ~ 40.01
        n_samples = int(sample_rate * 2.0)
        t = np.arange(n_samples, dtype=np.float64) / sample_rate
        audio = (np.sin(2.0 * np.pi * frequency * t) * 0.3).astype(np.float32)
        result = estimate_pitch_trajectory(
            audio, sample_rate=sample_rate, expected_midis=(40.0,)
        )
        assert result.valid is True, "Expected valid trajectory for pure sine"
        mean_midi = float(np.mean(result.midi))
        assert 38.0 <= mean_midi <= 42.0, f"Expected midi near 40, got {mean_midi}"
        assert len(result.times_ms) > 0
        assert len(result.midi) > 0
        assert len(result.periodicity) > 0

    def test_pure_sine_with_expected_midis(self):
        """Providing expected_midis does not prevent detection."""
        sample_rate = 44100
        n_samples = int(sample_rate * 2.0)
        t = np.arange(n_samples, dtype=np.float64) / sample_rate
        audio = (np.sin(2.0 * np.pi * 82.41 * t) * 0.3).astype(np.float32)
        result = estimate_pitch_trajectory(
            audio, sample_rate=sample_rate, expected_midis=(40.0,)
        )
        assert result.valid is True
        mean_midi = float(np.mean(result.midi))
        assert 38.0 <= mean_midi <= 42.0, f"Expected midi near 40, got {mean_midi}"


# ---------------------------------------------------------------------------
# direction_consistency
# ---------------------------------------------------------------------------


class TestDirectionConsistency:
    def test_ascending_values_direction_1_returns_one(self):
        values = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        score = direction_consistency(values, expected_direction=1)
        assert score == 1.0

    def test_descending_values_direction_1_returns_zero(self):
        values = np.array([20.0, 15.0, 10.0, 5.0, 0.0])
        score = direction_consistency(values, expected_direction=1)
        assert score == 0.0

    def test_ascending_values_direction_minus1_returns_zero(self):
        values = np.array([0.0, 5.0, 10.0, 15.0])
        score = direction_consistency(values, expected_direction=-1)
        assert score == 0.0

    def test_descending_values_direction_minus1_returns_one(self):
        values = np.array([20.0, 15.0, 10.0, 5.0, 0.0])
        score = direction_consistency(values, expected_direction=-1)
        assert score == 1.0

    def test_direction_zero_returns_zero(self):
        values = np.array([0.0, 5.0, 10.0, 15.0])
        score = direction_consistency(values, expected_direction=0)
        assert score == 0.0

    def test_less_than_two_values_returns_zero(self):
        values = np.array([10.0])
        score = direction_consistency(values, expected_direction=1)
        assert score == 0.0

    def test_empty_returns_zero(self):
        values = np.array([], dtype=np.float64)
        score = direction_consistency(values, expected_direction=1)
        assert score == 0.0

    def test_no_meaningful_steps_returns_zero(self):
        values = np.array([60.0, 60.5, 61.0, 60.8])
        score = direction_consistency(values, expected_direction=1)
        assert score == 0.0

    def test_mixed_directions_partial_score(self):
        values = np.array([0.0, 5.0, 1.0, 7.0, 7.5, 12.0])
        # deltas: [5, -4, 6, 0.5, 4.5]
        # meaningful (abs>=2): [T, T, T, F, T]
        # signed * 1:          [5, -4, 6, -, 4.5]
        # mean(signed >= -3) = (T + F + T + T) / 4 = 0.75
        score = direction_consistency(values, expected_direction=1)
        assert score == 0.75

    def test_all_small_deltas_not_meaningful(self):
        """No delta reaching the 2.0 threshold yields 0.0."""
        values = np.array([60.0, 60.1, 60.2, 60.3, 60.4])
        score = direction_consistency(values, expected_direction=1)
        assert score == 0.0

    def test_some_deltas_barely_above_threshold(self):
        """Deltas exactly at the threshold count as meaningful."""
        values = np.array([0.0, 2.0, 4.0, 6.0])  # deltas exactly 2.0
        score = direction_consistency(values, expected_direction=1)
        assert score == 1.0


# ---------------------------------------------------------------------------
# landing_stability
# ---------------------------------------------------------------------------


class TestLandingStability:
    def test_perfectly_stable_returns_one(self):
        """Identical final values give spread=0 → score=1.0."""
        values = np.array([55.0, 57.0, 60.0, 62.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0])
        score = landing_stability(values)
        # count = max(3, round(10*0.22)) = max(3, 2) = 3
        # last 3 = [63, 63, 63] → spread = 0 → score = 1.0
        assert score == 1.0

    def test_very_unstable_clamps_to_zero(self):
        """Spread > 55 clamps to 0.0."""
        values = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
        score = landing_stability(values)
        # count = max(3, round(6*0.22)) = max(3, 1) = 3
        # last 3 = [300, 400, 500] → spread ≈ 160 → score = 0.0
        assert score == 0.0

    def test_less_than_three_returns_zero(self):
        values = np.array([60.0, 61.0])
        score = landing_stability(values)
        assert score == 0.0

    def test_empty_returns_zero(self):
        values = np.array([], dtype=np.float64)
        score = landing_stability(values)
        assert score == 0.0

    def test_single_value_returns_zero(self):
        values = np.array([60.0])
        score = landing_stability(values)
        assert score == 0.0

    def test_fraction_parameter_affects_window(self):
        """A larger fraction includes more trailing values for stability."""
        values = np.array([50.0, 52.0, 55.0, 57.0, 60.0, 62.0, 63.0, 63.0, 63.0, 63.0])
        # fraction=0.5 → count = max(3, round(10*0.5)) = max(3, 5) = 5
        # last 5 = [60, 62, 63, 63, 63]
        # spread = 63 - 60 = 3 (approximately, using percentile interpolation)
        # score = 1 - 3/55 ≈ 0.945
        score = landing_stability(values, fraction=0.5)
        # Just verify fraction changes the result vs default (=0.22)
        default_score = landing_stability(values)
        assert score != default_score

    def test_small_spread_partial_stability(self):
        """A narrow spread in the tail produces a score between 0 and 1."""
        values = np.array([40.0, 42.0, 45.0, 48.0, 52.0, 55.0, 57.0, 60.0, 60.0])
        # len=9, count = max(3, round(9*0.22)) = 3
        # last 3 = [57, 60, 60]; 90th-10th ≈ 60-57.6 = 2.4
        # score = 1 - 2.4/55 ≈ 0.956
        score = landing_stability(values)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# normalized_curve_error
# ---------------------------------------------------------------------------


class TestNormalizedCurveError:
    def test_less_than_three_observed_points_returns_none(self):
        times = np.array([0.0, 1.0])
        observed = np.array([0.0, 100.0])
        curve = ((0.0, 0.0), (1.0, 100.0))
        assert normalized_curve_error(times, observed, curve) is None

    def test_less_than_two_authored_points_returns_none(self):
        times = np.array([0.0, 50.0, 100.0])
        observed = np.array([0.0, 50.0, 100.0])
        curve = ((0.0, 0.0),)
        assert normalized_curve_error(times, observed, curve) is None

    def test_exact_match_returns_one(self):
        times = np.array([0.0, 50.0, 100.0])
        observed = np.array([0.0, 50.0, 100.0])
        curve = ((0.0, 0.0), (1.0, 100.0))
        result = normalized_curve_error(times, observed, curve)
        # authored x: [0, 1], y: [0, 100]; normalized x: [0, 1]
        # observed normalized: [0, 0.5, 1.0]
        # interp at [0, 0.5, 1.0] → [0, 50, 100]
        # mae = 0 → score = 1.0
        assert result == 1.0

    def test_large_mismatch_clamps_to_zero(self):
        times = np.array([0.0, 50.0, 100.0])
        observed = np.array([0.0, 500.0, 200.0])
        curve = ((0.0, 0.0), (1.0, 100.0))
        result = normalized_curve_error(times, observed, curve)
        # interp at observed: [0, 50, 100]
        # mae = (0 + 450 + 100) / 3 ≈ 183.33
        # score = max(0, 1 - 183.33/80) = 0
        assert result == 0.0

    def test_zero_span_authored_curve_returns_none(self):
        times = np.array([0.0, 50.0, 100.0])
        observed = np.array([0.0, 50.0, 100.0])
        curve = ((1.0, 0.0), (1.0, 100.0))
        assert normalized_curve_error(times, observed, curve) is None

    def test_zero_observed_span_returns_none(self):
        times = np.array([42.0, 42.0, 42.0])
        observed = np.array([0.0, 50.0, 100.0])
        curve = ((0.0, 0.0), (1.0, 100.0))
        assert normalized_curve_error(times, observed, curve) is None

    def test_perfect_zero_bend(self):
        """Observing 0 cents throughout an authored zero-bend curve."""
        times = np.array([0.0, 50.0, 100.0])
        observed = np.array([0.0, 0.0, 0.0])
        curve = ((0.0, 0.0), (1.0, 0.0))
        assert normalized_curve_error(times, observed, curve) == 1.0

    def test_partial_match(self):
        """A small deviation yields a score between 0 and 1."""
        times = np.array([0.0, 50.0, 100.0])
        observed = np.array([0.0, 30.0, 100.0])
        curve = ((0.0, 0.0), (1.0, 100.0))
        result = normalized_curve_error(times, observed, curve)
        # interp at observed normalized: [0, 0.5, 1.0] → [0, 50, 100]
        # errors: [0, 20, 0] → mae = 20/3 ≈ 6.667
        # score = 1 - 6.667/80 ≈ 0.917
        expected = 1.0 - (20.0 / 3.0) / 80.0
        assert result == pytest.approx(expected, abs=1e-10)

    def test_curve_points_are_sorted_numerically(self):
        """Authored curve x-values are sorted regardless of input order."""
        times = np.array([0.0, 50.0, 100.0])
        observed = np.array([0.0, 50.0, 100.0])
        curve = ((1.0, 100.0), (0.0, 0.0))  # reversed order
        result = normalized_curve_error(times, observed, curve)
        assert result == 1.0


# ---------------------------------------------------------------------------
# _parabolic_peak
# ---------------------------------------------------------------------------


class TestParabolicPeak:
    def test_symmetric_peak_returns_integer_index(self):
        values = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        result = _parabolic_peak(values, 2)
        # left=1, center=2, right=1
        # denominator = 1 - 4 + 1 = -2
        # correction = 0.5 * (1 - 1) / (-2) = 0
        # result = 2.0
        assert result == 2.0

    def test_peak_skewed_left(self):
        values = np.array([1.0, 3.0, 2.0])
        result = _parabolic_peak(values, 1)
        # left=1, center=3, right=2
        # denominator = 1 - 6 + 2 = -3
        # correction = 0.5 * (1 - 2) / (-3) = 0.5 * (-1)/(-3) = 1/6 ≈ 0.1667
        # result = 1 + 0.1667 ≈ 1.1667
        expected = 1.0 + 0.5 * (1.0 - 2.0) / (1.0 - 6.0 + 2.0)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_peak_skewed_right(self):
        values = np.array([1.0, 3.0, 4.0])
        result = _parabolic_peak(values, 1)
        # left=1, center=3, right=4
        # denominator = 1 - 6 + 4 = -1
        # correction = 0.5 * (1 - 4) / (-1) = 0.5 * (-3)/(-1) = 1.5
        # result = 2.5
        assert result == 2.5

    def test_index_zero_returns_float_zero(self):
        values = np.array([5.0, 3.0, 1.0])
        result = _parabolic_peak(values, 0)
        assert result == 0.0
        assert isinstance(result, float)

    def test_index_at_last_position_returns_float_index(self):
        values = np.array([1.0, 3.0, 5.0])
        result = _parabolic_peak(values, 2)
        assert result == 2.0

    def test_flat_denominator_returns_integer_index(self):
        """When left - 2*center + right → 0, fall back to integer index."""
        values = np.array([1.0, 2.0, 3.0])
        result = _parabolic_peak(values, 1)
        # denominator = 1 - 4 + 3 = 0 → falls back due to abs check
        assert result == 1.0

    def test_single_element_returns_zero(self):
        values = np.array([5.0])
        result = _parabolic_peak(values, 0)
        # index 0 == len-1 = 0 → bracket guard returns float(0)
        assert result == 0.0

    def test_two_element_array_edge(self):
        values = np.array([3.0, 1.0])
        result = _parabolic_peak(values, 0)
        assert result == 0.0
        result = _parabolic_peak(values, 1)
        # index 1 == len-1 = 1 → bracket guard returns float(1)
        assert result == 1.0


# ---------------------------------------------------------------------------
# _median_smooth
# ---------------------------------------------------------------------------


class TestMedianSmooth:
    def test_smooths_spike(self):
        values = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
        result = _median_smooth(values, radius=1)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_smooths_spike_radius_2(self):
        values = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        result = _median_smooth(values, radius=2)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_radius_zero_returns_copy(self):
        values = np.array([0.0, 100.0, 0.0])
        result = _median_smooth(values, radius=0)
        np.testing.assert_array_equal(result, values)
        assert result is not values

    def test_less_than_three_values_returns_copy(self):
        values = np.array([5.0, 10.0])
        result = _median_smooth(values, radius=1)
        np.testing.assert_array_equal(result, values)
        assert result is not values

    def test_empty_returns_copy(self):
        values = np.array([], dtype=np.float64)
        result = _median_smooth(values, radius=1)
        assert len(result) == 0
        assert result is not values

    def test_preserves_trend(self):
        """Smoothing should preserve the overall shape."""
        values = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
        result = _median_smooth(values, radius=1)
        # index 0: [10, 12]        → median 11.0
        # index 1: [10, 12, 11]    → median 11.0
        # index 2: [12, 11, 13]    → median 12.0
        # index 3: [11, 13, 12]    → median 12.0
        # index 4: [13, 12]        → median 12.5
        expected = np.array([11.0, 11.0, 12.0, 12.0, 12.5])
        np.testing.assert_array_equal(result, expected)

    def test_negative_radius_returns_copy(self):
        """radius <= 0 triggers the early copy path."""
        values = np.array([1.0, 2.0, 3.0])
        result = _median_smooth(values, radius=-1)
        np.testing.assert_array_equal(result, values)
        assert result is not values

    def test_radius_larger_than_array(self):
        """radius >= len(values) still works — window clamped."""
        values = np.array([1.0, 100.0, 2.0])
        result = _median_smooth(values, radius=5)
        # index 0: values[0:3]  → [1, 100, 2] → median 2.0
        # index 1: values[0:3]  → [1, 100, 2] → median 2.0
        # index 2: values[0:3]  → [1, 100, 2] → median 2.0
        expected = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(result, expected)
