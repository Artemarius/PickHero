"""Tests for pickhero.timing — TimingObservation model and statistics."""

import math

import pytest

from pickhero.timing import (
    EARLY_THRESHOLD_MS,
    HISTOGRAM_BIN_WIDTH_MS,
    HISTOGRAM_NUM_BINS,
    LATE_THRESHOLD_MS,
    MeasureTimingStats,
    PitchVerdict,
    TimingObservation,
    TimingStats,
    TimingVerdict,
    classify_pitch_distance,
    classify_timing_error,
    compute_stats,
)


def _obs(
    error_ms: float,
    verdict: TimingVerdict | None = None,
    measure: int = 0,
    pitch_verdict: PitchVerdict = PitchVerdict.CORRECT,
) -> TimingObservation:
    """Build a TimingObservation with sensible defaults for testing."""
    if verdict is None:
        verdict = classify_timing_error(error_ms)
    return TimingObservation(
        detected_ms=1000.0 + error_ms,
        expected_ms=1000.0,
        timing_error_ms=error_ms,
        verdict=verdict,
        midi_note=64,
        expected_midi=64,
        measure=measure,
        confidence=0.9,
        pitch_verdict=pitch_verdict,
    )


class TestClassifyTimingError:
    def test_on_time(self):
        assert classify_timing_error(0.0) == TimingVerdict.ON_TIME
        assert classify_timing_error(24.9) == TimingVerdict.ON_TIME
        assert classify_timing_error(-24.9) == TimingVerdict.ON_TIME

    def test_early(self):
        assert classify_timing_error(-25.0) == TimingVerdict.EARLY
        assert classify_timing_error(-50.0) == TimingVerdict.EARLY

    def test_late(self):
        assert classify_timing_error(25.0) == TimingVerdict.LATE
        assert classify_timing_error(50.0) == TimingVerdict.LATE

    def test_exact_thresholds(self):
        assert classify_timing_error(EARLY_THRESHOLD_MS) == TimingVerdict.EARLY
        assert classify_timing_error(LATE_THRESHOLD_MS) == TimingVerdict.LATE


class TestClassifyPitchDistance:
    def test_correct(self):
        assert classify_pitch_distance(0) == PitchVerdict.CORRECT

    def test_near(self):
        assert classify_pitch_distance(1) == PitchVerdict.NEAR

    def test_wrong(self):
        assert classify_pitch_distance(2) == PitchVerdict.WRONG
        assert classify_pitch_distance(12) == PitchVerdict.WRONG


class TestComputeStats:
    def test_empty_list(self):
        stats = compute_stats([])
        assert stats.count == 0
        assert stats.mean_error_ms == 0.0
        assert stats.std_dev_ms == 0.0
        assert stats.early_count == 0
        assert stats.late_count == 0
        assert stats.on_time_count == 0
        assert stats.missed_count == 0
        assert stats.extra_count == 0
        assert len(stats.histogram_bins) == HISTOGRAM_NUM_BINS
        assert all(b == 0 for b in stats.histogram_bins)

    def test_all_on_time(self):
        obs = [_obs(0.0), _obs(5.0), _obs(-10.0)]
        stats = compute_stats(obs)
        assert stats.count == 3
        assert stats.on_time_count == 3
        assert stats.mean_error_ms == pytest.approx(-5.0 / 3.0)
        assert stats.early_count == 0
        assert stats.late_count == 0

    def test_all_early(self):
        obs = [_obs(-30.0), _obs(-40.0), _obs(-50.0)]
        stats = compute_stats(obs)
        assert stats.count == 3
        assert stats.early_count == 3
        assert stats.mean_error_ms == pytest.approx(-40.0)
        assert stats.on_time_count == 0

    def test_mixed_verdicts(self):
        obs = [
            _obs(0.0, measure=0),
            _obs(-30.0, measure=0),
            _obs(40.0, measure=1),
            _obs(10.0, measure=1),
        ]
        stats = compute_stats(obs)
        assert stats.count == 4
        assert stats.on_time_count == 2
        assert stats.early_count == 1
        assert stats.late_count == 1

    def test_missed_notes_excluded_from_mean(self):
        obs = [
            _obs(0.0, measure=0),
            TimingObservation(
                detected_ms=float("nan"),
                expected_ms=1000.0,
                timing_error_ms=float("nan"),
                verdict=TimingVerdict.MISSED,
                midi_note=0,
                expected_midi=64,
                measure=0,
                confidence=0.0,
            ),
        ]
        stats = compute_stats(obs)
        assert stats.count == 1
        assert stats.missed_count == 1
        assert stats.mean_error_ms == pytest.approx(0.0)

    def test_extra_notes_excluded_from_mean(self):
        obs = [
            _obs(0.0, measure=0),
            TimingObservation(
                detected_ms=500.0,
                expected_ms=1000.0,
                timing_error_ms=-500.0,
                verdict=TimingVerdict.EXTRA,
                midi_note=64,
                expected_midi=64,
                measure=-1,
                confidence=0.5,
            ),
        ]
        stats = compute_stats(obs)
        assert stats.count == 1
        assert stats.extra_count == 1
        assert stats.mean_error_ms == pytest.approx(0.0)

    def test_std_dev_single_value(self):
        obs = [_obs(10.0)]
        stats = compute_stats(obs)
        assert stats.std_dev_ms == 0.0

    def test_std_dev_multiple_values(self):
        obs = [_obs(-10.0), _obs(10.0)]
        stats = compute_stats(obs)
        assert stats.std_dev_ms == pytest.approx(10.0)

    def test_min_max_error(self):
        obs = [_obs(-50.0), _obs(30.0), _obs(10.0)]
        stats = compute_stats(obs)
        assert stats.min_error_ms == -50.0
        assert stats.max_error_ms == 30.0

    def test_histogram_bin_count(self):
        obs = [_obs(0.0), _obs(-5.0), _obs(10.0)]
        stats = compute_stats(obs)
        assert sum(stats.histogram_bins) == 3
        assert len(stats.histogram_bins) == HISTOGRAM_NUM_BINS

    def test_histogram_out_of_range_clamps(self):
        obs = [_obs(-150.0), _obs(150.0)]
        stats = compute_stats(obs)
        assert stats.histogram_bins[0] == 1  # clamped to first bin
        assert stats.histogram_bins[-1] == 1  # clamped to last bin

    def test_histogram_bin_placement(self):
        # 0ms should go to bin 10 (the [0, 10) bin)
        obs = [_obs(0.0)]
        stats = compute_stats(obs)
        assert stats.histogram_bins[10] == 1
        assert sum(stats.histogram_bins) == 1

    def test_per_measure_bucketing(self):
        obs = [
            _obs(0.0, measure=0),
            _obs(30.0, measure=0),
            _obs(-40.0, measure=1),
            _obs(10.0, measure=1),
            _obs(5.0, measure=2),
        ]
        stats = compute_stats(obs)
        assert len(stats.per_measure) == 3
        assert stats.per_measure[0].count == 2
        assert stats.per_measure[1].count == 2
        assert stats.per_measure[2].count == 1
        assert stats.per_measure[0].late_count == 1
        assert stats.per_measure[1].early_count == 1
        assert stats.per_measure[1].on_time_count == 1

    def test_per_measure_with_missed(self):
        obs = [
            _obs(0.0, measure=0),
            TimingObservation(
                detected_ms=float("nan"),
                expected_ms=1000.0,
                timing_error_ms=float("nan"),
                verdict=TimingVerdict.MISSED,
                midi_note=0,
                expected_midi=64,
                measure=0,
                confidence=0.0,
            ),
        ]
        stats = compute_stats(obs)
        assert stats.per_measure[0].count == 2
        assert stats.per_measure[0].missed_count == 1
        assert stats.per_measure[0].mean_error_ms == pytest.approx(0.0)

    def test_per_measure_skips_negative_measure(self):
        obs = [
            _obs(0.0, measure=0),
            TimingObservation(
                detected_ms=500.0,
                expected_ms=1000.0,
                timing_error_ms=-500.0,
                verdict=TimingVerdict.EXTRA,
                midi_note=64,
                expected_midi=64,
                measure=-1,
                confidence=0.5,
            ),
        ]
        stats = compute_stats(obs)
        assert 0 in stats.per_measure
        assert -1 not in stats.per_measure


class TestTimingTrend:
    """Test timing trend analysis."""

    def test_trend_rushing(self):
        """All-early observations produce 'rushing' trend."""
        obs = [_obs(-30.0, measure=0), _obs(-30.0, measure=1), _obs(-30.0, measure=2)]
        stats = compute_stats(obs)
        assert stats.trend == "rushing"

    def test_trend_fatiguing(self):
        """Errors increasing over measures produce 'fatiguing' trend."""
        obs = [_obs(-20.0, measure=0), _obs(0.0, measure=1), _obs(20.0, measure=2)]
        stats = compute_stats(obs)
        assert stats.trend == "fatiguing"

    def test_trend_stable(self):
        """Consistent on-time observations produce 'stable' trend."""
        obs = [_obs(1.0, measure=0), _obs(0.0, measure=1), _obs(1.0, measure=2)]
        stats = compute_stats(obs)
        assert stats.trend == "stable"
