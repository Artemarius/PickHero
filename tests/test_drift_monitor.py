"""Tests for pickhero.audio.drift_monitor — clock drift tracking."""

import pytest

from pickhero.audio.drift_monitor import DriftMonitor


class TestDriftMonitor:
    """DriftMonitor tracks ADC time vs sample-count clock divergence."""

    def test_basic_drift_tracking(self):
        """ADC time and sample count produce expected drift_ppm."""
        monitor = DriftMonitor(sample_rate=48000)

        # First update establishes the baseline
        monitor.update(adc_time=1.0, sample_count=0)
        report = monitor.get_drift_report()
        assert report["session_duration_s"] == 0.0  # first call only sets baseline
        assert report["drift_ppm_current"] == 0.0

        # Second update with a perfect clock: 48000 samples in 1.0 s ADC = 0 ppm
        monitor.update(adc_time=2.0, sample_count=48000)
        report = monitor.get_drift_report()
        assert report["session_duration_s"] == 1.0
        assert report["drift_ppm_current"] == 0.0
        assert report["drift_ppm_min"] == 0.0
        assert report["drift_ppm_max"] == 0.0
        assert report["sample_count"] == 48000

    def test_drift_detected_when_clock_diverges(self):
        """Positive drift when sample clock is faster than ADC time."""
        monitor = DriftMonitor(sample_rate=48000)
        monitor._first_adc_time = 0.0
        monitor._first_sample_count = 0
        monitor._started = True
        monitor._sample_rate = 48000.0

        # Inject a scenario: ADC says 1.0 s elapsed but sample clock says
        # 48480 samples / 48000 = 1.01 s — clock is fast by ~1%
        # drift_ppm = (1.01 - 1.0) / 1.0 * 1e6 = 10000 ppm
        monitor._last_adc_time = 1.0
        monitor._last_sample_count = 48480
        monitor._samples_prealloc[0] = 10000.0
        monitor._count = 1
        monitor._drift_ppm_min = 10000.0
        monitor._drift_ppm_max = 10000.0

        report = monitor.get_drift_report()
        assert report["drift_ppm_current"] == pytest.approx(10000.0, rel=0.01)
        assert report["drift_ppm_min"] == pytest.approx(10000.0, rel=0.01)
        assert report["drift_ppm_max"] == pytest.approx(10000.0, rel=0.01)

    def test_drift_ppm_computation(self):
        """Verify drift_ppm formula using actual update() calls."""
        monitor = DriftMonitor(sample_rate=48000)

        # Baseline at t=0.001 (must be positive to be accepted)
        monitor.update(adc_time=0.001, sample_count=0)
        assert monitor._started is True

        monitor.update(adc_time=1.0, sample_count=48100)
        report = monitor.get_drift_report()
        elapsed_adc = 1.0 - 0.001
        elapsed_samples = 48100 / 48000.0
        expected = (elapsed_samples - elapsed_adc) / elapsed_adc * 1e6
        assert report["drift_ppm_current"] == pytest.approx(expected, rel=0.01)

    def test_reset_clears_state(self):
        """reset() discards all prior observations."""
        monitor = DriftMonitor(sample_rate=48000)

        # Establish some state
        monitor.update(adc_time=1.0, sample_count=0)
        monitor.update(adc_time=2.0, sample_count=48000)
        assert monitor._started is True
        assert monitor._count == 1

        # Reset with new sample rate
        monitor.reset(sample_rate=44100)
        assert monitor._started is False
        assert monitor._count == 0
        assert monitor._sample_rate == 44100.0

        # Report after reset should be all zeros
        report = monitor.get_drift_report()
        assert report["session_duration_s"] == 0.0
        assert report["drift_ppm_current"] == 0.0
        assert report["sample_count"] == 0

    def test_update_ignores_negative_adc_time(self):
        """Non-positive adc_time or zero sample_rate is silently ignored."""
        monitor = DriftMonitor(sample_rate=0)
        monitor.update(adc_time=-1.0, sample_count=0)
        assert monitor._started is False

    def test_get_drift_report_empty_before_start(self):
        """get_drift_report returns zeros before first valid update."""
        monitor = DriftMonitor(sample_rate=48000)
        report = monitor.get_drift_report()
        assert report["session_duration_s"] == 0.0
        assert report["drift_ppm_current"] == 0.0
        assert report["drift_ppm_median"] == 0.0
        assert report["drift_ppm_min"] == 0.0
        assert report["drift_ppm_max"] == 0.0
        assert report["sample_count"] == 0
        assert report["adc_time_s"] == 0.0

    def test_drift_report_keys_present(self):
        """All required keys are returned in the drift report."""
        monitor = DriftMonitor(sample_rate=48000)
        monitor.update(adc_time=0.5, sample_count=0)
        monitor.update(adc_time=2.5, sample_count=96000)
        report = monitor.get_drift_report()
        expected_keys = {
            "session_duration_s",
            "drift_ppm_current",
            "drift_ppm_median",
            "drift_ppm_min",
            "drift_ppm_max",
            "sample_count",
            "adc_time_s",
        }
        assert set(report.keys()) == expected_keys

    def test_multiple_updates_track_min_max(self):
        """Multiple drift readings track the running min and max."""
        monitor = DriftMonitor(sample_rate=48000)

        # Baseline — must use positive adc_time
        monitor.update(adc_time=0.001, sample_count=0)
        # Small positive drift: ~2083 ppm
        monitor.update(adc_time=1.0, sample_count=48100)
        # Negative drift (clock slower later): ~-1389 ppm
        monitor.update(adc_time=3.0, sample_count=143800)

        report = monitor.get_drift_report()
        assert report["drift_ppm_min"] < -1000.0
        assert report["drift_ppm_max"] > 2000.0

    def test_median_computation_odd_count(self):
        """Median over odd number of samples."""
        monitor = DriftMonitor(sample_rate=48000)
        monitor._first_adc_time = 0.0
        monitor._first_sample_count = 0
        monitor._last_adc_time = 4.0
        monitor._last_sample_count = 192000
        monitor._started = True
        monitor._count = 3
        monitor._sample_rate = 48000.0
        monitor._samples_prealloc[:3] = [10.0, 20.0, 30.0]

        report = monitor.get_drift_report()
        assert report["drift_ppm_median"] == 20.0

    def test_median_computation_even_count(self):
        """Median over even number of samples."""
        monitor = DriftMonitor(sample_rate=48000)
        monitor._first_adc_time = 0.0
        monitor._first_sample_count = 0
        monitor._last_adc_time = 4.0
        monitor._last_sample_count = 192000
        monitor._started = True
        monitor._count = 4
        monitor._sample_rate = 48000.0
        monitor._samples_prealloc[:4] = [5.0, 15.0, 25.0, 35.0]

        report = monitor.get_drift_report()
        assert report["drift_ppm_median"] == 20.0  # (15+25)/2
