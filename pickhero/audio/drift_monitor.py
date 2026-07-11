"""Clock drift monitor for long audio sessions.

Tracks the divergence between host-API ADC timestamps and the sample-count
clock, reporting drift in parts per million (ppm).  The ``update()`` method
is designed for the audio callback hot path: it performs no heap allocations
and no Python function calls beyond built-in float/int ops.
"""

import numpy as np


class DriftMonitor:
    """Track ADC vs sample-count clock drift over a session.

    Usage::

        monitor = DriftMonitor(sample_rate=48000)

        # In the PortAudio callback:
        monitor.update(time_info.inputBufferAdcTime, ring_write_sample)

        # On demand (outside callback):
        report = monitor.get_drift_report()

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.  Should match the stream sample rate.
        May be updated via :meth:`reset`.
    max_samples : int
        Maximum number of drift readings retained for median computation.
    """

    __slots__ = (
        "_first_adc_time",
        "_first_sample_count",
        "_last_adc_time",
        "_last_sample_count",
        "_sample_rate",
        "_drift_ppm_min",
        "_drift_ppm_max",
        "_started",
        "_count",
        "_max_samples",
        "_samples_prealloc",
    )

    def __init__(self, sample_rate: int = 0, max_samples: int = 10000):
        self._first_adc_time: float = 0.0
        self._first_sample_count: int = 0
        self._last_adc_time: float = 0.0
        self._last_sample_count: int = 0
        self._sample_rate: float = float(sample_rate)
        self._drift_ppm_min: float = 0.0
        self._drift_ppm_max: float = 0.0
        self._started: bool = False
        self._count: int = 0
        self._max_samples: int = max_samples
        # Preallocated so update() never allocates.
        self._samples_prealloc: np.ndarray = np.zeros(max_samples, dtype=np.float64)

    def update(self, adc_time: float, sample_count: int) -> None:
        """Record a drift observation.

        Must be called from the PortAudio callback (or any context where
        ``adc_time`` is the host-API ADC timestamp of the current buffer).

        .. note::

           No heap allocations.  The first observation with a positive
           ``adc_time`` initialises the session baseline; subsequent calls
           compute ``drift_ppm`` and store it in the preallocated array.
        """
        if adc_time <= 0.0 or self._sample_rate <= 0.0:
            return

        if not self._started:
            # First valid ADC timestamp — establish session baseline.
            self._first_adc_time = adc_time
            self._first_sample_count = sample_count
            self._drift_ppm_min = 0.0
            self._drift_ppm_max = 0.0
            self._count = 0
            self._started = True
            self._last_adc_time = adc_time
            self._last_sample_count = sample_count
            return

        self._last_adc_time = adc_time
        self._last_sample_count = sample_count

        elapsed_adc = adc_time - self._first_adc_time
        if elapsed_adc <= 0.0:
            return

        elapsed_samples = (sample_count - self._first_sample_count) / self._sample_rate
        drift_ppm = (elapsed_samples - elapsed_adc) / elapsed_adc * 1e6

        # Update running min/max — simple float comparisons, no allocation.
        if self._count == 0:
            self._drift_ppm_min = drift_ppm
            self._drift_ppm_max = drift_ppm
        else:
            if drift_ppm < self._drift_ppm_min:
                self._drift_ppm_min = drift_ppm
            if drift_ppm > self._drift_ppm_max:
                self._drift_ppm_max = drift_ppm

        # Store in preallocated circular-free buffer.
        if self._count < self._max_samples:
            self._samples_prealloc[self._count] = drift_ppm
            self._count += 1

    def reset(self, sample_rate: int) -> None:
        """Reinitialise for a new session (e.g. on stream restart).

        Discards all prior observations.
        """
        self._started = False
        self._count = 0
        self._sample_rate = float(sample_rate)
        # Preallocated array is not zeroed — stale entries past _count are
        # never read.

    def get_drift_report(self) -> dict[str, float | int]:
        """Return a summary of clock drift for the current session.

        Returns
        -------
        dict
            Keys:
            ``session_duration_s``
                Elapsed wall-clock time in seconds (from ADC timestamps).
            ``drift_ppm_current``
                Most recent drift measurement.
            ``drift_ppm_median``
                Median drift over the session.
            ``drift_ppm_min`` / ``drift_ppm_max``
                Min and max drift observed.
            ``sample_count``
                Total samples captured since session start.
            ``adc_time_s``
                Elapsed ADC time in seconds (same as ``session_duration_s``).
        """
        if not self._started or self._count == 0:
            return {
                "session_duration_s": 0.0,
                "drift_ppm_current": 0.0,
                "drift_ppm_median": 0.0,
                "drift_ppm_min": 0.0,
                "drift_ppm_max": 0.0,
                "sample_count": 0,
                "adc_time_s": 0.0,
            }

        # Median over stored samples.  This *does* allocate a sorted copy but
        # is only called on-demand (outside the callback hot path).
        sorted_samples = np.sort(self._samples_prealloc[: self._count])
        n = self._count
        if n % 2 == 1:
            drift_median = float(sorted_samples[n // 2])
        else:
            drift_median = float(
                (sorted_samples[n // 2 - 1] + sorted_samples[n // 2]) / 2.0
            )

        current = float(self._samples_prealloc[self._count - 1])
        elapsed_adc = self._last_adc_time - self._first_adc_time

        return {
            "session_duration_s": round(elapsed_adc, 3),
            "drift_ppm_current": round(current, 3),
            "drift_ppm_median": round(drift_median, 3),
            "drift_ppm_min": round(self._drift_ppm_min, 3),
            "drift_ppm_max": round(self._drift_ppm_max, 3),
            "sample_count": self._last_sample_count - self._first_sample_count,
            "adc_time_s": round(elapsed_adc, 3),
        }
