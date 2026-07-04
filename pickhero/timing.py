"""Timing observation model and statistics for the Timing Judge.

Pure logic — no pygame, no audio dependencies. Computes per-onset timing
errors, aggregates them into statistics (mean, std dev, histogram), and
buckets results per measure for worst-bar analysis.

Conventions:
  - timing_error_ms = detected_ms - expected_ms (negative = early, positive = late)
  - All timestamps in milliseconds (float), consistent with the rest of the app.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum


# Timing verdict thresholds (ms). Inside ±EARLY/LATE = ON_TIME.
EARLY_THRESHOLD_MS = -25.0
LATE_THRESHOLD_MS = 25.0

# Histogram: 20 bins covering ±100ms in 10ms increments.
HISTOGRAM_NUM_BINS = 20
HISTOGRAM_RANGE_MS = 200.0  # total range: -100 to +100
HISTOGRAM_BIN_WIDTH_MS = HISTOGRAM_RANGE_MS / HISTOGRAM_NUM_BINS  # 10.0


class TimingVerdict(Enum):
    """Classification of a single timing observation."""
    EARLY = "early"
    ON_TIME = "on_time"
    LATE = "late"
    MISSED = "missed"      # no onset detected within window
    EXTRA = "extra"        # onset with no matching pending note


class PitchVerdict(Enum):
    """Pitch accuracy dimension, separate from timing."""
    CORRECT = "correct"    # exact semitone match
    NEAR = "near"          # ±1 semitone
    WRONG = "wrong"        # >1 semitone off
    UNKNOWN = "unknown"   # pitch couldn't be determined (low confidence onset)


@dataclass(frozen=True)
class TimingObservation:
    """A single timing observation from matching a detected onset against the tab.

    Attributes:
        detected_ms: Wall-clock onset time, ms from session start.
        expected_ms: Tab note timestamp, ms from song start.
        timing_error_ms: detected_ms - expected_ms (negative = early).
        verdict: Timing classification (EARLY/ON_TIME/LATE/MISSED/EXTRA).
        midi_note: Detected MIDI note (0 if unknown).
        expected_midi: Expected tab MIDI note.
        measure: Measure index of expected note (0-based). -1 if no match.
        confidence: Detector confidence (0.0-1.0).
        pitch_verdict: Pitch accuracy classification.
    """
    detected_ms: float
    expected_ms: float
    timing_error_ms: float
    verdict: TimingVerdict
    midi_note: int
    expected_midi: int
    measure: int
    confidence: float
    pitch_verdict: PitchVerdict = PitchVerdict.UNKNOWN
    articulation: str | None = None  # detected articulation (hammer_on, pull_off, etc.)


@dataclass
class MeasureTimingStats:
    """Per-measure timing statistics."""
    count: int = 0
    mean_error_ms: float = 0.0
    std_dev_ms: float = 0.0
    early_count: int = 0
    late_count: int = 0
    on_time_count: int = 0
    missed_count: int = 0


@dataclass
class TimingStats:
    """Aggregated timing statistics across all observations in a run."""
    count: int = 0
    mean_error_ms: float = 0.0
    std_dev_ms: float = 0.0
    early_count: int = 0
    late_count: int = 0
    on_time_count: int = 0
    missed_count: int = 0
    extra_count: int = 0
    min_error_ms: float = 0.0
    max_error_ms: float = 0.0
    per_measure: dict[int, MeasureTimingStats] = field(default_factory=dict)
    histogram_bins: list[int] = field(default_factory=list)
    histogram_range_ms: float = HISTOGRAM_RANGE_MS
    timing_slope_ms_per_measure: float = 0.0
    trend: str = "stable"  # "rushing", "dragging", "fatiguing", "improving", "stable"


def classify_timing_error(error_ms: float) -> TimingVerdict:
    """Classify a timing error into a verdict.

    NaN errors (missed notes) are not classified here — use MISSED directly.
    """
    if error_ms <= EARLY_THRESHOLD_MS:
        return TimingVerdict.EARLY
    if error_ms >= LATE_THRESHOLD_MS:
        return TimingVerdict.LATE
    return TimingVerdict.ON_TIME


def classify_pitch_distance(semitone_distance: int) -> PitchVerdict:
    """Classify pitch accuracy from semitone distance."""
    if semitone_distance == 0:
        return PitchVerdict.CORRECT
    if semitone_distance == 1:
        return PitchVerdict.NEAR
    return PitchVerdict.WRONG


def _histogram_bin(error_ms: float) -> int:
    """Map a timing error to a histogram bin index (0..HISTOGRAM_NUM_BINS-1).

    Bin 0 = [-100, -90), ..., bin 9 = [-10, 0), bin 10 = [0, 10), ..., bin 19 = [90, 100].
    Out-of-range errors clamp to edge bins.
    """
    half_range = HISTOGRAM_RANGE_MS / 2.0  # 100.0
    clamped = max(-half_range, min(half_range - 0.001, error_ms))
    return int((clamped + half_range) / HISTOGRAM_BIN_WIDTH_MS)


def compute_stats(observations: list[TimingObservation]) -> TimingStats:
    """Compute aggregate timing statistics from a list of observations.

    Skips NaN timing_error_ms (missed notes) in mean/std/min/max calculations,
    but counts them in missed_count. EXTRA observations are counted but excluded
    from mean/std/min/max (they have no expected-time reference).
    """
    if not observations:
        return TimingStats(
            histogram_bins=[0] * HISTOGRAM_NUM_BINS,
        )

    # Collect timing errors for mean/std (skip NaN and EXTRA)
    errors = [
        o.timing_error_ms
        for o in observations
        if o.verdict not in (TimingVerdict.MISSED, TimingVerdict.EXTRA)
        and not math.isnan(o.timing_error_ms)
    ]

    stats = TimingStats(
        count=len(errors),
        histogram_bins=[0] * HISTOGRAM_NUM_BINS,
    )

    # Counts by verdict
    for o in observations:
        if o.verdict == TimingVerdict.EARLY:
            stats.early_count += 1
        elif o.verdict == TimingVerdict.LATE:
            stats.late_count += 1
        elif o.verdict == TimingVerdict.ON_TIME:
            stats.on_time_count += 1
        elif o.verdict == TimingVerdict.MISSED:
            stats.missed_count += 1
        elif o.verdict == TimingVerdict.EXTRA:
            stats.extra_count += 1

    # Mean, std dev, min, max from valid errors only
    if errors:
        stats.mean_error_ms = statistics.fmean(errors)
        stats.std_dev_ms = statistics.pstdev(errors) if len(errors) > 1 else 0.0
        stats.min_error_ms = min(errors)
        stats.max_error_ms = max(errors)

        # Histogram
        for err in errors:
            stats.histogram_bins[_histogram_bin(err)] += 1

    # Per-measure bucketing (skip observations without a valid measure)
    measure_errors: dict[int, list[float]] = {}
    measure_verdicts: dict[int, list[TimingVerdict]] = {}
    for o in observations:
        if o.measure < 0:
            continue
        if o.verdict in (TimingVerdict.MISSED, TimingVerdict.EXTRA):
            measure_verdicts.setdefault(o.measure, []).append(o.verdict)
        elif not math.isnan(o.timing_error_ms):
            measure_errors.setdefault(o.measure, []).append(o.timing_error_ms)
            measure_verdicts.setdefault(o.measure, []).append(o.verdict)

    for measure_idx, errs in measure_errors.items():
        verdicts = measure_verdicts.get(measure_idx, [])
        ms = MeasureTimingStats(
            count=len(errs) + sum(1 for v in verdicts if v == TimingVerdict.MISSED),
            mean_error_ms=statistics.fmean(errs) if errs else 0.0,
            std_dev_ms=statistics.pstdev(errs) if len(errs) > 1 else 0.0,
        )
        for v in verdicts:
            if v == TimingVerdict.EARLY:
                ms.early_count += 1
            elif v == TimingVerdict.LATE:
                ms.late_count += 1
            elif v == TimingVerdict.ON_TIME:
                ms.on_time_count += 1
            elif v == TimingVerdict.MISSED:
                ms.missed_count += 1
        stats.per_measure[measure_idx] = ms

    # Timing trend: linear regression of error vs measure index
    trend_points = [
        (o.measure, o.timing_error_ms)
        for o in observations
        if o.verdict not in (TimingVerdict.MISSED, TimingVerdict.EXTRA)
        and not math.isnan(o.timing_error_ms)
        and o.measure >= 0
    ]
    if len(trend_points) >= 3:
        n = len(trend_points)
        sum_x = sum(x for x, _ in trend_points)
        sum_y = sum(y for _, y in trend_points)
        sum_xy = sum(x * y for x, y in trend_points)
        sum_x2 = sum(x * x for x, _ in trend_points)
        denom = n * sum_x2 - sum_x * sum_x
        if denom != 0:
            slope = (n * sum_xy - sum_x * sum_y) / denom
            stats.timing_slope_ms_per_measure = slope
            if slope < -0.5:
                stats.trend = "improving"
            elif slope > 0.5:
                stats.trend = "fatiguing"
            elif stats.mean_error_ms < -10:
                stats.trend = "rushing"
            elif stats.mean_error_ms > 10:
                stats.trend = "dragging"
            else:
                stats.trend = "stable"

    return stats
