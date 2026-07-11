"""Tests for pickhero.recommendations — all public functions and helpers."""

import pytest
from pickhero.progress import SongRecord
from pickhero.recommendations import (
    generate_recommendations,
    _attempt_milestone,
    _improvement_recommendation,
    _tempo_recommendation,
    detect_cliff,
    _section_recommendation,
    recommend_drill,
    _is_persistent_weakness,
)


# ---------------------------------------------------------------------------
# _attempt_milestone
# ---------------------------------------------------------------------------

class TestAttemptMilestone:
    """Milestone messages for notable attempt counts."""

    def test_first_attempt(self):
        assert _attempt_milestone(1) == (
            "First attempt! Play again to track progress."
        )

    def test_five_attempts(self):
        msg = _attempt_milestone(5)
        assert msg is not None and "muscle memory" in msg

    def test_ten_attempts(self):
        msg = _attempt_milestone(10)
        assert msg is not None and "Attempt #10" in msg

    def test_twentyfive_attempts(self):
        msg = _attempt_milestone(25)
        assert msg is not None and "25 attempts" in msg

    def test_fifty_attempts(self):
        msg = _attempt_milestone(50)
        assert msg is not None and "Attempt #50" in msg

    def test_onehundred_attempts(self):
        msg = _attempt_milestone(100)
        assert msg is not None and "Attempt #100" in msg

    def test_two_attempts_returns_none(self):
        assert _attempt_milestone(2) is None

    def test_three_attempts_returns_none(self):
        assert _attempt_milestone(3) is None

    def test_seven_attempts_returns_none(self):
        assert _attempt_milestone(7) is None

    def test_zero_attempts_returns_none(self):
        assert _attempt_milestone(0) is None


# ---------------------------------------------------------------------------
# _improvement_recommendation
# ---------------------------------------------------------------------------

class TestImprovementRecommendation:
    """Accuracy trend compared to recent history."""

    def test_returns_none_when_history_too_short(self):
        record = SongRecord(tempo_history=[{"attempt": 1, "tempo_factor": 1.0, "accuracy": 80.0}])
        assert _improvement_recommendation(record, 85.0) is None

    def test_returns_none_when_history_empty(self):
        record = SongRecord()
        assert _improvement_recommendation(record, 80.0) is None

    def test_improvement_above_5_percent(self):
        """diff > 5.0 → "Up X% from recent average"."""
        record = SongRecord(tempo_history=[
            {"attempt": 1, "tempo_factor": 1.0, "accuracy": 60.0},
            {"attempt": 2, "tempo_factor": 1.0, "accuracy": 60.0},
        ])
        msg = _improvement_recommendation(record, 80.0)
        assert msg is not None
        assert "Up " in msg
        assert "from recent average" in msg

    def test_small_improvement(self):
        """0 < diff <= 5.0 → "Improved X% since last few attempts"."""
        record = SongRecord(tempo_history=[
            {"attempt": 1, "tempo_factor": 1.0, "accuracy": 70.0},
            {"attempt": 2, "tempo_factor": 1.0, "accuracy": 70.0},
        ])
        msg = _improvement_recommendation(record, 73.0)
        assert msg is not None
        assert "Improved" in msg
        assert "since last few attempts" in msg

    def test_accuracy_dipped(self):
        """diff < -5.0 → "Accuracy dipped"."""
        record = SongRecord(tempo_history=[
            {"attempt": 1, "tempo_factor": 1.0, "accuracy": 80.0},
            {"attempt": 2, "tempo_factor": 1.0, "accuracy": 80.0},
        ])
        msg = _improvement_recommendation(record, 65.0)
        assert msg is not None
        assert "Accuracy dipped" in msg

    def test_plateau_detected(self):
        """spread < 2% across 3+ attempts + current accuracy < 95%."""
        record = SongRecord(tempo_history=[
            {"attempt": 1, "tempo_factor": 1.0, "accuracy": 81.0},
            {"attempt": 2, "tempo_factor": 1.0, "accuracy": 81.5},
            {"attempt": 3, "tempo_factor": 1.0, "accuracy": 81.5},
        ])
        # recent = history[:-1] = [81.0, 81.5], avg = 81.25
        # diff = 81.0 - 81.25 = -0.25, not > 5, not > 0 → plateau check
        # accuracies = [81.0, 81.5, 81.0], spread = 0.5 < 2.0 ✓
        msg = _improvement_recommendation(record, 81.0)
        assert msg is not None
        assert "Plateau detected" in msg

    def test_no_recommendation_when_stable(self):
        """diff between -5 and 0, no plateau → None."""
        record = SongRecord(tempo_history=[
            {"attempt": 1, "tempo_factor": 1.0, "accuracy": 80.0},
            {"attempt": 2, "tempo_factor": 1.0, "accuracy": 80.0},
        ])
        assert _improvement_recommendation(record, 78.0) is None


# ---------------------------------------------------------------------------
# _tempo_recommendation
# ---------------------------------------------------------------------------

class TestTempoRecommendation:
    """Tempo change suggestions based on accuracy."""

    def test_full_speed_well_done(self):
        msg = _tempo_recommendation(SongRecord(), accuracy=92.0, tempo_factor=1.0)
        assert msg is not None
        assert "Full speed" in msg

    def test_full_speed_slow_down(self):
        msg = _tempo_recommendation(SongRecord(), accuracy=55.0, tempo_factor=1.0)
        assert msg is not None
        assert "slowing" in msg

    def test_full_speed_middle_ground_no_recommendation(self):
        assert _tempo_recommendation(SongRecord(), accuracy=75.0, tempo_factor=1.0) is None

    def test_slow_speed_bump_up(self):
        msg = _tempo_recommendation(SongRecord(), accuracy=92.0, tempo_factor=0.8)
        assert msg is not None
        assert "bumping" in msg

    def test_slow_speed_consider_slowing(self):
        msg = _tempo_recommendation(SongRecord(), accuracy=55.0, tempo_factor=0.8)
        assert msg is not None
        assert "slowing" in msg

    def test_slow_speed_middle_ground_no_recommendation(self):
        assert _tempo_recommendation(SongRecord(), accuracy=75.0, tempo_factor=0.8) is None

    def test_minimum_tempo_factor_no_slow_down(self):
        """tempo_factor 0.5 → no slow-down suggestion (< 0.5 check)."""
        msg = _tempo_recommendation(SongRecord(), accuracy=55.0, tempo_factor=0.5)
        # accuracy < 60 but tempo_factor not > 0.5
        assert msg is None


# ---------------------------------------------------------------------------
# detect_cliff
# ---------------------------------------------------------------------------

class TestDetectCliff:
    """Detect BPM where accuracy drops sharply."""

    def test_returns_none_when_fewer_than_three_entries(self):
        record = SongRecord(tempo_history=[
            {"tempo_factor": 0.8, "accuracy": 80.0},
        ])
        assert detect_cliff(record, song_bpm=120) is None

    def test_returns_cliff_bpm_on_sharp_drop(self):
        record = SongRecord(tempo_history=[
            {"tempo_factor": 0.6, "accuracy": 90.0},
            {"tempo_factor": 0.8, "accuracy": 80.0},
            {"tempo_factor": 1.0, "accuracy": 55.0},
        ])
        # After sort by BPM: [(72, 90), (96, 80), (120, 55)]
        # drop at i=2: 80 - 55 = 25 > 20 → cliff = 120.0
        result = detect_cliff(record, song_bpm=120)
        assert result is not None
        assert result == 120.0

    def test_returns_none_on_consistent_accuracy(self):
        record = SongRecord(tempo_history=[
            {"tempo_factor": 0.6, "accuracy": 85.0},
            {"tempo_factor": 0.8, "accuracy": 87.0},
            {"tempo_factor": 1.0, "accuracy": 83.0},
        ])
        assert detect_cliff(record, song_bpm=120) is None


# ---------------------------------------------------------------------------
# _section_recommendation
# ---------------------------------------------------------------------------

class TestSectionRecommendation:
    """Loop suggestions for weak sections."""

    def test_empty_weakest_sections_returns_none(self):
        assert _section_recommendation(SongRecord(), [], 1.0) is None

    def test_weak_section_with_non_persistent(self):
        record = SongRecord(section_history=[
            {"attempt": 1, "sections": [{"measures": [5, 10], "accuracy": 60.0}]},
        ])
        msg = _section_recommendation(record, [(5, 10, 55.0)], 1.0)
        assert msg is not None
        assert "Try bars 6-11 on loop" in msg

    def test_persistent_weak_section_full_speed(self):
        record = SongRecord(section_history=[
            {"attempt": 1, "sections": [{"measures": [5, 10], "accuracy": 55.0}]},
            {"attempt": 2, "sections": [{"measures": [5, 10], "accuracy": 60.0}]},
            {"attempt": 3, "sections": [{"measures": [5, 10], "accuracy": 58.0}]},
        ])
        msg = _section_recommendation(record, [(5, 10, 55.0)], 1.0)
        assert msg is not None
        # tempo_factor=1.0 → no speed hint added
        assert "still weak" in msg

    def test_persistent_weak_section_slow_speed(self):
        """Slow speed hint appears for tempo_factor > 0.5."""
        record = SongRecord(section_history=[
            {"attempt": 1, "sections": [{"measures": [5, 10], "accuracy": 55.0}]},
            {"attempt": 2, "sections": [{"measures": [5, 10], "accuracy": 60.0}]},
            {"attempt": 3, "sections": [{"measures": [5, 10], "accuracy": 58.0}]},
        ])
        msg = _section_recommendation(record, [(5, 10, 55.0)], 0.8)
        assert msg is not None
        assert "at 70%" in msg or "loop at 70%" in msg

    def test_strong_section_returns_none(self):
        record = SongRecord(section_history=[
            {"attempt": 1, "sections": [{"measures": [5, 10], "accuracy": 90.0}]},
        ])
        assert _section_recommendation(record, [(5, 10, 85.0)], 1.0) is None


# ---------------------------------------------------------------------------
# generate_recommendations
# ---------------------------------------------------------------------------

class TestGenerateRecommendations:
    """Top-level recommendation orchestrator."""

    def test_returns_max_3_items(self):
        """Even with many possible recs, output is capped at 3."""
        record = SongRecord(
            attempts=100,
            tempo_history=[
                {"attempt": 97, "tempo_factor": 0.8, "accuracy": 95.0},
                {"attempt": 98, "tempo_factor": 0.8, "accuracy": 95.0},
                {"attempt": 99, "tempo_factor": 0.8, "accuracy": 95.0},
            ],
            section_history=[
                {"attempt": 98, "sections": [{"measures": [5, 10], "accuracy": 55.0}]},
                {"attempt": 99, "sections": [{"measures": [5, 10], "accuracy": 58.0}]},
                {"attempt": 100, "sections": [{"measures": [5, 10], "accuracy": 60.0}]},
            ],
        )
        current_stats = {"accuracy_percent": 95.0}
        weakest_sections = [(5, 10, 55.0)]
        recs = generate_recommendations(
            record, current_stats, weakest_sections, tempo_factor=1.0
        )
        assert len(recs) <= 3

    def test_new_record_milestone_only(self):
        record = SongRecord(attempts=1)
        current_stats = {"accuracy_percent": 75.0}
        recs = generate_recommendations(record, current_stats, [], 1.0)
        assert len(recs) == 1
        assert "First attempt" in recs[0]

    def test_empty_fields_no_recommendations(self):
        """Session with no attempt history, middle accuracy → no recs."""
        record = SongRecord()
        recs = generate_recommendations(record, {"accuracy_percent": 75.0}, [], 1.0)
        assert recs == []


# ---------------------------------------------------------------------------
# recommend_drill
# ---------------------------------------------------------------------------

class TestRecommendDrill:
    """Drill recommendations from technique heatmap."""

    def test_empty_heatmap_returns_none(self):
        assert recommend_drill({}, [(5, 10, 55.0)]) is None

    def test_technique_below_threshold_with_weak_section(self):
        heatmap = {"bend": {"accuracy": 45.0, "count": 10}}
        msg = recommend_drill(heatmap, [(5, 10, 55.0)])
        assert msg is not None
        assert "Loop bars 6-11 at 70%" in msg
        assert "bend" in msg

    def test_technique_above_threshold_returns_none(self):
        heatmap = {"bend": {"accuracy": 75.0, "count": 10}}
        assert recommend_drill(heatmap, [(5, 10, 55.0)]) is None

    def test_no_weak_sections_returns_none(self):
        heatmap = {"bend": {"accuracy": 45.0, "count": 10}}
        assert recommend_drill(heatmap, []) is None

    def test_technique_with_zero_count_ignored(self):
        """A technique with count=0 should not be considered weak."""
        heatmap = {
            "bend": {"accuracy": 45.0, "count": 0},
            "hammer_on": {"accuracy": 70.0, "count": 5},
        }
        # bend has count=0 → skipped. hammer_on has acc 70.0 >= 60.0
        assert recommend_drill(heatmap, [(5, 10, 55.0)]) is None

    def test_technique_with_no_count_field_ignored(self):
        """Missing 'count' key should default to 0."""
        heatmap = {"bend": {"accuracy": 45.0}}
        assert recommend_drill(heatmap, [(5, 10, 55.0)]) is None


# ---------------------------------------------------------------------------
# _is_persistent_weakness
# ---------------------------------------------------------------------------

class TestIsPersistentWeakness:
    """Check if a section has been weak across the last 3 attempts."""

    def test_less_than_three_entries_returns_false(self):
        record = SongRecord(section_history=[
            {"attempt": 1, "sections": [{"measures": [0, 4], "accuracy": 60.0}]},
        ])
        assert _is_persistent_weakness(record, 0, 4) is False

    def test_empty_history_returns_false(self):
        record = SongRecord()
        assert _is_persistent_weakness(record, 0, 4) is False

    def test_all_three_recent_weak(self):
        record = SongRecord(section_history=[
            {"attempt": 1, "sections": [{"measures": [5, 10], "accuracy": 60.0}]},
            {"attempt": 2, "sections": [{"measures": [5, 10], "accuracy": 55.0}]},
            {"attempt": 3, "sections": [{"measures": [5, 10], "accuracy": 65.0}]},
        ])
        assert _is_persistent_weakness(record, 5, 10) is True

    def test_one_of_three_not_weak(self):
        record = SongRecord(section_history=[
            {"attempt": 1, "sections": [{"measures": [5, 10], "accuracy": 60.0}]},
            {"attempt": 2, "sections": [{"measures": [5, 10], "accuracy": 85.0}]},  # not weak
            {"attempt": 3, "sections": [{"measures": [5, 10], "accuracy": 65.0}]},
        ])
        assert _is_persistent_weakness(record, 5, 10) is False

    def test_overlapping_sections_detected(self):
        """Section ranges only need to overlap, not match exactly."""
        record = SongRecord(section_history=[
            {"attempt": 1, "sections": [{"measures": [0, 7], "accuracy": 50.0}]},
            {"attempt": 2, "sections": [{"measures": [0, 7], "accuracy": 55.0}]},
            {"attempt": 3, "sections": [{"measures": [0, 7], "accuracy": 60.0}]},
        ])
        # query section [5, 10] overlaps [0, 7] in all 3 entries
        assert _is_persistent_weakness(record, 5, 10) is True

    def test_no_overlapping_sections(self):
        """Non-overlapping range should not count as a match."""
        record = SongRecord(section_history=[
            {"attempt": 1, "sections": [{"measures": [0, 4], "accuracy": 50.0}]},
            {"attempt": 2, "sections": [{"measures": [0, 4], "accuracy": 55.0}]},
            {"attempt": 3, "sections": [{"measures": [0, 4], "accuracy": 60.0}]},
        ])
        # query section [5, 10] does NOT overlap [0, 4]
        assert _is_persistent_weakness(record, 5, 10) is False

    def test_multiple_sections_per_entry(self):
        """Only the matching section is checked; others are ignored."""
        record = SongRecord(section_history=[
            {
                "attempt": 1,
                "sections": [
                    {"measures": [0, 4], "accuracy": 95.0},
                    {"measures": [5, 10], "accuracy": 55.0},
                ],
            },
            {
                "attempt": 2,
                "sections": [
                    {"measures": [0, 4], "accuracy": 97.0},
                    {"measures": [5, 10], "accuracy": 58.0},
                ],
            },
            {
                "attempt": 3,
                "sections": [
                    {"measures": [0, 4], "accuracy": 92.0},
                    {"measures": [5, 10], "accuracy": 62.0},
                ],
            },
        ])
        assert _is_persistent_weakness(record, 5, 10) is True
