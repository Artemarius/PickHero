"""Tests for pickhero.progress."""

import json
from unittest.mock import MagicMock, patch

import pytest

from pickhero import progress as progress_mod
from pickhero.progress import ProgressTracker, SongRecord


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_tracker(tmp_path):
    """Create a ProgressTracker backed by a tmp_path PROGRESS_FILE."""
    progress_mod.PROGRESS_FILE = tmp_path / "progress.json"
    return ProgressTracker()


def sample_stats(**overrides) -> dict:
    stats = {
        "accuracy_percent": 85.0,
        "hits": 42,
        "total": 50,
    }
    stats.update(overrides)
    return stats


# ---------------------------------------------------------------------------
# SongRecord defaults
# ---------------------------------------------------------------------------

class TestSongRecordDefaults:
    """Verify all dataclass fields have the correct default values."""

    def test_all_defaults(self):
        r = SongRecord()
        assert r.best_accuracy == 0.0
        assert r.best_hits == 0
        assert r.best_total == 0
        assert r.attempts == 0
        assert r.last_played == ""
        assert r.section_history == []
        assert r.tempo_history == []
        assert r.review_ease_factor == 2.5
        assert r.review_interval_days == 0
        assert r.review_repetitions == 0
        assert r.review_next_due == ""
        assert r.cliff_bpm is None
        assert r.phrase_mastery == {}


# ---------------------------------------------------------------------------
# record_result
# ---------------------------------------------------------------------------

class TestRecordResult:
    """ProgressTracker.record_result behaviour."""

    def test_new_best_returns_true_and_sets_fields(self, tmp_path):
        """First attempt should be a new best and populate fields."""
        tracker = make_tracker(tmp_path)
        result = tracker.record_result("song_1", sample_stats(accuracy_percent=90.0))

        assert result is True
        record = tracker.get_best("song_1")
        assert record is not None
        assert record.attempts == 1
        assert record.last_played != ""                     # set to ISO timestamp
        assert record.best_accuracy == 90.0
        assert record.best_hits == 42
        assert record.best_total == 50

    def test_higher_accuracy_is_new_best(self, tmp_path):
        """Second attempt with higher accuracy returns True and updates best."""
        tracker = make_tracker(tmp_path)
        tracker.record_result("song_1", sample_stats(accuracy_percent=80.0))
        result = tracker.record_result("song_1", sample_stats(accuracy_percent=95.0))

        assert result is True
        record = tracker.get_best("song_1")
        assert record.best_accuracy == 95.0
        assert record.best_hits == 42

    def test_lower_accuracy_does_not_overwrite_best(self, tmp_path):
        """Second attempt with lower accuracy returns False and keeps previous best."""
        tracker = make_tracker(tmp_path)
        tracker.record_result("song_1", sample_stats(accuracy_percent=95.0))
        result = tracker.record_result("song_1", sample_stats(accuracy_percent=80.0))

        assert result is False
        record = tracker.get_best("song_1")
        assert record.attempts == 2                      # attempts still incremented
        assert record.best_accuracy == 95.0              # unchanged
        assert record.best_hits == 42                    # unchanged (set on new-best only)
        assert record.best_total == 50

    def test_same_accuracy_is_not_new_best(self, tmp_path):
        """Attempt with equal accuracy should NOT be treated as new best."""
        tracker = make_tracker(tmp_path)
        tracker.record_result("song_1", sample_stats(accuracy_percent=85.0))
        result = tracker.record_result("song_1", sample_stats(accuracy_percent=85.0))

        assert result is False
        record = tracker.get_best("song_1")
        assert record.best_accuracy == 85.0              # first attempt's value kept

    def test_multiple_songs_tracked_independently(self, tmp_path):
        """Records for different songs should not interfere."""
        tracker = make_tracker(tmp_path)
        tracker.record_result("song_a", sample_stats(accuracy_percent=90.0))
        tracker.record_result("song_b", sample_stats(accuracy_percent=70.0))
        tracker.record_result("song_a", sample_stats(accuracy_percent=80.0))

        a = tracker.get_best("song_a")
        b = tracker.get_best("song_b")
        assert a.attempts == 2
        assert a.best_accuracy == 90.0
        assert b.attempts == 1
        assert b.best_accuracy == 70.0


# ---------------------------------------------------------------------------
# record_detailed_result
# ---------------------------------------------------------------------------

class TestRecordDetailedResult:
    """ProgressTracker.record_detailed_result behaviour."""

    def test_stores_history_and_recommendations(self, tmp_path):
        """Section/tempo history appended, SM-2 updated, recommendations returned."""
        tracker = make_tracker(tmp_path)
        stats = sample_stats(accuracy_percent=88.0)
        weakest = [(1, 4, 65.0), (8, 12, 55.0)]

        mock_review = MagicMock(
            ease_factor=2.6,
            interval_days=3,
            repetitions=1,
            next_review="2026-07-15T00:00:00+00:00",
        )

        with patch("pickhero.scheduler.update_review", return_value=mock_review):
            with patch(
                "pickhero.recommendations.generate_recommendations",
                return_value=["Practice measures 1-4 (65%)", "Try at 85% tempo"],
            ):
                is_new_best, recs = tracker.record_detailed_result(
                    "song_1", stats, weakest, tempo_factor=0.85, song_bpm=120,
                )

        assert is_new_best is True
        assert recs == ["Practice measures 1-4 (65%)", "Try at 85% tempo"]

        record = tracker.get_best("song_1")
        # section_history
        assert len(record.section_history) == 1
        entry = record.section_history[0]
        assert entry["attempt"] == 1
        assert entry["sections"] == [
            {"measures": [1, 4], "accuracy": 65.0},
            {"measures": [8, 12], "accuracy": 55.0},
        ]
        # tempo_history
        assert len(record.tempo_history) == 1
        assert record.tempo_history[0] == {
            "attempt": 1,
            "tempo_factor": 0.85,
            "accuracy": 88.0,
        }
        # SM-2 state
        assert record.review_ease_factor == 2.6
        assert record.review_interval_days == 3
        assert record.review_repetitions == 1
        assert record.review_next_due == "2026-07-15T00:00:00+00:00"

    def test_not_new_best_when_lower(self, tmp_path):
        """record_detailed_result should correctly report non-best attempts."""
        tracker = make_tracker(tmp_path)
        stats_best = sample_stats(accuracy_percent=95.0)
        stats_lower = sample_stats(accuracy_percent=60.0)
        weakest = [(2, 3, 60.0)]

        with patch("pickhero.scheduler.update_review") as mu:
            mu.return_value = MagicMock(
                ease_factor=2.5, interval_days=0, repetitions=0, next_review="",
            )
            with patch("pickhero.recommendations.generate_recommendations") as mg:
                mg.return_value = []
                tracker.record_detailed_result(
                    "song_1", stats_best, [], tempo_factor=1.0, song_bpm=0,
                )
                is_new_best, _ = tracker.record_detailed_result(
                    "song_1", stats_lower, weakest, tempo_factor=0.8, song_bpm=0,
                )

        assert is_new_best is False
        record = tracker.get_best("song_1")
        assert record.best_accuracy == 95.0

    def test_caps_history_at_50(self, tmp_path):
        """After 55 attempts, only the most recent 50 entries are kept."""
        tracker = make_tracker(tmp_path)
        stats = sample_stats(accuracy_percent=70.0)
        weakest = [(1, 2, 70.0)]

        with patch("pickhero.scheduler.update_review") as mu:
            mu.return_value = MagicMock(
                ease_factor=2.5, interval_days=0, repetitions=0, next_review="",
            )
            with patch("pickhero.recommendations.generate_recommendations") as mg:
                mg.return_value = []
                for _ in range(55):
                    tracker.record_detailed_result(
                        "song_1", stats, weakest, tempo_factor=1.0, song_bpm=0,
                    )

        record = tracker.get_best("song_1")
        assert len(record.section_history) == 50
        assert len(record.tempo_history) == 50
        # The first 5 entries (attempts 1-5) were dropped, so the earliest is attempt 6
        assert record.section_history[0]["attempt"] == 6
        assert record.tempo_history[0]["attempt"] == 6
        # The latest is attempt 55
        assert record.section_history[-1]["attempt"] == 55
        assert record.tempo_history[-1]["attempt"] == 55

    def test_song_bpm_zero_does_not_call_detect_cliff(self, tmp_path):
        """When song_bpm is 0, detect_cliff should NOT be called."""
        tracker = make_tracker(tmp_path)
        with patch("pickhero.scheduler.update_review") as mu:
            mu.return_value = MagicMock(
                ease_factor=2.5, interval_days=0, repetitions=0, next_review="",
            )
            with patch("pickhero.recommendations.generate_recommendations") as mg:
                mg.return_value = []
                with patch("pickhero.recommendations.detect_cliff") as md:
                    tracker.record_detailed_result(
                        "song_1", sample_stats(), [], tempo_factor=1.0, song_bpm=0,
                    )
                    md.assert_not_called()

    def test_song_bpm_positive_calls_detect_cliff(self, tmp_path):
        """When song_bpm > 0, detect_cliff should be called."""
        tracker = make_tracker(tmp_path)
        with patch("pickhero.scheduler.update_review") as mu:
            mu.return_value = MagicMock(
                ease_factor=2.5, interval_days=0, repetitions=0, next_review="",
            )
            with patch("pickhero.recommendations.generate_recommendations") as mg:
                mg.return_value = []
                with patch("pickhero.recommendations.detect_cliff", return_value=130.0) as md:
                    tracker.record_detailed_result(
                        "song_1", sample_stats(), [], tempo_factor=1.0, song_bpm=120,
                    )
                    md.assert_called_once()

        record = tracker.get_best("song_1")
        assert record.cliff_bpm == 130.0


# ---------------------------------------------------------------------------
# update_phrase_mastery
# ---------------------------------------------------------------------------

class TestUpdatePhraseMastery:
    """ProgressTracker.update_phrase_mastery."""

    def test_stores_phrase_mastery(self, tmp_path):
        tracker = make_tracker(tmp_path)
        mastery = {
            "0": {"accuracy": 0.85, "level": 3},
            "1": {"accuracy": 0.60, "level": 1},
        }
        tracker.update_phrase_mastery("song_1", mastery)

        record = tracker.get_best("song_1")
        assert record.phrase_mastery == mastery

    def test_overwrites_previous_mastery(self, tmp_path):
        tracker = make_tracker(tmp_path)
        tracker.update_phrase_mastery("song_1", {"0": {"accuracy": 0.9, "level": 4}})
        tracker.update_phrase_mastery("song_1", {"0": {"accuracy": 0.5, "level": 2}})

        record = tracker.get_best("song_1")
        assert record.phrase_mastery["0"]["accuracy"] == 0.5
        assert record.phrase_mastery["0"]["level"] == 2

    def test_does_not_affect_other_songs(self, tmp_path):
        tracker = make_tracker(tmp_path)
        tracker.update_phrase_mastery("song_1", {"0": {"accuracy": 0.9}})
        assert tracker.get_best("song_2") is None

    def test_persists_mastery_across_trackers(self, tmp_path):
        tracker1 = make_tracker(tmp_path)
        mastery = {"0": {"accuracy": 0.75, "level": 2}}
        tracker1.update_phrase_mastery("song_1", mastery)

        tracker2 = make_tracker(tmp_path)
        record = tracker2.get_best("song_1")
        assert record.phrase_mastery == mastery


# ---------------------------------------------------------------------------
# get_best
# ---------------------------------------------------------------------------

class TestGetBest:
    """ProgressTracker.get_best."""

    def test_unknown_song_returns_none(self, tmp_path):
        tracker = make_tracker(tmp_path)
        assert tracker.get_best("never_played") is None

    def test_known_song_returns_record(self, tmp_path):
        tracker = make_tracker(tmp_path)
        tracker.record_result("song_1", sample_stats())

        record = tracker.get_best("song_1")
        assert isinstance(record, SongRecord)
        assert record.best_accuracy == 85.0
        assert record.attempts == 1

    def test_returns_same_object_as_internal_data(self, tmp_path):
        tracker = make_tracker(tmp_path)
        tracker.record_result("song_1", sample_stats())
        assert tracker.get_best("song_1") is tracker._data["song_1"]


# ---------------------------------------------------------------------------
# _load
# ---------------------------------------------------------------------------

class TestLoad:
    """ProgressTracker._load error handling and backward compat."""

    def test_missing_file(self, tmp_path):
        """Missing progress file should silently produce empty data."""
        progress_mod.PROGRESS_FILE = tmp_path / "progress.json"
        assert not progress_mod.PROGRESS_FILE.exists()

        tracker = ProgressTracker()
        assert tracker._data == {}

    def test_corrupt_json(self, tmp_path):
        """Corrupt JSON should silently produce empty data."""
        p = tmp_path / "progress.json"
        p.write_text("not valid json {{{")
        progress_mod.PROGRESS_FILE = p

        tracker = ProgressTracker()
        assert tracker._data == {}

    def test_empty_json(self, tmp_path):
        """Empty object {} should produce empty data."""
        p = tmp_path / "progress.json"
        p.write_text("{}")
        progress_mod.PROGRESS_FILE = p

        tracker = ProgressTracker()
        assert tracker._data == {}

    def test_old_format_missing_new_fields(self, tmp_path):
        """Old progress files missing newer fields should get defaults."""
        p = tmp_path / "progress.json"
        old_data = {
            "song_1": {
                "best_accuracy": 80.0,
                "best_hits": 20,
                "best_total": 25,
                "attempts": 3,
                "last_played": "2025-01-01T00:00:00",
                # intentionally missing: section_history, tempo_history,
                # review_ease_factor, review_interval_days, review_repetitions,
                # review_next_due, cliff_bpm, phrase_mastery
            }
        }
        p.write_text(json.dumps(old_data))
        progress_mod.PROGRESS_FILE = p

        tracker = ProgressTracker()
        record = tracker.get_best("song_1")
        assert record is not None
        assert record.best_accuracy == 80.0
        assert record.attempts == 3
        assert record.last_played == "2025-01-01T00:00:00"
        # new fields get defaults
        assert record.section_history == []
        assert record.tempo_history == []
        assert record.review_ease_factor == 2.5
        assert record.review_interval_days == 0
        assert record.review_repetitions == 0
        assert record.review_next_due == ""
        assert record.cliff_bpm is None
        assert record.phrase_mastery == {}

    def test_unknown_fields_are_ignored(self, tmp_path):
        """Extra keys in stored data should be silently ignored."""
        p = tmp_path / "progress.json"
        data = {
            "song_1": {
                "best_accuracy": 90.0,
                "best_hits": 30,
                "best_total": 30,
                "attempts": 1,
                "last_played": "2025-06-01T00:00:00",
                "unknown_field": "should_be_ignored",
                "another_unknown": 42,
            }
        }
        p.write_text(json.dumps(data))
        progress_mod.PROGRESS_FILE = p

        tracker = ProgressTracker()
        record = tracker.get_best("song_1")
        assert record.best_accuracy == 90.0
        # Verify unknown_field never made it as an attribute
        with pytest.raises(AttributeError):
            _ = record.unknown_field


# ---------------------------------------------------------------------------
# _save
# ---------------------------------------------------------------------------

class TestSave:
    """ProgressTracker._save persistence."""

    def test_writes_valid_json(self, tmp_path):
        """_save should write valid JSON to PROGRESS_FILE (parent must exist)."""
        target = tmp_path / "progress.json"
        progress_mod.PROGRESS_FILE = target

        tracker = ProgressTracker()
        tracker.record_result("song_1", sample_stats())

        assert target.exists()
        data = json.loads(target.read_text())
        assert "song_1" in data
        assert data["song_1"]["best_accuracy"] == 85.0
        assert data["song_1"]["attempts"] == 1
        assert data["song_1"]["last_played"] != ""

    def test_round_trip(self, tmp_path):
        """Data saved by one tracker loads identically in another."""
        tracker1 = make_tracker(tmp_path)
        tracker1.record_result("song_1", sample_stats(accuracy_percent=92.0))

        tracker2 = make_tracker(tmp_path)
        record = tracker2.get_best("song_1")
        assert record is not None
        assert record.best_accuracy == 92.0
        assert record.attempts == 1

    def test_round_trip_detailed(self, tmp_path):
        """Detailed results survive a save/load cycle."""
        tracker1 = make_tracker(tmp_path)
        with patch("pickhero.scheduler.update_review") as mu:
            mu.return_value = MagicMock(
                ease_factor=2.6, interval_days=3, repetitions=1,
                next_review="2026-07-15T00:00:00+00:00",
            )
            with patch("pickhero.recommendations.generate_recommendations") as mg:
                mg.return_value = []
                tracker1.record_detailed_result(
                    "song_1", sample_stats(accuracy_percent=88.0),
                    [(1, 4, 65.0)], tempo_factor=0.85, song_bpm=0,
                )

        tracker2 = make_tracker(tmp_path)
        record = tracker2.get_best("song_1")
        assert record.best_accuracy == 88.0
        assert len(record.section_history) == 1
        assert len(record.tempo_history) == 1
        assert record.review_ease_factor == 2.6

    def test_creates_config_dir(self, tmp_path):
        """_save should ensure CONFIG_DIR exists (mkdir side-effect is not testable
        on PosixPath directly; instead verify the directory exists after saving)."""
        from pickhero.config import CONFIG_DIR
        target = tmp_path / "progress.json"
        progress_mod.PROGRESS_FILE = target

        tracker = ProgressTracker()
        tracker.record_result("song_1", sample_stats())

        assert CONFIG_DIR.exists()
