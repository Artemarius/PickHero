"""Tests for pickhero.scheduler — SM-2 spaced repetition algorithm."""

from datetime import datetime, timezone, timedelta

from pickhero.scheduler import (
    ReviewState,
    accuracy_to_rating,
    update_review,
    is_due,
    days_until_due,
)


class TestAccuracyToRating:
    def test_perfect(self):
        assert accuracy_to_rating(95.0) == 5

    def test_good(self):
        assert accuracy_to_rating(85.0) == 4

    def test_pass(self):
        assert accuracy_to_rating(70.0) == 3

    def test_barely(self):
        assert accuracy_to_rating(50.0) == 2

    def test_poor(self):
        assert accuracy_to_rating(30.0) == 1

    def test_black(self):
        assert accuracy_to_rating(10.0) == 0

    def test_boundary_95(self):
        assert accuracy_to_rating(94.9) == 4

    def test_boundary_85(self):
        assert accuracy_to_rating(84.9) == 3


class TestUpdateReview:
    def test_first_review_correct(self):
        """Accuracy 90% → interval=1, reps=1."""
        state = ReviewState()
        new = update_review(state, 90.0)
        assert new.interval_days == 1
        assert new.repetitions == 1

    def test_second_review_correct(self):
        """Accuracy 90% → interval=6, reps=2."""
        state = ReviewState(interval_days=1, repetitions=1)
        new = update_review(state, 90.0)
        assert new.interval_days == 6
        assert new.repetitions == 2

    def test_third_review_correct(self):
        """Accuracy 90% → interval=round(6*2.5)=15, reps=3."""
        state = ReviewState(ease_factor=2.5, interval_days=6, repetitions=2)
        new = update_review(state, 90.0)
        assert new.interval_days == 15
        assert new.repetitions == 3

    def test_failed_review_resets(self):
        """Accuracy 40% → interval=1, reps=0."""
        state = ReviewState(interval_days=6, repetitions=2)
        new = update_review(state, 40.0)
        assert new.interval_days == 1
        assert new.repetitions == 0

    def test_ease_factor_never_below_1_3(self):
        """Repeated failures should not drop ease factor below 1.3."""
        state = ReviewState(ease_factor=1.3)
        new = update_review(state, 0.0)
        assert new.ease_factor >= 1.3

    def test_next_review_is_future(self):
        """After a correct review, next_review should be in the future."""
        state = ReviewState()
        new = update_review(state, 90.0)
        due = datetime.fromisoformat(new.next_review)
        assert due > datetime.now(timezone.utc)

    def test_next_review_iso_format(self):
        """next_review should be a valid ISO datetime string."""
        state = ReviewState()
        new = update_review(state, 90.0)
        # Should parse without error
        datetime.fromisoformat(new.next_review)


class TestIsDue:
    def test_never_practiced(self):
        """Empty state (no next_review) → due."""
        state = ReviewState()
        assert is_due(state) is True

    def test_future_not_due(self):
        """Due in 5 days → not due."""
        future = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
        state = ReviewState(next_review=future)
        assert is_due(state) is False

    def test_past_is_due(self):
        """Due 2 days ago → due."""
        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        state = ReviewState(next_review=past)
        assert is_due(state) is True

    def test_invalid_format_is_due(self):
        """Invalid datetime string → due (safe fallback)."""
        state = ReviewState(next_review="not-a-date")
        assert is_due(state) is True


class TestDaysUntilDue:
    def test_never_practiced(self):
        """Empty state → -1."""
        state = ReviewState()
        assert days_until_due(state) == -1

    def test_future_days(self):
        """Due in 3 days → ~3 (within 1 day tolerance for time-of-day drift)."""
        future = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
        state = ReviewState(next_review=future)
        result = days_until_due(state)
        assert 2 <= result <= 3

    def test_past_days_negative(self):
        """Due 2 days ago → ~-2 (within 1 day tolerance for time-of-day drift)."""
        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        state = ReviewState(next_review=past)
        result = days_until_due(state)
        assert -3 <= result <= -2

    def test_invalid_format(self):
        """Invalid datetime → -1."""
        state = ReviewState(next_review="not-a-date")
        assert days_until_due(state) == -1


class TestDetectCliff:
    """Test tempo cliff detection."""

    def test_detect_cliff_finds_drop(self):
        """Synthetic tempo_history with a cliff at 120 BPM."""
        from pickhero.progress import SongRecord
        from pickhero.recommendations import detect_cliff

        record = SongRecord()
        # Song at 100 BPM: 80 BPM (90%), 100 BPM (88%), 120 BPM (45%)
        record.tempo_history = [
            {"tempo_factor": 0.8, "accuracy": 90.0},
            {"tempo_factor": 1.0, "accuracy": 88.0},
            {"tempo_factor": 1.2, "accuracy": 45.0},
        ]
        cliff = detect_cliff(record, song_bpm=100)
        assert cliff is not None
        assert cliff == 120.0

    def test_detect_cliff_insufficient_data(self):
        """< 3 entries → None."""
        from pickhero.progress import SongRecord
        from pickhero.recommendations import detect_cliff

        record = SongRecord()
        record.tempo_history = [
            {"tempo_factor": 0.8, "accuracy": 90.0},
            {"tempo_factor": 1.0, "accuracy": 85.0},
        ]
        assert detect_cliff(record, song_bpm=100) is None

    def test_detect_cliff_no_drop(self):
        """Consistent accuracy → None."""
        from pickhero.progress import SongRecord
        from pickhero.recommendations import detect_cliff

        record = SongRecord()
        record.tempo_history = [
            {"tempo_factor": 0.8, "accuracy": 90.0},
            {"tempo_factor": 0.9, "accuracy": 88.0},
            {"tempo_factor": 1.0, "accuracy": 87.0},
        ]
        assert detect_cliff(record, song_bpm=100) is None
