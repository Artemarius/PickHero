"""Spaced repetition scheduler for song/section practice review.

Uses the SM-2 algorithm (same as Anki) to schedule when a song should
be revisited based on accuracy of the last attempt.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta


@dataclass
class ReviewState:
    """SM-2 scheduling state for a single song."""
    ease_factor: float = 2.5
    interval_days: int = 0
    repetitions: int = 0
    next_review: str = ""  # ISO datetime


def accuracy_to_rating(accuracy_percent: float) -> int:
    """Map accuracy (0-100) to SM-2 quality rating (0-5)."""
    if accuracy_percent >= 95:
        return 5  # perfect
    if accuracy_percent >= 85:
        return 4  # good
    if accuracy_percent >= 70:
        return 3  # pass
    if accuracy_percent >= 50:
        return 2  # barely
    if accuracy_percent >= 30:
        return 1  # poor
    return 0  # black


def update_review(state: ReviewState, accuracy_percent: float) -> ReviewState:
    """Update SM-2 state after a practice attempt. Returns new state."""
    q = accuracy_to_rating(accuracy_percent)
    ef = state.ease_factor

    if q >= 3:
        # Correct: advance
        if state.repetitions == 0:
            interval = 1
        elif state.repetitions == 1:
            interval = 6
        else:
            interval = round(state.interval_days * ef)
        reps = state.repetitions + 1
    else:
        # Failed: reset
        interval = 1
        reps = 0

    # Update ease factor
    ef = ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    ef = max(1.3, ef)

    now = datetime.now(timezone.utc)
    due = now + timedelta(days=interval)

    return ReviewState(
        ease_factor=round(ef, 3),
        interval_days=interval,
        repetitions=reps,
        next_review=due.isoformat(),
    )


def is_due(state: ReviewState) -> bool:
    """Check if a song is due for review."""
    if not state.next_review:
        return True  # never practiced = due
    try:
        due = datetime.fromisoformat(state.next_review)
        return datetime.now(timezone.utc) >= due
    except (ValueError, TypeError):
        return True


def days_until_due(state: ReviewState) -> int:
    """Return days until due (negative = overdue)."""
    if not state.next_review:
        return -1
    try:
        due = datetime.fromisoformat(state.next_review)
        delta = due - datetime.now(timezone.utc)
        return max(-999, delta.days)
    except (ValueError, TypeError):
        return -1
