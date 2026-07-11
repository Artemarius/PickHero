"""Per-song progress tracking.

Stores best accuracy, attempt count, last played date, and detailed
section/tempo history per song. JSON file in the user's home directory
alongside settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from pickhero.config import CONFIG_DIR

PROGRESS_FILE = CONFIG_DIR / "progress.json"


@dataclass
class SongRecord:
    best_accuracy: float = 0.0
    best_hits: int = 0
    best_total: int = 0
    attempts: int = 0
    last_played: str = ""

    # Per-section accuracy history across attempts.
    # Each entry: {"attempt": N, "sections": [{"measures": [start, end], "accuracy": float}, ...]}
    section_history: list[dict] = field(default_factory=list)

    # Per-attempt tempo/accuracy tracking.
    # Each entry: {"attempt": N, "tempo_factor": float, "accuracy": float}
    tempo_history: list[dict] = field(default_factory=list)

    # SM-2 spaced repetition state
    review_ease_factor: float = 2.5
    review_interval_days: int = 0
    review_repetitions: int = 0
    review_next_due: str = ""  # ISO datetime

    # Detected tempo where accuracy drops sharply
    cliff_bpm: float | None = None

    # Rocksmith-style phrase mastery / dynamic difficulty state.
    # Keys are phrase ids serialized as strings; values are PhraseMastery dicts.
    phrase_mastery: dict[str, dict] = field(default_factory=dict)


class ProgressTracker:
    """Loads and saves per-song progress data."""

    def __init__(self) -> None:
        self._data: dict[str, SongRecord] = {}
        self._load()

    def record_result(self, song_key: str, stats: dict) -> bool:
        """Record a completed attempt. Returns True if new personal best."""
        record = self._data.get(song_key, SongRecord())
        record.attempts += 1
        record.last_played = datetime.now(timezone.utc).isoformat()

        accuracy = stats.get("accuracy_percent", 0.0)
        is_new_best = accuracy > record.best_accuracy
        if is_new_best:
            record.best_accuracy = accuracy
            record.best_hits = stats.get("hits", 0)
            record.best_total = stats.get("total", 0)

        self._data[song_key] = record
        self._save()
        return is_new_best

    def record_detailed_result(
        self,
        song_key: str,
        stats: dict,
        weakest_sections: list[tuple[int, int, float]],
        tempo_factor: float,
        song_bpm: int = 0,
    ) -> tuple[bool, list[str]]:
        """Record a completed attempt with section-level detail.

        Calls record_result internally for backward compatibility, then
        stores section and tempo history and generates recommendations.

        Args:
            song_key: Unique identifier for the song/track.
            stats: Dict from matcher.get_statistics().
            weakest_sections: List of (start_measure, end_measure, accuracy_pct).
            tempo_factor: Tempo scaling factor used for this attempt (0.5-1.0).

        Returns:
            Tuple of (is_new_best, recommendations) where recommendations
            is a list of short practice suggestion strings.
        """
        is_new_best = self.record_result(song_key, stats)
        record = self._data[song_key]
        accuracy = stats.get("accuracy_percent", 0.0)

        # Store section history for this attempt
        section_entry = {
            "attempt": record.attempts,
            "sections": [
                {"measures": [s, e], "accuracy": acc}
                for s, e, acc in weakest_sections
            ],
        }
        record.section_history.append(section_entry)

        # Store tempo history for this attempt
        tempo_entry = {
            "attempt": record.attempts,
            "tempo_factor": tempo_factor,
            "accuracy": accuracy,
        }
        record.tempo_history.append(tempo_entry)

        # Cap history length to prevent unbounded growth (keep last 50)
        if len(record.section_history) > 50:
            record.section_history = record.section_history[-50:]
        if len(record.tempo_history) > 50:
            record.tempo_history = record.tempo_history[-50:]

        # Update spaced-repetition review state
        from pickhero.scheduler import ReviewState, update_review
        review_state = ReviewState(
            ease_factor=record.review_ease_factor,
            interval_days=record.review_interval_days,
            repetitions=record.review_repetitions,
            next_review=record.review_next_due,
        )
        new_review = update_review(review_state, accuracy)
        record.review_ease_factor = new_review.ease_factor
        record.review_interval_days = new_review.interval_days
        record.review_repetitions = new_review.repetitions
        record.review_next_due = new_review.next_review

        # Detect tempo cliff if BPM is available
        if song_bpm > 0:
            from pickhero.recommendations import detect_cliff
            record.cliff_bpm = detect_cliff(record, song_bpm)

        self._save()

        # Generate recommendations
        from pickhero.recommendations import generate_recommendations

        recommendations = generate_recommendations(
            record, stats, weakest_sections, tempo_factor,
        )

        return is_new_best, recommendations

    def update_phrase_mastery(self, song_key: str, phrase_mastery: dict[str, dict]) -> None:
        """Persist dynamic-difficulty state independently of full-song results."""
        record = self._data.get(song_key, SongRecord())
        record.phrase_mastery = dict(phrase_mastery)
        self._data[song_key] = record
        self._save()

    def get_best(self, song_key: str) -> SongRecord | None:
        """Get best record for a song, or None if never played."""
        return self._data.get(song_key)

    def _load(self) -> None:
        if not PROGRESS_FILE.exists():
            return
        try:
            with open(PROGRESS_FILE) as f:
                raw = json.load(f)
            for key, val in raw.items():
                # Backward compat: ignore unknown fields, supply defaults
                # for new fields that may be missing from old data.
                known_fields = {f.name for f in SongRecord.__dataclass_fields__.values()}
                filtered = {k: v for k, v in val.items() if k in known_fields}
                self._data[key] = SongRecord(**filtered)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    def _save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(
                {k: asdict(v) for k, v in self._data.items()},
                f, indent=2,
            )
