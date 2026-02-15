"""Per-song progress tracking.

Stores best accuracy, attempt count, and last played date per song.
JSON file in the user's home directory alongside settings.
"""

import json
from dataclasses import dataclass, asdict
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


class ProgressTracker:
    """Loads and saves per-song progress data."""

    def __init__(self):
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
                self._data[key] = SongRecord(**val)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    def _save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(
                {k: asdict(v) for k, v in self._data.items()},
                f, indent=2,
            )
