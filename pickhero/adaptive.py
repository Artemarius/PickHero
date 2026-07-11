"""Phrase mastery and dynamic arrangement difficulty.

The controller deliberately changes *content density*, not detector strictness.
Each phrase owns a stable arrangement level with hysteresis so one unusually
clean or poor pass cannot make the chart oscillate between difficulties.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pickhero.tabs.timeline import NoteEvent, Timeline


@dataclass
class PhraseMastery:
    phrase_id: int
    level: int
    max_level: int = 5
    mastery: float = 0.0
    attempts: int = 0
    success_streak: int = 0
    failure_streak: int = 0
    cooldown: int = 0
    last_accuracy: float = 0.0

    @classmethod
    def from_dict(
        cls,
        phrase_id: int,
        payload: dict,
        *,
        default_level: int,
        max_level: int,
    ) -> "PhraseMastery":
        known = {
            "level",
            "max_level",
            "mastery",
            "attempts",
            "success_streak",
            "failure_streak",
            "cooldown",
            "last_accuracy",
        }
        values = {key: value for key, value in payload.items() if key in known}
        values.setdefault("level", default_level)
        values["max_level"] = max_level
        state = cls(phrase_id=phrase_id, **values)
        state.level = max(1, min(max_level, int(state.level)))
        state.mastery = _normalise_accuracy(state.mastery)
        state.last_accuracy = _normalise_accuracy(state.last_accuracy)
        state.attempts = max(0, int(state.attempts))
        state.success_streak = max(0, int(state.success_streak))
        state.failure_streak = max(0, int(state.failure_streak))
        state.cooldown = max(0, int(state.cooldown))
        return state

    def export(self) -> dict:
        payload = asdict(self)
        payload.pop("phrase_id", None)
        return payload


def _normalise_accuracy(value: float) -> float:
    number = float(value)
    if number > 1.0:
        number /= 100.0
    return max(0.0, min(1.0, number))


class AdaptiveDifficultyController:
    """Maintain one stable arrangement level per phrase.

    Promotion requires two strong passes. Demotion requires two materially poor
    passes. A one-pass cooldown after a level change prevents rapid toggling.
    """

    def __init__(
        self,
        timeline: Timeline,
        *,
        enabled: bool = True,
        initial_level: int = 3,
        target_accuracy: float = 0.88,
        persisted: dict[str, dict] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.target_accuracy = _normalise_accuracy(target_accuracy)
        self._timeline = timeline
        self._notes_by_phrase: dict[int, list[NoteEvent]] = {}
        self._phrases_by_measure: dict[int, set[int]] = {}
        for note in timeline.notes:
            phrase_id = max(0, int(note.phrase_id))
            self._notes_by_phrase.setdefault(phrase_id, []).append(note)
            self._phrases_by_measure.setdefault(note.measure, set()).add(phrase_id)

        saved = persisted or {}
        self.phrases: dict[int, PhraseMastery] = {}
        for phrase_id, notes in self._notes_by_phrase.items():
            max_level = max((note.difficulty_level for note in notes), default=1)
            max_level = max(1, min(5, int(max_level)))
            default_level = max(1, min(max_level, int(initial_level)))
            raw = saved.get(str(phrase_id), {})
            if isinstance(raw, dict):
                state = PhraseMastery.from_dict(
                    phrase_id,
                    raw,
                    default_level=default_level,
                    max_level=max_level,
                )
            else:
                state = PhraseMastery(
                    phrase_id=phrase_id,
                    level=default_level,
                    max_level=max_level,
                )
            self.phrases[phrase_id] = state

    def accepts(self, note: NoteEvent) -> bool:
        if not self.enabled:
            return True
        state = self.phrases.get(max(0, int(note.phrase_id)))
        if state is None:
            return True
        return int(note.difficulty_level) <= state.level

    def update_phrase(self, phrase_id: int, accuracy: float) -> bool:
        """Update mastery and return True when the arrangement level changes."""
        state = self.phrases.get(int(phrase_id))
        if state is None:
            return False

        accuracy = _normalise_accuracy(accuracy)
        state.attempts += 1
        state.last_accuracy = accuracy
        if state.attempts == 1:
            state.mastery = accuracy
        else:
            # Recent attempts matter, but a single pass cannot erase history.
            state.mastery = state.mastery * 0.72 + accuracy * 0.28

        if state.cooldown > 0:
            state.cooldown -= 1

        promote_threshold = min(0.98, self.target_accuracy + 0.02)
        demote_threshold = max(0.35, self.target_accuracy - 0.18)
        if accuracy >= promote_threshold:
            state.success_streak += 1
            state.failure_streak = 0
        elif accuracy < demote_threshold:
            state.failure_streak += 1
            state.success_streak = 0
        else:
            # Neutral passes retain a little momentum but cannot trigger a change.
            state.success_streak = max(0, state.success_streak - 1)
            state.failure_streak = max(0, state.failure_streak - 1)

        changed = False
        if state.cooldown == 0 and state.success_streak >= 2 and state.level < state.max_level:
            state.level += 1
            state.success_streak = 0
            state.failure_streak = 0
            state.cooldown = 1
            changed = True
        elif state.cooldown == 0 and state.failure_streak >= 2 and state.level > 1:
            state.level -= 1
            state.success_streak = 0
            state.failure_streak = 0
            state.cooldown = 1
            changed = True
        return changed

    def update_measure_range(
        self,
        start_measure: int,
        end_measure: int,
        accuracy: float,
    ) -> bool:
        phrase_ids: set[int] = set()
        lo, hi = sorted((int(start_measure), int(end_measure)))
        for measure in range(lo, hi + 1):
            phrase_ids.update(self._phrases_by_measure.get(measure, ()))
        changed = False
        for phrase_id in sorted(phrase_ids):
            changed = self.update_phrase(phrase_id, accuracy) or changed
        return changed

    def phrase_accuracy_from_measure_stats(
        self,
        measure_stats: dict[int, dict[str, int | float]],
    ) -> dict[int, float]:
        """Aggregate authoritative per-event quality into phrase accuracy."""
        totals: dict[int, list[float]] = {}
        for measure, stats in measure_stats.items():
            phrase_ids = self._phrases_by_measure.get(int(measure), set())
            if not phrase_ids:
                continue
            quality_events = int(stats.get("quality_events", 0) or 0)
            if quality_events > 0:
                score = float(stats.get("quality_sum", 0.0)) / quality_events
                weight = float(quality_events)
            else:
                hits = int(stats.get("hits", 0) or 0)
                close = int(stats.get("close", 0) or 0)
                misses = int(stats.get("misses", 0) or 0)
                count = hits + close + misses
                if count <= 0:
                    continue
                score = (hits + close * 0.45) / count
                weight = float(count)
            for phrase_id in phrase_ids:
                bucket = totals.setdefault(phrase_id, [0.0, 0.0])
                bucket[0] += score * weight
                bucket[1] += weight
        return {
            phrase_id: weighted / weight
            for phrase_id, (weighted, weight) in totals.items()
            if weight > 0.0
        }

    def export(self) -> dict[str, dict]:
        return {
            str(phrase_id): state.export()
            for phrase_id, state in sorted(self.phrases.items())
        }
