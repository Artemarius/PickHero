"""Practice recommendation engine.

Analyzes per-song progress history and generates short, actionable
practice suggestions shown on the completion screen.

INTEGRATION NOTES (for scrolling.py):
1. Import: from pickhero.recommendations import generate_recommendations
2. In update() song completion block (~line 285-301):
   - Call progress_tracker.record_detailed_result(...) instead of record_result(...)
   - Example:
       self._is_new_best, self._recommendations = (
           self._progress_tracker.record_detailed_result(
               self._song_key, stats,
               self._matcher.get_weakest_sections(),
               self._tempo_factor,
           )
       )
   - Store returned recommendations in self._recommendations (list[str])
3. In _draw_completion_overlay():
   - After weakest section text (~line 652), render recommendation strings:
       recs = getattr(self, "_recommendations", [])
       rec_y = center_y + 170  # or adjust spacing as needed
       for rec in recs:
           rec_surf = hint_font.render(rec, True, t.hud_text)
           surface.blit(rec_surf, (w // 2 - rec_surf.get_width() // 2, rec_y))
           rec_y += 24
   - Shift the controls hint down accordingly
4. In toggle_play() restart block, reset: self._recommendations = []
5. No config.py changes needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pickhero.progress import SongRecord

# Maximum number of recommendations to return
_MAX_RECOMMENDATIONS = 3

# Thresholds
_WEAK_SECTION_ACCURACY = 70.0   # below this = weak
_TEMPO_UP_ACCURACY = 90.0       # above this at current tempo = suggest faster
_TEMPO_DOWN_ACCURACY = 60.0     # below this = suggest slower
_PLATEAU_THRESHOLD = 2.0        # accuracy change < this across 3 attempts = plateau


def generate_recommendations(
    record: SongRecord,
    current_stats: dict,
    weakest_sections: list[tuple[int, int, float]],
    tempo_factor: float,
) -> list[str]:
    """Generate practice recommendations based on progress history.

    Args:
        record: The SongRecord with full history (already updated with
                this attempt's data).
        current_stats: Dict from matcher.get_statistics() for this attempt.
        weakest_sections: List of (start_measure, end_measure, accuracy_pct)
                          from matcher.get_weakest_sections().
        tempo_factor: Current tempo scaling factor (0.5 to 1.0).

    Returns:
        List of short recommendation strings (max 3 items, ~60 chars each).
    """
    recs: list[str] = []
    accuracy = current_stats.get("accuracy_percent", 0.0)
    attempts = record.attempts

    # 1. Attempt milestones
    milestone = _attempt_milestone(attempts)
    if milestone:
        recs.append(milestone)

    # 2. Improvement tracking (compare to recent attempts)
    improvement = _improvement_recommendation(record, accuracy)
    if improvement:
        recs.append(improvement)

    # 3. Tempo progression
    tempo_rec = _tempo_recommendation(record, accuracy, tempo_factor)
    if tempo_rec:
        recs.append(tempo_rec)

    # 4. Weakest section focus
    section_rec = _section_recommendation(record, weakest_sections, tempo_factor)
    if section_rec:
        recs.append(section_rec)

    # Trim to max and return
    return recs[:_MAX_RECOMMENDATIONS]


def _attempt_milestone(attempts: int) -> str | None:
    """Return a milestone message for notable attempt counts."""
    if attempts == 1:
        return "First attempt! Play again to track progress."
    if attempts == 5:
        return "5 attempts in -- building solid muscle memory."
    if attempts == 10:
        return "Attempt #10 -- you know this song well!"
    if attempts == 25:
        return "25 attempts! Dedication pays off."
    if attempts > 0 and attempts % 50 == 0:
        return f"Attempt #{attempts} -- true mastery takes practice."
    return None


def _improvement_recommendation(
    record: SongRecord, current_accuracy: float,
) -> str | None:
    """Compare current accuracy to recent history."""
    history = record.tempo_history
    if len(history) < 2:
        return None

    # Get accuracy from recent attempts (exclude the current one which is
    # already appended to history by record_detailed_result)
    recent = history[-4:-1] if len(history) > 3 else history[:-1]
    if not recent:
        return None

    recent_avg = sum(h["accuracy"] for h in recent) / len(recent)
    diff = current_accuracy - recent_avg

    if diff > 5.0:
        return f"Up {diff:.0f}% from recent average -- nice work!"
    if diff > 0:
        return f"Improved {diff:.1f}% since last few attempts."

    # Check for plateau (last 3+ attempts within threshold)
    if len(recent) >= 2:
        accuracies = [h["accuracy"] for h in recent] + [current_accuracy]
        spread = max(accuracies) - min(accuracies)
        if spread < _PLATEAU_THRESHOLD and current_accuracy < 95.0:
            return "Plateau detected -- try a different approach."

    if diff < -5.0:
        return "Accuracy dipped -- slow down and refocus."

    return None


def _tempo_recommendation(
    record: SongRecord, accuracy: float, tempo_factor: float,
) -> str | None:
    """Suggest tempo changes based on accuracy at current speed."""
    pct = int(tempo_factor * 100)

    if tempo_factor >= 1.0:
        # Already at full speed
        if accuracy >= _TEMPO_UP_ACCURACY:
            return "Full speed at 90%+ accuracy -- well done!"
        if accuracy < _TEMPO_DOWN_ACCURACY:
            return f"Try slowing to {pct - 10}% to build accuracy."
        return None

    if accuracy >= _TEMPO_UP_ACCURACY:
        # Doing great at reduced tempo -- suggest increase
        new_pct = min(100, pct + 5)
        return f"90%+ at {pct}% speed -- try bumping to {new_pct}%."

    if accuracy < _TEMPO_DOWN_ACCURACY and tempo_factor > 0.5:
        new_pct = max(50, pct - 5)
        return f"Consider slowing to {new_pct}% to nail trouble spots."

    return None


def _section_recommendation(
    record: SongRecord,
    weakest_sections: list[tuple[int, int, float]],
    tempo_factor: float,
) -> str | None:
    """Suggest looping weak sections, especially persistent ones."""
    if not weakest_sections:
        return None

    worst = weakest_sections[0]
    start, end, acc = worst[0], worst[1], worst[2]

    # Check if this section has been consistently weak
    persistent = _is_persistent_weakness(record, start, end)
    pct = int(tempo_factor * 100)

    if persistent and acc < _WEAK_SECTION_ACCURACY:
        speed_hint = f" at {max(50, pct - 10)}%" if tempo_factor > 0.5 else ""
        return f"Bars {start+1}-{end+1} still weak -- loop{speed_hint}."

    if acc < _WEAK_SECTION_ACCURACY:
        return f"Try bars {start+1}-{end+1} on loop ({acc:.0f}% accuracy)."

    return None


def _is_persistent_weakness(
    record: SongRecord, start_measure: int, end_measure: int,
) -> bool:
    """Check if a section has been weak across the last 3 attempts."""
    history = record.section_history
    if len(history) < 3:
        return False

    recent = history[-3:]
    weak_count = 0
    for entry in recent:
        for section in entry.get("sections", []):
            measures = section.get("measures", [])
            if len(measures) == 2:
                s, e = measures[0], measures[1]
                # Overlapping range check
                if s <= end_measure and e >= start_measure:
                    if section.get("accuracy", 100.0) < _WEAK_SECTION_ACCURACY:
                        weak_count += 1
                        break
    return weak_count >= 3
