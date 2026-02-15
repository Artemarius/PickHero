"""Visual feedback for note matching.

Manages color effects, streak counter, and accuracy stats overlay.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from pickhero.matcher import MatchResult, MatchType
from pickhero.tabs.timeline import NoteEvent
from pickhero.ui.colors import (
    FEEDBACK_HIT,
    FEEDBACK_CLOSE,
    FEEDBACK_MISS,
    FEEDBACK_STREAK,
    HUD_TEXT_COLOR,
    dimmed,
)

EFFECT_DURATION_MS = 500.0

_MATCH_COLORS = {
    MatchType.HIT: FEEDBACK_HIT,
    MatchType.CLOSE: FEEDBACK_CLOSE,
    MatchType.MISS: FEEDBACK_MISS,
}


@dataclass
class _FeedbackEffect:
    """A timed visual effect on a note."""
    note_key: tuple[float, int]  # (timestamp_ms, string)
    match_type: MatchType
    start_ms: float


class FeedbackRenderer:
    """Manages visual feedback effects for matched notes."""

    def __init__(self):
        self._effects: dict[tuple[float, int], _FeedbackEffect] = {}
        self._streak = 0

    @property
    def streak(self) -> int:
        return self._streak

    def add_results(self, results: list[MatchResult], playback_ms: float) -> None:
        """Create visual effects from match results and update streak."""
        for result in results:
            for event in result.matched_events:
                key = (event.timestamp_ms, event.string)
                self._effects[key] = _FeedbackEffect(
                    note_key=key,
                    match_type=result.match_type,
                    start_ms=playback_ms,
                )

            # Update streak
            if result.match_type == MatchType.HIT:
                self._streak += 1
            elif result.match_type in (MatchType.MISS, MatchType.CLOSE):
                self._streak = 0

    def get_note_color(
        self,
        event: NoteEvent,
        base_color: tuple[int, int, int],
        playback_ms: float,
        is_past: bool,
    ) -> tuple[int, int, int]:
        """Get the display color for a note event.

        Returns feedback color if an active effect exists, otherwise
        base_color (or dimmed if past the hit zone).
        """
        key = (event.timestamp_ms, event.string)
        effect = self._effects.get(key)
        if effect is not None:
            elapsed = playback_ms - effect.start_ms
            if elapsed <= EFFECT_DURATION_MS:
                return _MATCH_COLORS.get(effect.match_type, base_color)
            # Effect expired but state is permanent — still show color (dimmed)
            color = _MATCH_COLORS.get(effect.match_type)
            if color is not None:
                return dimmed(color, 0.6)

        return dimmed(base_color) if is_past else base_color

    def draw_streak(self, surface: pygame.Surface, font: pygame.font.Font,
                    x: int, y: int) -> None:
        """Draw streak counter if streak >= 3."""
        if self._streak < 3:
            return
        text = f"{self._streak}x streak"
        surf = font.render(text, True, FEEDBACK_STREAK)
        surface.blit(surf, (x - surf.get_width() // 2, y))

    def draw_stats(self, surface: pygame.Surface, stats: dict,
                   font: pygame.font.Font, x: int, y: int) -> None:
        """Draw accuracy stats (top-right area)."""
        acc = stats["accuracy_percent"]
        line1 = f"Accuracy: {acc:.0f}%"
        line2 = f"H:{stats['hits']} C:{stats['close']} M:{stats['misses']}"

        surf1 = font.render(line1, True, FEEDBACK_HIT if acc >= 80 else HUD_TEXT_COLOR)
        surf2 = font.render(line2, True, HUD_TEXT_COLOR)

        surface.blit(surf1, (x - surf1.get_width(), y))
        surface.blit(surf2, (x - surf2.get_width(), y + surf1.get_height() + 2))

    def cleanup(self, playback_ms: float) -> None:
        """Remove expired effects."""
        expired = [
            key for key, eff in self._effects.items()
            if playback_ms - eff.start_ms > EFFECT_DURATION_MS * 2
        ]
        for key in expired:
            del self._effects[key]

    def reset(self) -> None:
        """Clear all effects and streak."""
        self._effects.clear()
        self._streak = 0
