"""Shared match-mode enum used by matcher and audio pipeline.

Placed in the audio package so audio modules (stabilizer, capture) can
reference it without depending on the matcher module, breaking a circular
import.
"""

from __future__ import annotations

from enum import Enum


class MatchMode(Enum):
    """Matching strictness profile.

    ARCADE: forgiving — octave equivalence, chord auto-complete (mark all siblings).
    PRACTICE: partial credit — only the matched note is marked, no auto-complete.
    JUDGE: strict — every note must be independently supported; pitch_strict forced on.
    """

    ARCADE = "arcade"
    PRACTICE = "practice"
    JUDGE = "judge"


def _coerce_match_mode(value: MatchMode | str) -> MatchMode:
    """Accept a MatchMode or its string value (e.g. from config)."""
    if isinstance(value, MatchMode):
        return value
    normalized = str(value).strip().lower()
    for m in MatchMode:
        if m.value == normalized:
            return m
    raise ValueError(f"unknown MatchMode: {value!r}")
