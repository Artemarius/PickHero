"""Tests for pickhero.ui.feedback module."""

import pytest

from pickhero.matcher import MatchResult, MatchType
from pickhero.tabs.timeline import NoteEvent
from pickhero.ui.colors import FEEDBACK_HIT, FEEDBACK_CLOSE, FEEDBACK_MISS, dimmed
from pickhero.ui.feedback import EFFECT_DURATION_MS, FeedbackRenderer


def _note_event(timestamp_ms: float = 1000.0, midi_note: int = 64,
                string: int = 1) -> NoteEvent:
    return NoteEvent(
        timestamp_ms=timestamp_ms,
        duration_ms=500.0,
        midi_note=midi_note,
        string=string,
        fret=0,
    )


def _result(match_type: MatchType, events: list[NoteEvent],
            semitone_distance: int = 0) -> MatchResult:
    return MatchResult(
        match_type=match_type,
        matched_events=events,
        semitone_distance=semitone_distance,
    )


class TestNoteColor:
    def test_hit_color(self):
        """Active HIT effect returns green."""
        renderer = FeedbackRenderer()
        note = _note_event()
        renderer.add_results([_result(MatchType.HIT, [note])], playback_ms=1000.0)

        color = renderer.get_note_color(note, (200, 50, 50), 1100.0, is_past=True)
        assert color == FEEDBACK_HIT

    def test_close_color(self):
        """Active CLOSE effect returns yellow."""
        renderer = FeedbackRenderer()
        note = _note_event()
        renderer.add_results(
            [_result(MatchType.CLOSE, [note], semitone_distance=1)],
            playback_ms=1000.0,
        )

        color = renderer.get_note_color(note, (200, 50, 50), 1100.0, is_past=True)
        assert color == FEEDBACK_CLOSE

    def test_miss_color(self):
        """Active MISS effect returns red."""
        renderer = FeedbackRenderer()
        note = _note_event()
        renderer.add_results([_result(MatchType.MISS, [note])], playback_ms=1000.0)

        color = renderer.get_note_color(note, (200, 50, 50), 1100.0, is_past=True)
        assert color == FEEDBACK_MISS

    def test_default_color_no_effect(self):
        """No active effect -> base color or dimmed."""
        renderer = FeedbackRenderer()
        note = _note_event()
        base = (200, 50, 50)

        # Not past hit zone -> base color
        color = renderer.get_note_color(note, base, 900.0, is_past=False)
        assert color == base

        # Past hit zone -> dimmed
        color = renderer.get_note_color(note, base, 1600.0, is_past=True)
        assert color == dimmed(base)

    def test_effect_expiry(self):
        """Effect stops showing bright color after EFFECT_DURATION_MS."""
        renderer = FeedbackRenderer()
        note = _note_event()
        renderer.add_results([_result(MatchType.HIT, [note])], playback_ms=1000.0)

        # Within duration: bright green
        color = renderer.get_note_color(note, (200, 50, 50), 1400.0, is_past=True)
        assert color == FEEDBACK_HIT

        # After duration: dimmed green (not bright)
        color = renderer.get_note_color(
            note, (200, 50, 50),
            1000.0 + EFFECT_DURATION_MS + 1,
            is_past=True,
        )
        assert color != FEEDBACK_HIT
        # Should be dimmed version of the feedback color
        assert color == dimmed(FEEDBACK_HIT, 0.6)


class TestStreak:
    def test_streak_increments_on_hits(self):
        """Consecutive HITs increase streak."""
        renderer = FeedbackRenderer()

        for i in range(5):
            note = _note_event(timestamp_ms=float(i * 1000), string=1)
            renderer.add_results(
                [_result(MatchType.HIT, [note])],
                playback_ms=float(i * 1000 + 50),
            )

        assert renderer.streak == 5

    def test_streak_resets_on_miss(self):
        """MISS resets streak to 0."""
        renderer = FeedbackRenderer()

        # Build up a streak
        for i in range(3):
            note = _note_event(timestamp_ms=float(i * 1000), string=1)
            renderer.add_results([_result(MatchType.HIT, [note])], playback_ms=float(i * 1000))

        assert renderer.streak == 3

        # Miss resets
        miss_note = _note_event(timestamp_ms=3000.0, string=2)
        renderer.add_results([_result(MatchType.MISS, [miss_note])], playback_ms=3100.0)
        assert renderer.streak == 0

    def test_streak_resets_on_close(self):
        """CLOSE also resets streak."""
        renderer = FeedbackRenderer()

        note1 = _note_event(timestamp_ms=1000.0)
        renderer.add_results([_result(MatchType.HIT, [note1])], playback_ms=1050.0)
        assert renderer.streak == 1

        note2 = _note_event(timestamp_ms=2000.0)
        renderer.add_results(
            [_result(MatchType.CLOSE, [note2], semitone_distance=1)],
            playback_ms=2050.0,
        )
        assert renderer.streak == 0


class TestReset:
    def test_reset_clears_streak_and_effects(self):
        renderer = FeedbackRenderer()
        note = _note_event()
        renderer.add_results([_result(MatchType.HIT, [note])], playback_ms=1000.0)
        assert renderer.streak == 1

        renderer.reset()
        assert renderer.streak == 0

        # Effect should be gone — returns default color
        base = (200, 50, 50)
        color = renderer.get_note_color(note, base, 1100.0, is_past=True)
        assert color == dimmed(base)


class TestCleanup:
    def test_cleanup_removes_old_effects(self):
        renderer = FeedbackRenderer()
        note = _note_event()
        renderer.add_results([_result(MatchType.HIT, [note])], playback_ms=1000.0)

        # Way past expiry
        renderer.cleanup(1000.0 + EFFECT_DURATION_MS * 3)

        # Effect should be removed
        base = (200, 50, 50)
        color = renderer.get_note_color(note, base, 3000.0, is_past=True)
        assert color == dimmed(base)
