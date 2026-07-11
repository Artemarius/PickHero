"""Tests for pickhero.adaptive — phrase mastery and difficulty control."""

import copy

import pytest

from pickhero.adaptive import (
    AdaptiveDifficultyController,
    PhraseMastery,
    _normalise_accuracy,
)
from pickhero.tabs.timeline import NoteEvent, SongMetadata, Timeline


# ─── Helper factories ─────────────────────────────────────────────────


def make_note(
    timestamp_ms: float = 1000.0,
    midi_note: int = 40,
    string: int = 6,
    fret: int = 0,
    measure: int = 0,
    phrase_id: int = 0,
    difficulty_level: int = 3,
    duration_ms: float = 500.0,
) -> NoteEvent:
    return NoteEvent(
        timestamp_ms=timestamp_ms,
        duration_ms=duration_ms,
        midi_note=midi_note,
        string=string,
        fret=fret,
        measure=measure,
        phrase_id=phrase_id,
        difficulty_level=difficulty_level,
    )


def make_timeline(notes: list[NoteEvent]) -> Timeline:
    return Timeline(
        notes=notes,
        metadata=SongMetadata(title="test", artist="test", tempo=120),
    )


# ─── PhraseMastery default construction ───────────────────────────────


class TestPhraseMasteryDefaults:
    """Verify dataclass field defaults stay stable."""

    def test_default_level(self):
        pm = PhraseMastery(phrase_id=1, level=3)
        assert pm.phrase_id == 1
        assert pm.level == 3

    def test_max_level_default(self):
        pm = PhraseMastery(phrase_id=1, level=3)
        assert pm.max_level == 5

    def test_mastery_zero(self):
        pm = PhraseMastery(phrase_id=1, level=3)
        assert pm.mastery == 0.0

    def test_attempts_zero(self):
        pm = PhraseMastery(phrase_id=1, level=3)
        assert pm.attempts == 0

    def test_streaks_zero(self):
        pm = PhraseMastery(phrase_id=1, level=3)
        assert pm.success_streak == 0
        assert pm.failure_streak == 0

    def test_cooldown_zero(self):
        pm = PhraseMastery(phrase_id=1, level=3)
        assert pm.cooldown == 0

    def test_last_accuracy_zero(self):
        pm = PhraseMastery(phrase_id=1, level=3)
        assert pm.last_accuracy == 0.0


# ─── PhraseMastery.from_dict ──────────────────────────────────────────


class TestPhraseMasteryFromDict:
    def test_loads_all_known_fields(self):
        payload = dict(
            level=4,
            max_level=7,
            mastery=0.85,
            attempts=12,
            success_streak=3,
            failure_streak=1,
            cooldown=2,
            last_accuracy=0.9,
        )
        pm = PhraseMastery.from_dict(
            42, payload, default_level=3, max_level=7
        )
        assert pm.phrase_id == 42
        assert pm.level == 4
        assert pm.max_level == 7
        assert pm.mastery == pytest.approx(0.85)
        assert pm.attempts == 12
        assert pm.success_streak == 3
        assert pm.failure_streak == 1
        assert pm.cooldown == 2
        assert pm.last_accuracy == pytest.approx(0.9)

    def test_ignores_unknown_fields(self):
        payload = dict(level=3, unknown_thing="spam", extra=True)
        pm = PhraseMastery.from_dict(
            1, payload, default_level=2, max_level=5
        )
        assert pm.phrase_id == 1
        assert pm.level == 3
        # Unknown fields should not produce AttributeError or be present
        with pytest.raises(AttributeError):
            _ = pm.unknown_thing  # type: ignore[attr-defined]

    def test_default_level_when_missing_from_dict(self):
        pm = PhraseMastery.from_dict(
            2, {}, default_level=3, max_level=5
        )
        assert pm.level == 3

    def test_clamps_level_below_one(self):
        payload = dict(level=0)
        pm = PhraseMastery.from_dict(
            1, payload, default_level=3, max_level=5
        )
        assert pm.level == 1

    def test_clamps_level_at_max(self):
        payload = dict(level=99)
        pm = PhraseMastery.from_dict(
            1, payload, default_level=3, max_level=5
        )
        assert pm.level == 5

    def test_normalizes_accuracy_from_whole_percent(self):
        """mastery=85.0 (percent) → 0.85"""
        payload = dict(level=3, mastery=85.0, last_accuracy=20.0)
        pm = PhraseMastery.from_dict(
            1, payload, default_level=3, max_level=5
        )
        assert pm.mastery == pytest.approx(0.85)
        assert pm.last_accuracy == pytest.approx(0.20)

    def test_keeps_accuracy_already_normalized(self):
        payload = dict(level=3, mastery=0.75, last_accuracy=0.33)
        pm = PhraseMastery.from_dict(
            1, payload, default_level=3, max_level=5
        )
        assert pm.mastery == pytest.approx(0.75)
        assert pm.last_accuracy == pytest.approx(0.33)

    def test_normalizes_accuracy_over_100(self):
        """Clamp at 1.0 after dividing by 100."""
        payload = dict(level=3, mastery=150.0)
        pm = PhraseMastery.from_dict(
            1, payload, default_level=3, max_level=5
        )
        assert pm.mastery == pytest.approx(1.0)

    def test_clamps_negative_accuracy_to_zero(self):
        payload = dict(level=3, mastery=-0.5)
        pm = PhraseMastery.from_dict(
            1, payload, default_level=3, max_level=5
        )
        assert pm.mastery == pytest.approx(0.0)

    def test_enforces_non_negative_counts(self):
        payload = dict(
            level=3,
            attempts=-5,
            success_streak=-1,
            failure_streak=-2,
            cooldown=-3,
        )
        pm = PhraseMastery.from_dict(
            1, payload, default_level=3, max_level=5
        )
        assert pm.attempts == 0
        assert pm.success_streak == 0
        assert pm.failure_streak == 0
        assert pm.cooldown == 0

    def test_max_level_from_argument_overrides_payload(self):
        """The max_level argument always wins."""
        payload = dict(level=3, max_level=10)
        pm = PhraseMastery.from_dict(
            1, payload, default_level=3, max_level=7
        )
        assert pm.max_level == 7  # argument, not payload


# ─── PhraseMastery.export ─────────────────────────────────────────────


class TestPhraseMasteryExport:
    def test_export_omits_phrase_id(self):
        pm = PhraseMastery(phrase_id=7, level=3, mastery=0.5)
        exported = pm.export()
        assert "phrase_id" not in exported

    def test_export_contains_all_remaining_fields(self):
        pm = PhraseMastery(
            phrase_id=7,
            level=3,
            max_level=5,
            mastery=0.5,
            attempts=10,
            success_streak=2,
            failure_streak=0,
            cooldown=1,
            last_accuracy=0.88,
        )
        exported = pm.export()
        assert exported["level"] == 3
        assert exported["max_level"] == 5
        assert exported["mastery"] == pytest.approx(0.5)
        assert exported["attempts"] == 10
        assert exported["success_streak"] == 2
        assert exported["failure_streak"] == 0
        assert exported["cooldown"] == 1
        assert exported["last_accuracy"] == pytest.approx(0.88)
        assert len(exported) == 8

    def test_export_does_not_mutate_original(self):
        pm = PhraseMastery(phrase_id=1, level=3)
        exported = pm.export()
        exported["level"] = 99
        assert pm.level == 3


# ─── _normalise_accuracy ──────────────────────────────────────────────


class TestNormaliseAccuracy:
    def test_already_normalised(self):
        assert _normalise_accuracy(0.5) == pytest.approx(0.5)

    def test_percent_form(self):
        """50.0 → 0.5 (divide by 100)."""
        assert _normalise_accuracy(50.0) == pytest.approx(0.5)

    def test_over_100_clamps_to_one(self):
        assert _normalise_accuracy(150.0) == pytest.approx(1.0)

    def test_negative_clamps_to_zero(self):
        assert _normalise_accuracy(-0.1) == pytest.approx(0.0)

    def test_exactly_one_stays_one(self):
        assert _normalise_accuracy(1.0) == pytest.approx(1.0)

    def test_exactly_zero_stays_zero(self):
        assert _normalise_accuracy(0.0) == pytest.approx(0.0)

    def test_100_percent_becomes_one_after_division(self):
        assert _normalise_accuracy(100.0) == pytest.approx(1.0)


# ─── AdaptiveDifficultyController — disabled ──────────────────────────


class TestControllerDisabled:
    def test_always_accepts_when_disabled(self):
        notes = [
            make_note(phrase_id=0, difficulty_level=5),
            make_note(timestamp_ms=1500.0, phrase_id=1, difficulty_level=5),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=False)
        assert ctrl.accepts(make_note(difficulty_level=99))


# ─── AdaptiveDifficultyController — accepts ───────────────────────────


class TestControllerAccepts:
    def test_accepts_when_level_below_phrase_level(self):
        notes = [
            make_note(phrase_id=0, difficulty_level=3),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        # Initial controller level = min(initial_level=3, max_level from notes)
        # note with difficulty_level=2 is allowed
        assert ctrl.accepts(make_note(phrase_id=0, difficulty_level=2))

    def test_accepts_when_level_equal_to_phrase_level(self):
        notes = [
            make_note(phrase_id=0, difficulty_level=3),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        assert ctrl.accepts(make_note(phrase_id=0, difficulty_level=3))

    def test_rejects_when_level_exceeds_phrase_level(self):
        notes = [
            make_note(phrase_id=0, difficulty_level=3),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        assert not ctrl.accepts(make_note(phrase_id=0, difficulty_level=4))

    def test_accepts_with_unknown_phrase_id(self):
        """A note whose phrase_id has no registered notes is accepted."""
        notes = [
            make_note(phrase_id=0, difficulty_level=3),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        assert ctrl.accepts(make_note(phrase_id=99, difficulty_level=5))

    def test_accepts_negative_phrase_id_as_zero(self):
        notes = [
            make_note(phrase_id=0, difficulty_level=3),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        assert ctrl.accepts(make_note(phrase_id=-1, difficulty_level=3))

    def test_accepts_zero_difficulty_level_always_passes(self):
        """difficulty_level=0 always passes because 0 <= state.level."""
        notes = [
            make_note(phrase_id=0, difficulty_level=5),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        assert ctrl.accepts(make_note(phrase_id=0, difficulty_level=0))


# ─── AdaptiveDifficultyController — update_phrase ─────────────────────


class TestControllerUpdatePhrase:
    def test_level_up_after_two_strong_passes(self):
        notes = [make_note(phrase_id=0, difficulty_level=3)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        assert ctrl.phrases[0].level == 3

        # First strong pass
        changed = ctrl.update_phrase(0, 0.95)
    def test_level_up_after_two_strong_passes(self):
        notes = [make_note(phrase_id=0, difficulty_level=5)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(
            timeline, enabled=True, initial_level=3
        )
        assert ctrl.phrases[0].level == 3

        # First strong pass
        changed = ctrl.update_phrase(0, 0.95)
        assert not changed  # need two passes
        assert ctrl.phrases[0].level == 3

        # Second strong pass
        changed = ctrl.update_phrase(0, 0.95)
        assert changed
        assert ctrl.phrases[0].level == 4

    def test_level_down_after_two_weak_passes(self):
        notes = [make_note(phrase_id=0, difficulty_level=3)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        # Start at level 3; need level > 1 to demote
        ctrl.phrases[0].level = 3

        # First weak pass
        changed = ctrl.update_phrase(0, 0.3)
        assert not changed

        # Second weak pass
        changed = ctrl.update_phrase(0, 0.3)
        assert changed
        assert ctrl.phrases[0].level == 2

    def test_cooldown_prevents_immediate_re_change(self):
        """After a level change, cooldown=1 blocks the next update from
        changing levels again."""
        notes = [make_note(phrase_id=0, difficulty_level=5)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(
            timeline, enabled=True, initial_level=3
        )
        # Two strong passes → level up to 4
        ctrl.update_phrase(0, 0.95)
        ctrl.update_phrase(0, 0.95)  # level becomes 4, cooldown=1
        assert ctrl.phrases[0].level == 4
        assert ctrl.phrases[0].cooldown == 1

        # Next pass: cooldown still 1 (decremented to 0 after this, but
        # not yet before the promote/demote check)
        changed = ctrl.update_phrase(0, 0.95)
        assert not changed  # cooldown was 1, now 0, but not applied yet
        assert ctrl.phrases[0].level == 4  # no double-promotion

        # Now cooldown is 0, so next strong pass can promote
        changed = ctrl.update_phrase(0, 0.95)
        assert changed
        assert ctrl.phrases[0].level == 5

    def test_no_promotion_at_max_level(self):
        notes = [make_note(phrase_id=0, difficulty_level=5)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(
            timeline, enabled=True, initial_level=5
        )
        assert ctrl.phrases[0].level == 5
        changed = ctrl.update_phrase(0, 0.95)
        assert not changed
        changed = ctrl.update_phrase(0, 0.95)
        assert not changed  # already at max

    def test_no_demotion_below_level_one(self):
        notes = [make_note(phrase_id=0, difficulty_level=5)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        ctrl.phrases[0].level = 1
        changed = ctrl.update_phrase(0, 0.1)
        assert not changed
        changed = ctrl.update_phrase(0, 0.1)
        assert not changed  # already at floor

    def test_unknown_phrase_id_returns_false(self):
        notes = [make_note(phrase_id=0, difficulty_level=3)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        assert not ctrl.update_phrase(99, 0.95)

    def test_neutral_pass_does_not_trigger_change(self):
        """Accuracy in the neutral zone only decrements streaks."""
        notes = [make_note(phrase_id=0, difficulty_level=5)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(
            timeline, enabled=True, initial_level=3
        )
        # accuracy of 0.8 is neutral when target=0.88 (promote=0.90, demote=0.70)
        changed = ctrl.update_phrase(0, 0.80)
        assert not changed
        assert ctrl.phrases[0].success_streak == 0
        assert ctrl.phrases[0].failure_streak == 0

    def test_success_streak_resets_after_demotion(self):
        """Prompt a promotion, then two weak passes → streaks reset."""
        notes = [make_note(phrase_id=0, difficulty_level=5)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(
            timeline, enabled=True, initial_level=3
        )
        # Promote to 4
        ctrl.update_phrase(0, 0.95)
        ctrl.update_phrase(0, 0.95)
        assert ctrl.phrases[0].level == 4
        # After cooldown passes (one neutral pass)
        ctrl.update_phrase(0, 0.80)  # cooldown 1→0, neutral
        # Two weak passes → demote to 3
        ctrl.update_phrase(0, 0.30)
        ctrl.update_phrase(0, 0.30)
        assert ctrl.phrases[0].level == 3
        assert ctrl.phrases[0].success_streak == 0
        assert ctrl.phrases[0].failure_streak == 0

    def test_streak_decrement_on_neutral(self):
        """Neutral passes reduce existing streaks by 1 each."""
        notes = [make_note(phrase_id=0, difficulty_level=3)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        ctrl.update_phrase(0, 0.95)  # success_streak=1
        ctrl.update_phrase(0, 0.80)  # neutral → success_streak=0
        assert ctrl.phrases[0].success_streak == 0

    def test_mastery_ema_update(self):
        """Mastery is updated as EMA with 0.72/0.28 weighting."""
        notes = [make_note(phrase_id=0, difficulty_level=3)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        ctrl.update_phrase(0, 0.8)
        assert ctrl.phrases[0].mastery == pytest.approx(0.8)
        ctrl.update_phrase(0, 0.5)
        expected = 0.8 * 0.72 + 0.5 * 0.28
        assert ctrl.phrases[0].mastery == pytest.approx(expected)


# ─── AdaptiveDifficultyController — update_measure_range ──────────────


class TestControllerUpdateMeasureRange:
    def test_updates_single_phrase_across_its_measures(self):
        notes = [
            make_note(measure=0, phrase_id=0, difficulty_level=3),
            make_note(timestamp_ms=1500.0, measure=1, phrase_id=0, difficulty_level=3),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        assert ctrl.phrases[0].attempts == 0
        ctrl.update_measure_range(0, 1, 0.95)
        assert ctrl.phrases[0].attempts == 1

    def test_updates_multiple_phrases(self):
        notes = [
            make_note(measure=0, phrase_id=0, difficulty_level=3),
            make_note(timestamp_ms=1500.0, measure=0, phrase_id=1, difficulty_level=3),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        ctrl.update_measure_range(0, 0, 0.95)
        assert ctrl.phrases[0].attempts == 1
        assert ctrl.phrases[1].attempts == 1

    def test_orders_measure_range_lo_hi(self):
        """Even when start > end, the measure range is sorted."""
        notes = [
            make_note(measure=3, phrase_id=0, difficulty_level=3),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        ctrl.update_measure_range(3, 1, 0.95)
        assert ctrl.phrases[0].attempts == 1

    def test_empty_range_no_crash(self):
        timeline = make_timeline([])
        ctrl = AdaptiveDifficultyController(
            timeline, enabled=True
        )
        changed = ctrl.update_measure_range(0, 10, 0.95)
        assert not changed

    def test_skips_measures_with_no_notes(self):
        notes = [
            make_note(measure=0, phrase_id=0, difficulty_level=3),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        # update range covering measure 1 (which has nothing)
        changed = ctrl.update_measure_range(1, 1, 0.95)
        assert not changed

    def test_returns_true_when_any_phrase_changed(self):
        notes = [
            make_note(measure=0, phrase_id=0, difficulty_level=5),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(
            timeline, enabled=True, initial_level=3
        )
        changed = ctrl.update_measure_range(0, 0, 0.95)
        # not changed because only 1 strong pass so far
        assert not changed
        changed = ctrl.update_measure_range(0, 0, 0.95)
        # now changed because 2 strong passes → promoted
        assert changed

# ─── phrase_accuracy_from_measure_stats ────────────────────────────────


class TestPhraseAccuracyFromMeasureStats:
    def test_uses_quality_events_when_available(self):
        notes = [make_note(measure=0, phrase_id=0, difficulty_level=3)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        result = ctrl.phrase_accuracy_from_measure_stats(
            {0: {"quality_events": 4, "quality_sum": 3.2}}
        )
        assert result[0] == pytest.approx(0.8)  # 3.2 / 4

    def test_falls_back_to_hits_close_misses(self):
        notes = [make_note(measure=0, phrase_id=0, difficulty_level=3)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        result = ctrl.phrase_accuracy_from_measure_stats(
            {0: {"hits": 3, "close": 2, "misses": 1}}
        )
        # score = (3 + 2*0.45) / (3+2+1) = (3+0.9)/6 = 3.9/6 = 0.65
        assert result[0] == pytest.approx(0.65)

    def test_skips_measure_with_zero_total_events(self):
        notes = [make_note(measure=0, phrase_id=0, difficulty_level=3)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        result = ctrl.phrase_accuracy_from_measure_stats(
            {0: {"hits": 0, "close": 0, "misses": 0}}
        )
        assert 0 not in result

    def test_skips_measure_with_no_notes(self):
        notes = [make_note(measure=0, phrase_id=0, difficulty_level=3)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        result = ctrl.phrase_accuracy_from_measure_stats(
            {1: {"hits": 5, "close": 0, "misses": 0}}
        )
        assert result == {}

    def test_aggregates_across_multiple_measures(self):
        """Weighted average across two measures for same phrase."""
        notes = [
            make_note(measure=0, phrase_id=0, difficulty_level=3),
            make_note(timestamp_ms=1500.0, measure=1, phrase_id=0, difficulty_level=3),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        result = ctrl.phrase_accuracy_from_measure_stats({
            0: {"quality_events": 10, "quality_sum": 8.0},
            1: {"quality_events": 2, "quality_sum": 1.0},
        })
        # score_0 = 8.0/10 = 0.8, weight_0 = 10
        # score_1 = 1.0/2 = 0.5, weight_1 = 2
        # weighted = (0.8*10 + 0.5*2) / (10+2) = 9.0/12 = 0.75
        assert result[0] == pytest.approx(0.75)
        """One measure uses quality, another uses hits/close/misses fallback."""
        notes = [
            make_note(measure=0, phrase_id=0, difficulty_level=3),
            make_note(timestamp_ms=1500.0, measure=1, phrase_id=0, difficulty_level=3),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        result = ctrl.phrase_accuracy_from_measure_stats({
            0: {"quality_events": 4, "quality_sum": 3.6},  # score=0.9, weight=4
            1: {"hits": 2, "close": 1, "misses": 1},        # score=2.45/4=0.6125, weight=4
        })
        # weighted avg = (0.9*4 + 0.6125*4) / 8 = 6.05/8 = 0.75625
        assert result[0] == pytest.approx(0.75625)

    def test_multiple_phrases(self):
        notes = [
            make_note(measure=0, phrase_id=0, difficulty_level=3),
            make_note(timestamp_ms=1500.0, measure=0, phrase_id=1, difficulty_level=3),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        result = ctrl.phrase_accuracy_from_measure_stats({
            0: {"quality_events": 4, "quality_sum": 3.6},
        })
        assert result[0] == pytest.approx(0.9)
        assert result[1] == pytest.approx(0.9)

    def test_zero_quality_events_zero_count(self):
        """quality_events=0 and no hits/close/misses → skip."""
        notes = [make_note(measure=0, phrase_id=0, difficulty_level=3)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        result = ctrl.phrase_accuracy_from_measure_stats({
            0: {"quality_events": 0, "quality_sum": 0.0},
        })
        assert 0 not in result


# ─── Controller export ────────────────────────────────────────────────


class TestControllerExport:
    def test_returns_dict_with_string_keys(self):
        notes = [
            make_note(phrase_id=0, difficulty_level=3),
            make_note(timestamp_ms=1500.0, phrase_id=1, difficulty_level=4),
        ]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        exported = ctrl.export()
        assert isinstance(exported, dict)
        assert "0" in exported  # string key
        assert "1" in exported
        assert len(exported) == 2

    def test_exported_values_have_no_phrase_id(self):
        notes = [make_note(phrase_id=0, difficulty_level=3)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        exported = ctrl.export()
        assert "phrase_id" not in exported["0"]

    def test_export_reflects_updated_state(self):
        notes = [make_note(phrase_id=0, difficulty_level=3)]
        timeline = make_timeline(notes)
        ctrl = AdaptiveDifficultyController(
            timeline, enabled=True, initial_level=3
        )
        ctrl.update_phrase(0, 0.95)
        exported = ctrl.export()
        assert "last_accuracy" in exported["0"]
        assert exported["0"]["last_accuracy"] == pytest.approx(0.95)


# ─── Persisted state loading ──────────────────────────────────────────


class TestControllerWithPersistedState:
    def test_restores_from_persisted_dict(self):
        notes = [make_note(phrase_id=0, difficulty_level=5)]
        persisted = {
            "0": {
                "level": 4,
                "mastery": 0.75,
                "attempts": 10,
                "success_streak": 2,
                "failure_streak": 0,
                "cooldown": 0,
                "last_accuracy": 0.8,
            }
        }
        ctrl = AdaptiveDifficultyController(
            timeline=make_timeline(notes),
            enabled=True,
            persisted=persisted,
        )
        state = ctrl.phrases[0]
        assert state.level == 4
        assert state.mastery == pytest.approx(0.75)
        assert state.attempts == 10

    def test_bad_persisted_dict_falls_to_default(self):
        """When a persisted entry is not a dict, use the default level."""
        notes = [make_note(phrase_id=0, difficulty_level=5)]
        persisted = {"0": None}
        ctrl = AdaptiveDifficultyController(
            timeline=make_timeline(notes),
            enabled=True,
            initial_level=3,
            persisted=persisted,
        )
        state = ctrl.phrases[0]
        assert state.level == 3  # initial_level default


# ─── Empty timeline ───────────────────────────────────────────────────


class TestControllerEmptyTimeline:
    def test_no_crash_with_no_notes(self):
        timeline = make_timeline([])
        ctrl = AdaptiveDifficultyController(timeline, enabled=True)
        assert ctrl.phrases == {}
        # Accepts always True for unknown phrases
        assert ctrl.accepts(make_note())
        # update_phrase returns False for unknown
        assert not ctrl.update_phrase(0, 0.95)
        # export empty
        assert ctrl.export() == {}
