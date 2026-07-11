"""Tests for EventState enum and EventRuntime / ChordRoleVerdict dataclasses."""

import pytest
from pickhero.audio.event_state import EventState, EventRuntime, ChordRoleVerdict


class TestEventState:
    """EventState enum must expose all 8 values."""

    def test_values_count(self):
        assert len(EventState) == 8

    def test_pending(self):
        assert EventState.PENDING.value == "pending"

    def test_attacking(self):
        assert EventState.ATTACKING.value == "attacking"

    def test_pitched(self):
        assert EventState.PITCHED.value == "pitched"

    def test_sustaining(self):
        assert EventState.SUSTAINING.value == "sustaining"

    def test_released(self):
        assert EventState.RELEASED.value == "released"

    def test_hit(self):
        assert EventState.HIT.value == "hit"

    def test_partial(self):
        assert EventState.PARTIAL.value == "partial"

    def test_miss(self):
        assert EventState.MISS.value == "miss"

    def test_all_values_are_strings(self):
        for s in EventState:
            assert isinstance(s.value, str)


class TestEventRuntime:
    """EventRuntime dataclass default construction and sustain_ratio."""

    def test_default_state_pending(self):
        r = EventRuntime()
        assert r.state == EventState.PENDING

    def test_default_onset_ms_none(self):
        r = EventRuntime()
        assert r.onset_ms is None

    def test_default_first_pitch_ms_none(self):
        r = EventRuntime()
        assert r.first_pitch_ms is None

    def test_default_confidence_peak_zero(self):
        r = EventRuntime()
        assert r.confidence_peak == 0.0

    def test_default_sustain_hits_zero(self):
        r = EventRuntime()
        assert r.sustain_hits == 0

    def test_default_sustain_checks_zero(self):
        r = EventRuntime()
        assert r.sustain_checks == 0

    def test_default_checked_sustain_points_empty(self):
        r = EventRuntime()
        assert r.checked_sustain_points == set()

    def test_default_technique_evidence_empty(self):
        r = EventRuntime()
        assert r.technique_evidence == []

    def test_default_attack_quality_zero(self):
        r = EventRuntime()
        assert r.attack_quality == 0.0

    def test_default_release_quality_none(self):
        r = EventRuntime()
        assert r.release_quality is None

    def test_default_transition_quality_none(self):
        r = EventRuntime()
        assert r.transition_quality is None

    def test_default_technique_finalized_false(self):
        r = EventRuntime()
        assert r.technique_finalized is False

    def test_default_sustain_feedback_emitted_false(self):
        r = EventRuntime()
        assert r.sustain_feedback_emitted is False

    def test_default_terminal_emitted_false(self):
        r = EventRuntime()
        assert r.terminal_emitted is False

    def test_default_last_evaluated_ms_negative_one(self):
        r = EventRuntime()
        assert r.last_evaluated_ms == -1.0

    def test_sustain_ratio_default_returns_one(self):
        """When sustain_checks == 0, sustain_ratio returns 1.0."""
        r = EventRuntime()
        assert r.sustain_ratio == 1.0

    def test_sustain_ratio_zero_checks_explicit(self):
        r = EventRuntime(sustain_hits=5, sustain_checks=0)
        assert r.sustain_ratio == 1.0

    def test_sustain_ratio_negative_checks_returns_one(self):
        r = EventRuntime(sustain_hits=0, sustain_checks=-5)
        assert r.sustain_ratio == 1.0

    def test_sustain_ratio_half(self):
        r = EventRuntime(sustain_hits=3, sustain_checks=6)
        assert r.sustain_ratio == 0.5

    def test_sustain_ratio_sixty_percent(self):
        r = EventRuntime(sustain_hits=3, sustain_checks=5)
        assert r.sustain_ratio == 0.6

    def test_sustain_ratio_perfect(self):
        r = EventRuntime(sustain_hits=10, sustain_checks=10)
        assert r.sustain_ratio == 1.0

    def test_sustain_ratio_zero_hits(self):
        r = EventRuntime(sustain_hits=0, sustain_checks=4)
        assert r.sustain_ratio == 0.0

    def test_sustain_ratio_exceeds_one(self):
        """If sustain_hits > sustain_checks, ratio can exceed 1.0 (realistic?)."""
        r = EventRuntime(sustain_hits=7, sustain_checks=5)
        assert r.sustain_ratio == 1.4


class TestChordRoleVerdictIsHit:
    """ChordRoleVerdict.is_hit property."""

    def test_is_hit_all_true(self):
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=True,
            seventh_detected=True,
            extra_pitch_classes=0,
        )
        assert v.is_hit is True

    def test_is_hit_third_seventh_none(self):
        """None is not False → is_hit."""
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=None,
            seventh_detected=None,
            extra_pitch_classes=0,
        )
        assert v.is_hit is True

    def test_is_hit_third_true_seventh_none(self):
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=True,
            seventh_detected=None,
            extra_pitch_classes=0,
        )
        assert v.is_hit is True

    def test_is_hit_third_none_seventh_true(self):
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=None,
            seventh_detected=True,
            extra_pitch_classes=0,
        )
        assert v.is_hit is True

    def test_is_hit_root_missing(self):
        v = ChordRoleVerdict(root_detected=False)
        assert v.is_hit is False

    def test_is_hit_third_false(self):
        """third_detected is False → is_hit=False."""
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=False,
            seventh_detected=True,
            extra_pitch_classes=0,
        )
        assert v.is_hit is False

    def test_is_hit_seventh_false(self):
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=True,
            seventh_detected=False,
            extra_pitch_classes=0,
        )
        assert v.is_hit is False

    def test_is_hit_extra_pitch_classes(self):
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=True,
            seventh_detected=True,
            extra_pitch_classes=2,
        )
        assert v.is_hit is False

    def test_is_hit_root_missing_still_false_even_with_all_others(self):
        v = ChordRoleVerdict(
            root_detected=False,
            third_detected=True,
            seventh_detected=True,
            extra_pitch_classes=0,
        )
        assert v.is_hit is False


class TestChordRoleVerdictIsPartial:
    """ChordRoleVerdict.is_partial property."""

    def test_is_partial_root_and_third(self):
        v = ChordRoleVerdict(
            root_detected=True, third_detected=True, seventh_detected=None
        )
        assert v.is_partial is True

    def test_is_partial_root_and_seventh(self):
        v = ChordRoleVerdict(
            root_detected=True, third_detected=None, seventh_detected=True
        )
        assert v.is_partial is True

    def test_is_partial_root_and_both(self):
        v = ChordRoleVerdict(
            root_detected=True, third_detected=True, seventh_detected=True
        )
        assert v.is_partial is True

    def test_is_partial_root_third_seventh_none(self):
        """Root only, no third or seventh → not partial."""
        v = ChordRoleVerdict(
            root_detected=True, third_detected=None, seventh_detected=None
        )
        assert v.is_partial is False

    def test_is_partial_root_only_third_false_seventh_none(self):
        """third_detected is False → bool(False) = False."""
        v = ChordRoleVerdict(
            root_detected=True, third_detected=False, seventh_detected=None
        )
        assert v.is_partial is False

    def test_is_partial_root_missing(self):
        v = ChordRoleVerdict(
            root_detected=False, third_detected=True, seventh_detected=True
        )
        assert v.is_partial is False

    def test_is_partial_extra_pitch_classes_ignored(self):
        """Extra pitch classes do not block is_partial."""
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=True,
            seventh_detected=True,
            extra_pitch_classes=5,
        )
        assert v.is_partial is True

    def test_is_partial_third_seventh_both_false(self):
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=False,
            seventh_detected=False,
        )
        assert v.is_partial is False


class TestChordRoleVerdictIsClose:
    """ChordRoleVerdict.is_close property."""

    def test_is_close_root_only_all_defaults(self):
        v = ChordRoleVerdict(root_detected=True)
        assert v.is_close is True

    def test_is_close_root_only_explicit_none(self):
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=None,
            seventh_detected=None,
            fifth_detected=False,
            extra_pitch_classes=0,
        )
        assert v.is_close is True

    def test_is_close_third_detected_blocks(self):
        v = ChordRoleVerdict(
            root_detected=True, third_detected=True, fifth_detected=False
        )
        assert v.is_close is False

    def test_is_close_seventh_detected_blocks(self):
        v = ChordRoleVerdict(
            root_detected=True, seventh_detected=True, fifth_detected=False
        )
        assert v.is_close is False

    def test_is_close_fifth_detected_blocks(self):
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=None,
            seventh_detected=None,
            fifth_detected=True,
            extra_pitch_classes=0,
        )
        assert v.is_close is False

    def test_is_close_extra_pitch_classes_blocks(self):
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=None,
            seventh_detected=None,
            fifth_detected=False,
            extra_pitch_classes=1,
        )
        assert v.is_close is False

    def test_is_close_root_missing(self):
        v = ChordRoleVerdict(root_detected=False)
        assert v.is_close is False

    def test_is_close_third_false_does_not_block(self):
        """False is falsy, so third doesn't block is_close."""
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=False,
            seventh_detected=None,
            fifth_detected=False,
            extra_pitch_classes=0,
        )
        assert v.is_close is True

    def test_is_close_seventh_false_does_not_block(self):
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=None,
            seventh_detected=False,
            fifth_detected=False,
            extra_pitch_classes=0,
        )
        assert v.is_close is True


class TestChordRoleVerdictEdgeCases:
    """Edge cases not covered by the main property tests."""

    def test_root_not_detected_is_hit_false(self):
        v = ChordRoleVerdict(root_detected=False)
        assert v.is_hit is False

    def test_root_not_detected_is_partial_false(self):
        v = ChordRoleVerdict(root_detected=False)
        assert v.is_partial is False

    def test_root_not_detected_is_close_false(self):
        v = ChordRoleVerdict(root_detected=False)
        assert v.is_close is False

    def test_all_defaults_no_root(self):
        """Fully default ChordRoleVerdict has root_detected=False."""
        v = ChordRoleVerdict()
        assert v.is_hit is False
        assert v.is_partial is False
        assert v.is_close is False
        assert v.root_detected is False

    def test_dataclass_frozen(self):
        """ChordRoleVerdict should be frozen (immutable)."""
        v = ChordRoleVerdict(root_detected=True)
        with pytest.raises(AttributeError):
            v.root_detected = False

    def test_all_three_properties_independent(self):
        """A single verdict with a specific configuration yields
        a specific combination of is_hit, is_partial, is_close."""
        # Full hit
        hit = ChordRoleVerdict(
            root_detected=True,
            third_detected=True,
            seventh_detected=True,
            fifth_detected=True,
            extra_pitch_classes=0,
        )
        assert hit.is_hit is True
        assert hit.is_partial is True
        assert hit.is_close is False

        # Root only, close
        close = ChordRoleVerdict(
            root_detected=True,
            third_detected=None,
            seventh_detected=None,
            fifth_detected=False,
            extra_pitch_classes=0,
        )
        assert close.is_hit is True  # None is not False
        assert close.is_partial is False
        assert close.is_close is True

        # Partial: root + third, no seventh
        partial = ChordRoleVerdict(
            root_detected=True,
            third_detected=True,
            seventh_detected=None,
            fifth_detected=True,
            extra_pitch_classes=0,
        )
        assert partial.is_hit is True  # seventh=None is not False
        assert partial.is_partial is True
        assert partial.is_close is False

    def test_third_none_and_seventh_none_is_hit_true(self):
        """Both None is not False → is_hit."""
        v = ChordRoleVerdict(
            root_detected=True,
            third_detected=None,
            seventh_detected=None,
            extra_pitch_classes=0,
        )
        assert v.is_hit is True
