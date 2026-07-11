"""Tests for pickhero.audio.verification_policy — mode-aware policies."""

import pytest

from pickhero.audio.match_mode import MatchMode
from pickhero.audio.verification_policy import VerificationPolicy


class TestVerificationPolicyFromMode:
    """VerificationPolicy.from_mode() returns mode-appropriate defaults."""

    def test_arcade_mode(self):
        """ARCADE mode is the most forgiving."""
        policy = VerificationPolicy.from_mode(MatchMode.ARCADE)
        assert policy.name == "arcade"
        assert policy.pitch_cents_tolerance == 85.0
        assert policy.min_note_confidence == 0.34
        assert policy.min_chord_confidence == 0.30
        assert policy.allow_semitone_fallback is True
        assert policy.chord_hit_threshold == 0.72
        assert policy.chord_partial_threshold == 0.34
        assert policy.max_extra_pitch_classes == 2
        assert policy.sustain_required_ratio == 0.38
        assert policy.technique_present_threshold == 0.50
        assert policy.technique_uncertain_threshold == 0.22
        assert policy.max_strum_spread_ms == 220.0
        assert policy.timing_window_scale == 1.18

    def test_practice_mode(self):
        """PRACTICE mode is moderate."""
        policy = VerificationPolicy.from_mode(MatchMode.PRACTICE)
        assert policy.name == "practice"
        assert policy.pitch_cents_tolerance == 70.0
        assert policy.min_note_confidence == 0.50
        assert policy.min_chord_confidence == 0.44
        assert policy.allow_semitone_fallback is True
        assert policy.chord_hit_threshold == 0.80
        assert policy.chord_partial_threshold == 0.44
        assert policy.max_extra_pitch_classes == 1
        assert policy.sustain_required_ratio == 0.55
        assert policy.technique_present_threshold == 0.58
        assert policy.technique_uncertain_threshold == 0.28
        assert policy.max_strum_spread_ms == 145.0
        assert policy.timing_window_scale == 1.0

    def test_judge_mode(self):
        """JUDGE mode is the strictest."""
        policy = VerificationPolicy.from_mode(MatchMode.JUDGE)
        assert policy.name == "judge"
        assert policy.pitch_cents_tolerance == 45.0
        assert policy.min_note_confidence == 0.72
        assert policy.min_chord_confidence == 0.60
        assert policy.allow_semitone_fallback is False
        assert policy.chord_hit_threshold == 0.88
        assert policy.chord_partial_threshold == 0.58
        assert policy.max_extra_pitch_classes == 0
        assert policy.sustain_required_ratio == 0.72
        assert policy.technique_present_threshold == 0.68
        assert policy.technique_uncertain_threshold == 0.34
        assert policy.max_strum_spread_ms == 90.0
        assert policy.timing_window_scale == 0.82

    def test_all_fields_accessible(self):
        """Every VerificationPolicy field is accessible as an attribute."""
        policy = VerificationPolicy.from_mode(MatchMode.ARCADE)
        fields = [
            "name", "pitch_cents_tolerance", "min_note_confidence",
            "min_chord_confidence", "require_all_chord_notes",
            "allow_semitone_fallback", "chord_hit_threshold",
            "chord_partial_threshold", "max_extra_pitch_classes",
            "sustain_required_ratio", "technique_present_threshold",
            "technique_uncertain_threshold", "max_strum_spread_ms",
            "timing_window_scale",
        ]
        for field in fields:
            assert hasattr(policy, field), f"Missing field: {field}"


class TestVerificationPolicyAdapted:
    """VerificationPolicy.adapted() adjusts timing_window_scale based on accuracy."""

    @pytest.fixture
    def arcade(self):
        return VerificationPolicy.from_mode(MatchMode.ARCADE)

    @pytest.fixture
    def practice(self):
        return VerificationPolicy.from_mode(MatchMode.PRACTICE)

    def test_none_accuracy_returns_self(self, arcade):
        """recent_accuracy=None returns the same policy unchanged."""
        adapted = arcade.adapted(None)
        assert adapted is arcade

    def test_judge_mode_unchanged(self):
        """Judge mode never adapts regardless of accuracy."""
        judge = VerificationPolicy.from_mode(MatchMode.JUDGE)
        assert judge.adapted(0.5) is judge
        assert judge.adapted(0.01) is judge
        assert judge.adapted(0.99) is judge

    def test_low_accuracy_scales_timing_window(self, practice):
        """Accuracy below 0.55 increases timing_window_scale."""
        adapted = practice.adapted(0.30)
        assert adapted.timing_window_scale > practice.timing_window_scale
        # min_note_confidence should decrease
        assert adapted.min_note_confidence < practice.min_note_confidence
        # chord_hit_threshold should decrease
        assert adapted.chord_hit_threshold < practice.chord_hit_threshold

    def test_high_accuracy_scales_timing_window_down(self, practice):
        """Accuracy above 0.92 decreases timing_window_scale."""
        adapted = practice.adapted(0.95)
        assert adapted.timing_window_scale < practice.timing_window_scale
        # min_note_confidence should increase
        assert adapted.min_note_confidence > practice.min_note_confidence

    def test_medium_accuracy_returns_self(self, practice):
        """Accuracy in the normal range (0.55-0.92) returns policy unchanged."""
        adapted = practice.adapted(0.75)
        assert adapted is practice

    def test_accuracy_clamped(self, practice):
        """Accuracy is clamped to [0.0, 1.0] before computing adaptation."""
        # Values outside [0.0, 1.0] should not crash
        practice.adapted(-0.5)
        practice.adapted(1.5)
