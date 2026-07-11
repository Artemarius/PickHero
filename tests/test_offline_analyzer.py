"""Tests for pickhero.audio.offline_analyzer -- after-take analysis scaffold."""
from __future__ import annotations

import numpy as np
import pytest

from pickhero.audio.offline_analyzer import OfflineAnalyzer, OfflineResult


class TestOfflineAnalyzerConstruction:
    """OfflineAnalyzer constructor tests."""

    def test_default_sample_rate(self):
        """Constructor uses default sample_rate 44100."""
        analyzer = OfflineAnalyzer()
        assert analyzer.sample_rate == 44100

    def test_custom_sample_rate(self):
        """Constructor accepts a custom sample_rate."""
        analyzer = OfflineAnalyzer(sample_rate=48000)
        assert analyzer.sample_rate == 48000


class TestBasicPitchAvailability:
    """Basic Pitch detection tests."""

    def test_is_full_analysis_available_returns_bool(self):
        """is_full_analysis_available returns a bool (False when not installed)."""
        analyzer = OfflineAnalyzer()
        assert isinstance(analyzer.is_full_analysis_available, bool)
        # basic_pitch is not installed in the test environment.
        assert analyzer.is_full_analysis_available is False


class TestAnalyzeMethod:
    """OfflineAnalyzer.analyze() tests."""

    def test_empty_audio_returns_empty_list(self):
        """analyze with empty audio returns an empty list."""
        analyzer = OfflineAnalyzer()
        audio = np.zeros(0, dtype=np.float32)
        result = analyzer.analyze(audio)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_non_empty_audio_returns_list(self):
        """analyze with non-empty audio returns a list (may be empty)."""
        analyzer = OfflineAnalyzer()
        audio = np.zeros(44100, dtype=np.float32)
        result = analyzer.analyze(audio)
        assert isinstance(result, list)

    def test_analyze_with_matched_pairs(self):
        """analyze accepts matched_pairs argument without error."""
        analyzer = OfflineAnalyzer()
        audio = np.zeros(44100, dtype=np.float32)
        pairs = [("dummy_pair",)]
        result = analyzer.analyze(audio, matched_pairs=pairs)
        assert isinstance(result, list)

    def test_analyze_with_none_matched_pairs(self):
        """analyze accepts matched_pairs=None."""
        analyzer = OfflineAnalyzer()
        audio = np.zeros(44100, dtype=np.float32)
        result = analyzer.analyze(audio, matched_pairs=None)
        assert isinstance(result, list)

    def test_analyze_returns_new_list_each_call(self):
        """Each analyze call returns an independent list."""
        analyzer = OfflineAnalyzer()
        audio = np.zeros(44100, dtype=np.float32)
        r1 = analyzer.analyze(audio)
        r2 = analyzer.analyze(audio)
        assert r1 is not r2


class TestTranscribeMethod:
    """OfflineAnalyzer.transcribe() tests."""

    def test_empty_audio_returns_offline_result_with_empty_lists(self):
        """transcribe with empty audio returns OfflineResult with empty lists."""
        analyzer = OfflineAnalyzer()
        audio = np.zeros(0, dtype=np.float32)
        result = analyzer.transcribe(audio)
        assert isinstance(result, OfflineResult)
        assert result.midi_notes == []
        assert result.timing_errors_ms == []
        assert result.pitch_bend_curves == {}
        assert result.technique_verdicts == []

    def test_non_empty_audio_returns_offline_result(self):
        """transcribe with non-empty audio returns an OfflineResult."""
        analyzer = OfflineAnalyzer()
        audio = np.zeros(44100, dtype=np.float32)
        result = analyzer.transcribe(audio)
        assert isinstance(result, OfflineResult)
        # Scaffold returns empty lists for non-empty audio.
        assert result.midi_notes == []
        assert result.timing_errors_ms == []


class TestOfflineResult:
    """OfflineResult dataclass tests."""

    def test_defaults(self):
        """OfflineResult defaults to empty lists/dicts."""
        result = OfflineResult()
        assert result.midi_notes == []
        assert result.timing_errors_ms == []
        assert result.pitch_bend_curves == {}
        assert result.technique_verdicts == []

    def test_to_dict_returns_expected_keys(self):
        """to_dict() returns the expected keys with correct types."""
        result = OfflineResult()
        d = result.to_dict()
        assert set(d.keys()) == {"midi_notes", "timing_errors_ms", "pitch_bend_curves", "technique_verdicts"}
        assert isinstance(d["midi_notes"], list)
        assert isinstance(d["timing_errors_ms"], list)
        assert isinstance(d["pitch_bend_curves"], dict)
        # technique_verdicts is serialized as a count (len).
        assert isinstance(d["technique_verdicts"], int)

    def test_to_dict_with_data(self):
        """to_dict() serialises populated fields correctly."""
        result = OfflineResult(
            midi_notes=[40, 45, 47],
            timing_errors_ms=[10.5, -3.2, 0.0],
            pitch_bend_curves={40: [0.0, 50.0, 100.0]},
            technique_verdicts=["dummy"],
        )
        d = result.to_dict()
        assert d["midi_notes"] == [40, 45, 47]
        assert d["timing_errors_ms"] == [10.5, -3.2, 0.0]
        assert d["pitch_bend_curves"] == {40: [0.0, 50.0, 100.0]}
        assert d["technique_verdicts"] == 1

    def test_pitch_bend_curves_copied(self):
        """to_dict() returns a copy, not the original dict reference."""
        result = OfflineResult(pitch_bend_curves={60: [0.0]})
        d = result.to_dict()
        # Mutating the dict should not affect the original.
        d["pitch_bend_curves"]["new"] = [1.0]
        assert 60 in result.pitch_bend_curves
        assert "new" not in result.pitch_bend_curves
