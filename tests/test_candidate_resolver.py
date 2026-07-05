"""Tests for the multi-candidate resolver in PitchEngine._resolve_candidates.

Covers: rejection of impossible candidates, tab-prior preference, octave
handling, and confidence-based selection.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pickhero.audio.detector import DetectedNote
from pickhero.audio.pitch_engine import PitchCandidate, PitchEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate(midi: int | None, freq: float, confidence: float,
               source: str = "yin_primary", cents_error: float | None = 0.0,
               source_flags: set[str] | None = None) -> PitchCandidate:
    """Build a PitchCandidate suitable for resolver tests."""
    return PitchCandidate(
        best_midi=midi,
        cents_error=cents_error,
        raw_frequency=freq,
        confidence=confidence,
        source_flags=source_flags or {source},
    )


# ---------------------------------------------------------------------------
# 1. test_resolver_rejects_zero_freq
# ---------------------------------------------------------------------------

class TestResolverRejection:
    """Reject candidates that are physically impossible."""

    def test_rejects_zero_freq(self):
        """A candidate with raw_frequency=0 is rejected even if midi looks OK."""
        engine = PitchEngine()
        candidates = [_candidate(midi=40, freq=0.0, confidence=0.95)]
        result = engine._resolve_candidates(candidates, tab_prior_midi=None)
        assert result is None

    def test_rejects_none_midi(self):
        """A candidate with best_midi=None is rejected."""
        engine = PitchEngine()
        candidates = [_candidate(midi=None, freq=82.41, confidence=0.9)]
        result = engine._resolve_candidates(candidates, tab_prior_midi=None)
        assert result is None

    def test_rejects_out_of_range_low(self):
        """MIDI below E2 (27) is rejected — too low for standard guitar."""
        engine = PitchEngine()
        candidates = [_candidate(midi=0, freq=16.35, confidence=0.95)]
        result = engine._resolve_candidates(candidates, tab_prior_midi=None)
        assert result is None

    def test_rejects_out_of_range_high(self):
        """MIDI above E5 (108) is rejected — too high for standard guitar."""
        engine = PitchEngine()
        candidates = [_candidate(midi=120, freq=12543.85, confidence=0.95)]
        result = engine._resolve_candidates(candidates, tab_prior_midi=None)
        assert result is None

    def test_rejects_low_confidence(self):
        """Candidates with confidence below 0.1 are rejected."""
        engine = PitchEngine()
        candidates = [_candidate(midi=40, freq=82.41, confidence=0.05)]
        result = engine._resolve_candidates(candidates, tab_prior_midi=None)
        assert result is None

    def test_rejects_multiple_impossible_leaves_empty(self):
        """When ALL candidates are invalid, resolver returns None."""
        engine = PitchEngine()
        candidates = [
            _candidate(midi=None, freq=0.0, confidence=0.0),
            _candidate(midi=120, freq=12543.85, confidence=0.95),
            _candidate(midi=40, freq=82.41, confidence=0.05),
        ]
        result = engine._resolve_candidates(candidates, tab_prior_midi=None)
        assert result is None


# ---------------------------------------------------------------------------
# 2. test_resolver_prefers_tab_prior
# ---------------------------------------------------------------------------

class TestResolverTabPrior:
    """Tab-prior should boost matching candidates."""

    def test_prefers_tab_prior_e2(self):
        """Two candidates at E2 (40) and E3 (52), tab_prior=40 → picks E2."""
        engine = PitchEngine()
        candidates = [
            _candidate(midi=52, freq=164.81, confidence=0.85, source="yin_primary"),
            _candidate(midi=40, freq=82.41, confidence=0.80, source="yin_large"),
        ]
        result = engine._resolve_candidates(candidates, tab_prior_midi=40)
        assert result is not None
        assert result.best_midi == 40

    def test_prefers_tab_prior_e3(self):
        """Two candidates at E2 and E3, tab_prior=52 → picks E3."""
        engine = PitchEngine()
        candidates = [
            _candidate(midi=40, freq=82.41, confidence=0.85, source="yin_primary"),
            _candidate(midi=52, freq=164.81, confidence=0.80, source="yin_large"),
        ]
        result = engine._resolve_candidates(candidates, tab_prior_midi=52)
        assert result is not None
        assert result.best_midi == 52

    def test_tab_prior_within_octave_bonus(self):
        """Tab prior within ±1 octave gets a confidence boost."""
        engine = PitchEngine()
        # G2 (55) and G4 (67), tab prior G3 (43) — both >12 semitones away
        candidates = [
            _candidate(midi=55, freq=98.0, confidence=0.85, source="yin_primary"),
            _candidate(midi=67, freq=415.3, confidence=0.80, source="yin_large"),
        ]
        result = engine._resolve_candidates(candidates, tab_prior_midi=43)
        assert result is not None
        # G2 is 12 semitones away (gets bonus), G4 is 24 away (no bonus)
        assert result.best_midi == 55


# ---------------------------------------------------------------------------
# 3. test_resolver_keeps_multiple_octaves_during_attack
# ---------------------------------------------------------------------------

class TestResolverOctaveHandling:
    """Multiple octave candidates handled via fundamental preference."""

    def test_e2_and_e3_both_valid_but_resolver_picks_lower(self):
        """E2 (40) and E3 (52) differ in octave group; lower fundamental wins
        when confidence diff < 0.15."""
        engine = PitchEngine()
        candidates = [
            _candidate(midi=52, freq=164.81, confidence=0.85, source="yin_primary"),
            _candidate(midi=40, freq=82.41, confidence=0.82, source="yin_large"),
        ]
        result = engine._resolve_candidates(candidates, tab_prior_midi=None)
        assert result is not None
        assert result.best_midi == 40
        assert "octave_corrected" in result.source_flags

    def test_higher_confidence_overrules_octave_bias(self):
        """When the upper candidate has substantially higher confidence, it wins."""
        engine = PitchEngine()
        candidates = [
            _candidate(midi=52, freq=164.81, confidence=0.98, source="yin_primary"),
            _candidate(midi=40, freq=82.41, confidence=0.70, source="yin_large"),
        ]
        result = engine._resolve_candidates(candidates, tab_prior_midi=None)
        assert result is not None
        assert result.best_midi == 52

    def test_single_candidate_passes_through(self):
        """A single valid candidate passes through unchanged."""
        engine = PitchEngine()
        candidates = [_candidate(midi=60, freq=261.63, confidence=0.90, source="yin_primary")]
        result = engine._resolve_candidates(candidates, tab_prior_midi=None)
        assert result is not None
        assert result.best_midi == 60
        assert result.raw_frequency == 261.63


# ---------------------------------------------------------------------------
# 4. test_resolver_picks_higher_confidence
# ---------------------------------------------------------------------------

class TestResolverConfidence:
    """When candidates share an octave group, confidence wins."""

    def test_same_octave_higher_confidence_wins(self):
        """E3 at 0.95 beats E3 at 0.80 (same midi, different sources)."""
        engine = PitchEngine()
        candidates = [
            _candidate(midi=52, freq=164.81, confidence=0.80, source="yin_large"),
            _candidate(midi=52, freq=164.90, confidence=0.95, source="yin_primary"),
        ]
        result = engine._resolve_candidates(candidates, tab_prior_midi=None)
        assert result is not None
        assert result.best_midi == 52
        assert result.confidence == 0.95

    def test_spectral_candidate_loses_to_primary(self):
        """When spectral and primary agree on midi, primary wins."""
        engine = PitchEngine()
        candidates = [
            _candidate(midi=52, freq=164.81, confidence=0.90, source="yin_primary"),
            _candidate(midi=52, freq=165.0, confidence=0.81, source="spectral"),
        ]
        result = engine._resolve_candidates(candidates, tab_prior_midi=None)
        assert result is not None
        assert result.best_midi == 52
        assert result.confidence == 0.90

    def test_no_candidate_clears_all_filters(self):
        """Empty candidate list returns None."""
        engine = PitchEngine()
        result = engine._resolve_candidates([], tab_prior_midi=None)
        assert result is None


# ---------------------------------------------------------------------------
# 5. Integration: full pipeline through _process_chunk
# ---------------------------------------------------------------------------

class TestProcessChunkIntegration:
    """End-to-end tests wiring _process_chunk → resolver."""

    def test_process_chunk_single_candidate(self):
        """A clean sine wave routed through the detector produces a valid
        PitchCandidate that the resolver accepts."""
        engine = PitchEngine(noise_gate_db=-120.0)
        sr = engine.sample_rate
        hop = engine.hop_size
        freq = 82.41  # E2 — low enough that 256 samples has enough cycles

        # Build a short sine; we mock the detector so the integration truly
        # tests the resolver path, not aubio's YIN algorithm.
        t = np.arange(hop, dtype=np.float32) / sr
        chunk = np.sin(2 * np.pi * freq * t).astype(np.float32)

        with patch.object(
            engine._detector, "process",
            return_value=DetectedNote(
                midi_note=40,          # E2
                frequency=82.41,
                confidence=0.95,
                name="E2",
                is_onset=True,
                onset_sample=0,
                performance=None,
            ),
        ):
            result = engine._process_chunk(chunk)

        assert result is not None
        assert result.candidate.best_midi == 40
        assert result.candidate.confidence > 0
        assert "yin_primary" in result.candidate.source_flags

    def test_process_chunk_zero_audio_returns_null_candidate(self):
        """Silent input (detector returns None) propagates through cleanly."""
        engine = PitchEngine()
        chunk = np.zeros(engine.hop_size, dtype=np.float32)

        with patch.object(
            engine._detector, "process",
            return_value=DetectedNote(
                midi_note=0,
                frequency=0.0,
                confidence=0.0,
                name="",
                is_onset=False,
                onset_sample=None,
                performance=None,
            ),
        ):
            result = engine._process_chunk(chunk)

        assert result is not None
        assert result.candidate.best_midi is None

    def test_process_chunk_tab_prior_boosts_candidate(self):
        """When tab_prior contains the detected MIDI, confidence is boosted."""
        engine = PitchEngine(noise_gate_db=-120.0)
        chunk = np.zeros(engine.hop_size, dtype=np.float32)

        with patch.object(
            engine._detector, "process",
            return_value=DetectedNote(
                midi_note=40,
                frequency=82.41,
                confidence=0.80,
                name="E2",
                is_onset=True,
                onset_sample=0,
                performance=None,
            ),
        ):
            # Set tab_prior to {40}
            with engine._tab_prior_lock:
                engine._tab_prior.clear()
                engine._tab_prior.add(40)

            result = engine._process_chunk(chunk)

        assert result is not None
        assert result.candidate.best_midi == 40
        assert "tab_prior" in result.candidate.source_flags
        assert result.candidate.confidence > 0.80
