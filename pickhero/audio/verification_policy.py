"""Mode-aware and performance-adaptive verification policies."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pickhero.audio.match_mode import MatchMode


@dataclass(frozen=True)
class VerificationPolicy:
    name: str
    pitch_cents_tolerance: float
    min_note_confidence: float
    min_chord_confidence: float
    require_all_chord_notes: bool
    allow_semitone_fallback: bool
    chord_hit_threshold: float
    chord_partial_threshold: float
    max_extra_pitch_classes: int
    sustain_required_ratio: float
    technique_present_threshold: float
    technique_uncertain_threshold: float
    max_strum_spread_ms: float
    timing_window_scale: float = 1.0

    @classmethod
    def from_mode(cls, mode: MatchMode) -> "VerificationPolicy":
        if mode == MatchMode.JUDGE:
            return cls(
                name="judge",
                pitch_cents_tolerance=45.0,
                min_note_confidence=0.72,
                min_chord_confidence=0.60,
                require_all_chord_notes=False,
                allow_semitone_fallback=False,
                chord_hit_threshold=0.88,
                chord_partial_threshold=0.58,
                max_extra_pitch_classes=0,
                sustain_required_ratio=0.72,
                technique_present_threshold=0.68,
                technique_uncertain_threshold=0.34,
                max_strum_spread_ms=90.0,
                timing_window_scale=0.82,
            )
        if mode == MatchMode.PRACTICE:
            return cls(
                name="practice",
                pitch_cents_tolerance=70.0,
                min_note_confidence=0.50,
                min_chord_confidence=0.44,
                require_all_chord_notes=False,
                allow_semitone_fallback=True,
                chord_hit_threshold=0.80,
                chord_partial_threshold=0.44,
                max_extra_pitch_classes=1,
                sustain_required_ratio=0.55,
                technique_present_threshold=0.58,
                technique_uncertain_threshold=0.28,
                max_strum_spread_ms=145.0,
                timing_window_scale=1.0,
            )
        return cls(
            name="arcade",
            pitch_cents_tolerance=85.0,
            min_note_confidence=0.34,
            min_chord_confidence=0.30,
            require_all_chord_notes=False,
            allow_semitone_fallback=True,
            chord_hit_threshold=0.72,
            chord_partial_threshold=0.34,
            max_extra_pitch_classes=2,
            sustain_required_ratio=0.38,
            technique_present_threshold=0.50,
            technique_uncertain_threshold=0.22,
            max_strum_spread_ms=220.0,
            timing_window_scale=1.18,
        )

    def adapted(self, recent_accuracy: float | None) -> "VerificationPolicy":
        """Apply gentle hysteretic assistance without changing note content.

        Dynamic difficulty controls arrangement density. This adaptation only
        prevents the detector from becoming punishing while a player is still
        learning; it never loosens Judge mode enough to accept wrong pitches.
        """
        if recent_accuracy is None or self.name == "judge":
            return self
        accuracy = max(0.0, min(1.0, recent_accuracy))
        if accuracy < 0.55:
            return replace(
                self,
                min_note_confidence=max(0.25, self.min_note_confidence - 0.08),
                min_chord_confidence=max(0.22, self.min_chord_confidence - 0.08),
                chord_hit_threshold=max(0.66, self.chord_hit_threshold - 0.06),
                timing_window_scale=min(1.35, self.timing_window_scale + 0.12),
            )
        if accuracy > 0.92:
            return replace(
                self,
                min_note_confidence=min(0.78, self.min_note_confidence + 0.04),
                chord_hit_threshold=min(0.90, self.chord_hit_threshold + 0.03),
                timing_window_scale=max(0.88, self.timing_window_scale - 0.06),
            )
        return self
