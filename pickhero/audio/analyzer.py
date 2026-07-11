"""After-take performance analyzer (the Judge framework).

Takes a list of ``(PerformanceEvent, NoteEvent)`` matched pairs (handed over
by :class:`~pickhero.matcher.NoteMatcher`) and the user's
:class:`~pickhero.config.ToneProfile`, and produces
:class:`~pickhero.audio.performance.TechniqueVerdict` entries — one per
expected technique, plus a "missing technique" verdict for any expected
technique the detector did not hear.

Phase 1 implements eight judges. Each judge's ``grade()`` thresholds and
explanation f-string are specified in the plan (Step 8) and implemented
verbatim. Later phases add judges to the same framework.

Detection ceilings (TENT, Su et al. 2019, DOI 10.5334/tismir.23) are the
realistic upper bound for in-phrase detection; the coaching thresholds here
are intentionally stricter than the detection gates in
:mod:`pickhero.audio.articulation`.
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

from pickhero.audio.performance import (
    PerformanceEvent,
    TechniqueSpec,
    TechniqueVerdict,
)

if TYPE_CHECKING:
    from pickhero.config import ToneProfile
    from pickhero.tabs.timeline import NoteEvent


__all__ = ["PerformanceAnalyzer", "Judge"]


# ─── Coaching thresholds (distinct from articulation.py detection gates) ─────

# Vibrato coaching band: 4-8 Hz, 30-80 cents. The detector accepts 3-8 Hz /
# 10-60 cents (articulation.py). These are the grading bands.
_VIB_COACH_RATE_GOOD = (4.0, 8.0)
_VIB_COACH_RATE_OK = (3.0, 8.0)
_VIB_COACH_DEPTH_GOOD = (30.0, 80.0)
_VIB_COACH_DEPTH_OK = (15.0, 80.0)
_VIB_REGULARITY_GOOD = 0.8

# Bend grading (cents from target)
_BEND_GOOD_TOL = 15.0
_BEND_OK_TOL = 30.0
_BEND_HOLD_GOOD = 20.0
_BEND_MISS_FRACTION = 0.5

# Slide landing (cents from target fret)
_SLIDE_GOOD_TOL = 15.0
_SLIDE_OK_TOL = 30.0

# Legato
_LEGATO_NOPICK_GOOD = 0.7
_LEGATO_VOLUME_GOOD = 0.4
_LEGATO_STABLE_CENTS = 20.0

# Palm mute
_PM_TIGHT_GOOD = (0.6, 0.9)
_PM_TOO_DEAD = 0.4
_PM_TOO_OPEN = 0.9

# Harmonic
_HARMONIC_PITCH_TOL = 30.0
_HARMONIC_STRENGTH_GOOD = 0.6


class Judge:
    """Abstract base for a Phase-1 technique judge."""

    kind: str = ""

    def grade(
        self,
        event: PerformanceEvent,
        note: "NoteEvent",
        spec: TechniqueSpec,
        tone_profile: "ToneProfile | None" = None,
    ) -> TechniqueVerdict:
        raise NotImplementedError


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _cents_curve(event: PerformanceEvent) -> list[float]:
    """Return the cents-from-onset values from the event's f0_curve."""
    return [pt[2] for pt in event.f0_curve]


def _empty_curve_verdict(kind: str) -> TechniqueVerdict:
    return TechniqueVerdict(
        kind=kind, grade="missed", score=0.0,
        explanation="no pitch detected",
    )


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


# ─── BendJudge ──────────────────────────────────────────────────────────────


class BendJudge(Judge):
    kind = "bend"

    def grade(self, event, note, spec, tone_profile=None) -> TechniqueVerdict:
        cents = _cents_curve(event)
        if not cents:
            return _empty_curve_verdict(self.kind)
        target = spec.target_cents if spec.target_cents is not None else 0.0
        if target <= 0:
            # No tab target — grade on raw rise only.
            target = max(cents) if cents else 0.0
        detected = max(cents)
        # Time to target: first time cents >= 0.9 * target
        time_to_target_ms = None
        for t, _, c in event.f0_curve:
            if c >= target * 0.9:
                time_to_target_ms = t - event.onset_ms
                break
        # Hold stability: stddev of cents while above 90% of target
        hold_values = [c for c in cents if c >= target * 0.9]
        hold_stability = _stddev(hold_values)
        overshoot = detected - target
        delta = detected - target

        if detected < target * _BEND_MISS_FRACTION:
            grade = "missed"
            score = 0.0
        elif abs(delta) <= _BEND_GOOD_TOL and hold_stability <= _BEND_HOLD_GOOD:
            grade = "good"
            score = 1.0
        elif abs(delta) <= _BEND_OK_TOL:
            grade = "ok"
            score = 0.6
        else:
            grade = "weak"
            score = 0.3

        explanation = (
            f"Bend reached {detected:.0f} of {target:.0f} cents — "
            f"{delta:+.0f} from target. Hold ±{hold_stability:.0f} cents."
        )
        return TechniqueVerdict(
            kind=self.kind, grade=grade, score=score,
            metrics={
                "detected_cents": detected,
                "target_cents": target,
                "delta_cents": delta,
                "time_to_target_ms": time_to_target_ms,
                "hold_stability_cents": hold_stability,
                "overshoot_cents": overshoot,
            },
            explanation=explanation,
        )


# ─── VibratoJudge ───────────────────────────────────────────────────────────


class VibratoJudge(Judge):
    kind = "vibrato"

    def grade(self, event, note, spec, tone_profile=None) -> TechniqueVerdict:
        cents = _cents_curve(event)
        if not cents:
            return _empty_curve_verdict(self.kind)
        # Use the sustain portion (skip onset frame)
        sustain = cents[1:] if len(cents) > 1 else cents
        if len(sustain) < 4:
            return TechniqueVerdict(
                kind=self.kind, grade="weak", score=0.2,
                explanation="Vibrato too short to grade.",
            )
        mean = statistics.fmean(sustain)
        # Zero crossings of the detrended curve
        crossings = 0
        for i in range(len(sustain) - 1):
            if (sustain[i] - mean) * (sustain[i + 1] - mean) < 0:
                crossings += 1
        duration_s = (event.f0_curve[-1][0] - event.f0_curve[0][0]) / 1000.0
        if duration_s <= 0:
            return TechniqueVerdict(
                kind=self.kind, grade="weak", score=0.2,
                explanation="Vibrato duration too short.",
            )
        rate_hz = crossings / (2.0 * duration_s)
        depth = (max(sustain) - min(sustain)) / 2.0
        # Regularity via inter-peak intervals
        peaks = _find_peaks(sustain, mean)
        regularity = _peak_regularity(peaks)
        center = mean

        # Grade rate
        rate_good = _VIB_COACH_RATE_GOOD[0] <= rate_hz <= _VIB_COACH_RATE_GOOD[1]
        rate_ok = _VIB_COACH_RATE_OK[0] <= rate_hz <= _VIB_COACH_RATE_OK[1]
        # Grade depth
        depth_good = _VIB_COACH_DEPTH_GOOD[0] <= depth <= _VIB_COACH_DEPTH_GOOD[1]
        depth_ok = _VIB_COACH_DEPTH_OK[0] <= depth <= _VIB_COACH_DEPTH_OK[1]
        # Grade regularity
        reg_good = regularity >= _VIB_REGULARITY_GOOD

        good_count = sum([rate_good, depth_good, reg_good])
        ok_count = sum([rate_ok, depth_ok, reg_good or regularity >= 0.5])
        if good_count >= 2:
            grade = "good"
            score = 1.0
        elif ok_count >= 2:
            grade = "ok"
            score = 0.6
        else:
            grade = "weak"
            score = 0.3

        if rate_good and depth_good:
            regularity_note = "well-controlled" if reg_good else "irregular"
        elif not rate_good:
            regularity_note = f"rate {rate_hz:.1f} Hz outside 4-8 Hz band"
        else:
            regularity_note = f"depth {depth:.0f} cents outside 30-80 band"

        explanation = (
            f"Vibrato {rate_hz:.1f} Hz, ±{depth:.0f} cents, center {center:+.0f}. "
            f"{regularity_note}."
        )
        return TechniqueVerdict(
            kind=self.kind, grade=grade, score=score,
            metrics={
                "rate_hz": rate_hz,
                "depth_cents": depth,
                "regularity": regularity,
                "center_offset_cents": center,
            },
            explanation=explanation,
        )


def _find_peaks(buffer: list[float], mean: float) -> list[int]:
    peaks = []
    for i in range(1, len(buffer) - 1):
        if buffer[i] > buffer[i - 1] and buffer[i] > buffer[i + 1] and buffer[i] > mean:
            peaks.append(i)
    return peaks


def _peak_regularity(peaks: list[int]) -> float:
    if len(peaks) < 2:
        return 0.0
    intervals = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    mean = statistics.fmean(intervals)
    if mean <= 0:
        return 0.0
    std = _stddev(intervals)
    cv = std / mean
    return max(0.0, 1.0 - cv)


# ─── SlideJudge ─────────────────────────────────────────────────────────────


class SlideJudge(Judge):
    kind = "slide"

    def grade(self, event, note, spec, tone_profile=None) -> TechniqueVerdict:
        cents = _cents_curve(event)
        if not cents:
            return _empty_curve_verdict(self.kind)
        # Find the monotonic slide segment (longest run of same-sign deltas)
        best_start, best_end = 0, 0
        cur_start = 0
        for i in range(1, len(cents)):
            if (cents[i] - cents[i - 1] >= 0) != (cents[cur_start + 1] - cents[cur_start] >= 0):
                if i - cur_start > best_end - best_start:
                    best_start, best_end = cur_start, i
                cur_start = i
        if len(cents) - cur_start > best_end - best_start:
            best_start, best_end = cur_start, len(cents)
        seg = cents[best_start:best_end] if best_end > best_start else cents
        if not seg:
            return TechniqueVerdict(
                kind=self.kind, grade="weak", score=0.3,
                explanation="Slide segment not found.",
            )
        # Landing: final cents relative to the slide target (Patch 5 populates
        # spec.target_cents from the tab's end_fret). When the loader couldn't
        # resolve a destination (grace-note slides, slide_out), grade on landing
        # stability instead of against zero — grading against zero would fail
        # any slide landing on a non-zero fret (Judge B finding).
        target_cents = spec.target_cents
        start_fret = spec.start_fret if spec.start_fret is not None else note.fret
        if target_cents is None and spec.end_fret is not None:
            target_cents = float((spec.end_fret - start_fret) * 100)
        landing = seg[-1]
        # Direction
        direction = "up" if seg[-1] >= seg[0] else "down"
        end_fret = spec.end_fret if spec.end_fret is not None else note.fret
        # Noise: mean spectral flatness during slide frames
        flatness_values = [f.get("flatness", 0.0) for f in event.spectral_features]
        noise = statistics.fmean(flatness_values) if flatness_values else 0.0
        duration_ms = (event.f0_curve[-1][0] - event.f0_curve[0][0]) if event.f0_curve else 0.0

        if target_cents is not None:
            # Known destination: grade on landing error vs target.
            landing_error = landing - target_cents
            abs_landing_error = abs(landing_error)
            if abs_landing_error <= _SLIDE_GOOD_TOL:
                grade = "good"
                score = 1.0
            elif abs_landing_error <= _SLIDE_OK_TOL:
                grade = "ok"
                score = 0.6
            else:
                grade = "weak"
                score = 0.3
        else:
            # No destination resolved (grace-note slide, slide_out): grade on
            # landing stability — how flat the pitch is at the end of the slide.
            # A stable landing (last few frames within _SLIDE_GOOD_TOL) is good.
            tail = seg[-3:] if len(seg) >= 3 else seg
            tail_spread = max(tail) - min(tail) if tail else abs(landing)
            if tail_spread <= _SLIDE_GOOD_TOL:
                grade = "good"
                score = 1.0
            elif tail_spread <= _SLIDE_OK_TOL:
                grade = "ok"
                score = 0.6
            else:
                grade = "weak"
                score = 0.3
            landing_error = landing  # no target to error against

        if target_cents is not None:
            explanation = (
                f"Slide landed {landing_error:+.0f} cents {direction} of fret {end_fret}."
            )
        else:
            explanation = (
                f"Slide {direction}, landed at fret {end_fret} (no target resolved)."
            )
        return TechniqueVerdict(
            kind=self.kind, grade=grade, score=score,
            metrics={
                "landing_error_cents": landing_error if target_cents is not None else None,
                "landing_cents": landing,
                "target_cents": target_cents,
                "direction": direction,
                "noise": noise,
                "duration_ms": duration_ms,
                "end_fret": end_fret,
            },
            explanation=explanation,
        )
# ─── LegatoJudge (hammer_on / pull_off) ─────────────────────────────────────


class LegatoJudge(Judge):
    kind = "hammer_on"

    def grade(self, event, note, spec, tone_profile=None) -> TechniqueVerdict:
        # Direction already resolved by the matcher; kind may be hammer_on or pull_off.
        kind_label = spec.kind  # "hammer_on" or "pull_off"
        onset_features = event.onset_features or {}
        pick_transient = onset_features.get("pick_transient_strength", 1.0)
        no_pick = max(0.0, 1.0 - pick_transient)
        # If the articulation detector classified this as a legato_transition
        # (Patch 2), the event itself IS the no-pick evidence — the destination
        # note sounded without a pick. Treat no_pick as maximal.
        if getattr(event, "event_kind", "pick_onset") == "legato_transition":
            no_pick = 1.0
        # Volume ratio: peak rms of this note vs previous picked note.
        # We don't have the previous event here directly; approximate from energy envelope.
        # Use a proxy if no prior data: assume 0.5 (medium) — graded as ok at best.
        volume_ratio = onset_features.get("hammer_volume_ratio", 0.5)
        # Transition time: from onset_features or default
        transition_ms = onset_features.get("transition_ms", 0.0)
        # Pitch stability over first 100ms
        early_cents = [c for t, _, c in event.f0_curve if t - event.onset_ms <= 100.0]
        stable = _stddev(early_cents) < _LEGATO_STABLE_CENTS

        good_count = sum([
            no_pick > _LEGATO_NOPICK_GOOD,
            volume_ratio > _LEGATO_VOLUME_GOOD,
            stable,
        ])
        if good_count >= 3:
            grade = "good"
            score = 1.0
        elif good_count >= 2:
            grade = "ok"
            score = 0.6
        else:
            grade = "weak"
            score = 0.3

        strength = "strong" if no_pick > _LEGATO_NOPICK_GOOD else "weak"
        stab_label = "stable" if stable else "unstable"
        explanation = (
            f"{kind_label.replace('_', '-').title()} {strength}, "
            f"transition {transition_ms:.0f}ms, pitch {stab_label}."
        )
        return TechniqueVerdict(
            kind=spec.kind, grade=grade, score=score,
            metrics={
                "no_pick_confidence": no_pick,
                "hammer_volume_ratio": volume_ratio,
                "transition_ms": transition_ms,
                "dest_pitch_stable": stable,
            },
            explanation=explanation,
        )


# ─── PalmMuteJudge ──────────────────────────────────────────────────────────


class PalmMuteJudge(Judge):
    kind = "palm_mute"

    # Hardcoded fallback thresholds (used when no ToneProfile).
    _NORMAL_HALFLIFE_MS = 200.0
    _NORMAL_CENTROID_HZ = 1500.0

    def grade(self, event, note, spec, tone_profile=None) -> TechniqueVerdict:
        # Find the palm-mute candidate metrics, or compute from the envelope.
        pm_candidate = next(
            (c for c in event.technique_candidates if c.kind == "palm_mute"), None
        )
        if pm_candidate is not None:
            halflife = pm_candidate.metrics.get("decay_halflife_ms", 0.0)
            centroid = pm_candidate.metrics.get("centroid_hz", 0.0)
        else:
            # Compute from energy envelope
            energies = [e[1] for e in event.energy_envelope]
            halflife = self._halflife_from_energy(energies, event)
            centroids = [f.get("centroid", 0.0) for f in event.spectral_features]
            centroid = statistics.fmean(centroids) if centroids else 0.0

        normal_halflife = self._NORMAL_HALFLIFE_MS
        normal_centroid = self._NORMAL_CENTROID_HZ
        if tone_profile is not None and tone_profile.templates:
            normal = tone_profile.templates.get("normal") or tone_profile.templates.get("palm_mute")
            if normal:
                normal_halflife = normal.get("decay_halflife_ms", normal_halflife)
                normal_centroid = normal.get("centroid_hz", normal_centroid)

        if normal_halflife > 0:
            tightness = 1.0 - (halflife / normal_halflife)
        else:
            tightness = 0.5
        centroid_ratio = (centroid / normal_centroid) if normal_centroid > 0 else 0.5

        # tightness = 1.0 - (halflife / normal_halflife): HIGH = very dead
        # (short decay), LOW = open (long sustain). Labels were inverted.
        if tightness < _PM_TOO_DEAD or centroid_ratio < _PM_TOO_DEAD:
            grade = "weak"
            score = 0.3
            note_text = "too open — mute more"
        elif tightness > _PM_TOO_OPEN or centroid_ratio > _PM_TOO_OPEN:
            grade = "weak"
            score = 0.3
            note_text = "too dead — lighten muting"
        elif _PM_TIGHT_GOOD[0] <= tightness <= _PM_TIGHT_GOOD[1] and centroid_ratio < 1.0:
            grade = "good"
            score = 1.0
            note_text = "well-controlled"
        else:
            grade = "ok"
            score = 0.6
            note_text = "acceptable"

        explanation = (
            f"Palm mute {tightness:.2f}, decay {halflife:.0f}ms. {note_text}."
        )
        return TechniqueVerdict(
            kind=self.kind, grade=grade, score=score,
            metrics={
                "decay_halflife_ms": halflife,
                "centroid_hz": centroid,
                "tightness_score": tightness,
                "centroid_ratio": centroid_ratio,
            },
            explanation=explanation,
        )

    @staticmethod
    def _halflife_from_energy(energies: list[float], event: PerformanceEvent) -> float:
        if not energies:
            return 0.0
        peak = max(energies)
        if peak <= 0:
            return 0.0
        half = peak / 2.0
        frame_ms = (event.f0_curve[1][0] - event.f0_curve[0][0]) if len(event.f0_curve) > 1 else 11.6
        for i, e in enumerate(energies):
            if e <= half:
                return i * frame_ms
        return len(energies) * frame_ms


# ─── HarmonicJudge (natural only in Phase 1) ───────────────────────────────


class HarmonicJudge(Judge):
    kind = "harmonic"

    def grade(self, event, note, spec, tone_profile=None) -> TechniqueVerdict:
        if not event.f0_curve:
            return _empty_curve_verdict(self.kind)
        # Expected sounding pitch for a natural harmonic at the played fret.
        # Patch 5 populates spec.expected_sounding_midi from the open string +
        # node ratio (fretted midi × ratio was wrong — it used the fretted
        # pitch, not the open string). Fall back to the old formula when the
        # loader hasn't populated the field.
        fret = note.fret
        ratio = _harmonic_ratio(fret)
        if getattr(spec, "expected_sounding_midi", None) is not None:
            expected_freq = _midi_to_freq(spec.expected_sounding_midi)
        else:
            expected_freq = _midi_to_freq(note.midi_note) * ratio
        detected_freq = event.f0_curve[0][1] if event.f0_curve else 0.0
        if expected_freq <= 0 or detected_freq <= 0:
            return TechniqueVerdict(
                kind=self.kind, grade="weak", score=0.3,
                explanation="Natural harmonic: pitch undetectable.",
            )
        cents_off = 1200.0 * math.log2(detected_freq / expected_freq)
        # Strength: from onset_features or first spectral frame
        strength = 0.0
        if event.onset_features:
            strength = event.onset_features.get("harmonic_strength", 0.0)
        if not strength and event.spectral_features:
            strength = event.spectral_features[0].get("hnr", 0.0)

        if abs(cents_off) <= _HARMONIC_PITCH_TOL and strength > _HARMONIC_STRENGTH_GOOD:
            grade = "good"
            score = 1.0
            detail = "clear and in tune"
        elif abs(cents_off) > _HARMONIC_PITCH_TOL and strength < 0.3:
            grade = "weak"
            score = 0.3
            detail = "too much fundamental — touch too heavy"
        else:
            grade = "ok"
            score = 0.6
            detail = f"pitch {cents_off:+.0f} cents off"

        verdict_label = grade
        explanation = f"Natural harmonic {verdict_label}: {detail}."
        return TechniqueVerdict(
            kind=self.kind, grade=grade, score=score,
            metrics={
                "expected_freq": expected_freq,
                "detected_freq": detected_freq,
                "cents_off": cents_off,
                "strength": strength,
            },
            explanation=explanation,
        )


def _harmonic_ratio(fret: int) -> float:
    """String-length ratio for common natural harmonic frets."""
    if fret == 12:
        return 2.0
    if fret == 7 or fret == 19:
        return 3.0
    if fret == 5 or fret == 24:
        return 4.0
    if fret == 9 or fret == 16:
        return 5.0  # not used in Phase-1 grading but supported
    # Default: assume octave (12th-fret equivalent)
    return 2.0


def _midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


# ─── DeadNoteJudge ──────────────────────────────────────────────────────────


class DeadNoteJudge(Judge):
    kind = "dead_note"

    def grade(self, event, note, spec, tone_profile=None) -> TechniqueVerdict:
        onset_features = event.onset_features or {}
        noise_burst = onset_features.get("noise_burst", 0.0)
        # No pitched sustain: f0_curve empty or all-zero cents with low confidence
        # has_pitch: a stable pitched note has a non-zero fundamental frequency
        # in the f0_curve. Checking cents != 0 misclassifies a stable pitched
        # note (cents == 0 relative to base) as a dead note.
        has_pitch = bool(event.f0_curve) and any(
            freq > 0 for _, freq, _ in event.f0_curve
        )
        if not has_pitch and noise_burst > 0.5:
            grade = "good"
            score = 1.0
            explanation = "Dead note struck cleanly"
        elif has_pitch:
            grade = "missed"
            score = 0.0
            explanation = "Dead note: pitched sustain detected"
        else:
            grade = "ok"
            score = 0.5
            explanation = "Dead note: weak percussive hit"
        return TechniqueVerdict(
            kind=self.kind, grade=grade, score=score,
            metrics={"noise_burst": noise_burst, "has_pitch": has_pitch},
            explanation=explanation,
        )


# ─── MissingTechniqueJudge ──────────────────────────────────────────────────


class MissingTechniqueJudge(Judge):
    """Emits a 'missed' verdict for any expected technique with no candidate."""

    kind = "missing"

    def grade(self, event, note, spec, tone_profile=None) -> TechniqueVerdict:
        return TechniqueVerdict(
            kind=spec.kind, grade="missed", score=0.0,
            explanation=f"Expected {spec.kind} but not detected.",
        )


class UnexpectedTechniqueJudge(Judge):
    """Emits a 'weak' penalty when a TechniqueCandidate is present on the
    event but not expected by any TechniqueSpec on the matched NoteEvent.

    Catches e.g. a palm-muted candidate on a note the tab expects clean.
    Skips ``harmonic``/``normal`` to avoid penalizing contextual harmonics."""

    kind = "unexpected"

    def grade(self, event, note, spec, tone_profile=None) -> TechniqueVerdict:
        # `spec` here is the candidate itself (passed by analyze as a
        # TechniqueCandidate wrapped in a shim). The candidate's kind is what
        # was unexpected.
        return TechniqueVerdict(
            kind="unexpected", grade="weak", score=0.3,
            explanation=f"Unexpected {spec.kind} detected on a clean note.",
        )

# ─── PerformanceAnalyzer ────────────────────────────────────────────────────


class PerformanceAnalyzer:
    """Runs the registered Phase-1 judges over matched (event, note) pairs."""

    def __init__(self, tone_profile: "ToneProfile | None" = None):
        self._tone_profile = tone_profile
        self._judges: dict[str, Judge] = {}
        self._register_phase1_judges()

    def _register_phase1_judges(self) -> None:
        for j in (
            BendJudge(),
            VibratoJudge(),
            SlideJudge(),
            LegatoJudge(),
            PalmMuteJudge(),
            HarmonicJudge(),
            DeadNoteJudge(),
        ):
            self._judges[j.kind] = j

    def _judge_for(self, kind: str) -> Judge:
        # hammer_on and pull_off both route to LegatoJudge
        if kind in ("hammer_on", "pull_off"):
            return self._judges["hammer_on"]
        return self._judges.get(kind, MissingTechniqueJudge())

    def analyze(self, pairs: list[tuple[PerformanceEvent, "NoteEvent"]]) -> list[PerformanceEvent]:
        """Grade each matched pair. Returns the events with verdicts appended."""
        for event, note in pairs:
            candidate_kinds = {c.kind for c in event.technique_candidates}
            for spec in note.techniques:
                judge = self._judge_for(spec.kind)
                verdict = judge.grade(event, note, spec, self._tone_profile)
                event.verdicts.append(verdict)
                # If the technique was expected but not detected, the judge
                # already returns 'missed'. MissingTechniqueJudge handles
                # specs whose kind has no dedicated judge.
            # Run the missing-technique check for any spec with no candidate
            # AND no dedicated judge that already produced a missed verdict.
            for spec in note.techniques:
                if spec.kind in candidate_kinds:
                    continue
                if spec.kind in self._judges or spec.kind in ("hammer_on", "pull_off"):
                    # A dedicated judge already ran — skip the generic missed.
                    continue
                event.verdicts.append(
                    MissingTechniqueJudge().grade(event, note, spec, self._tone_profile)
                )
            # UnexpectedTechniqueJudge: penalize candidates whose kind isn't
            # expected by any spec on this note (skip harmonic/normal to avoid
            # penalizing contextual harmonics).
            expected_kinds = {s.kind for s in note.techniques}
            for cand in event.technique_candidates:
                if cand.kind in expected_kinds:
                    continue
                if cand.kind in ("harmonic", "normal"):
                    continue
                event.verdicts.append(
                    UnexpectedTechniqueJudge().grade(event, note, cand, self._tone_profile)
                )
        return [e for e, _ in pairs]
