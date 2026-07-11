"""Musically weighted chord judgment for mono guitar input.

A mono cable cannot prove which physical string produced a duplicated pitch.
The scorer therefore judges unique pitch classes and their harmonic role. A
missing third or seventh matters much more than a missing doubled root/fifth.
"""

from __future__ import annotations

from dataclasses import dataclass

from pickhero.audio.evidence import ChordVerification, ExpectedNote


@dataclass(frozen=True)
class ChordScore:
    verdict: str
    score: float
    root_pitch_class: int
    bass_pitch_class: int
    expected_pitch_classes: tuple[int, ...]
    observed_pitch_classes: tuple[int, ...]
    missing_critical_roles: tuple[str, ...]
    extra_pitch_classes: tuple[int, ...]
    role_quality: dict[str, float]
    strum_spread_ms: float | None = None


# Common guitar chord families. The template is only used to infer the root and
# roles from a voicing; the player is still judged against the authored pitches.
_TEMPLATES: tuple[tuple[str, frozenset[int]], ...] = (
    ("major", frozenset({0, 4, 7})),
    ("minor", frozenset({0, 3, 7})),
    ("power", frozenset({0, 7})),
    ("sus2", frozenset({0, 2, 7})),
    ("sus4", frozenset({0, 5, 7})),
    ("diminished", frozenset({0, 3, 6})),
    ("augmented", frozenset({0, 4, 8})),
    ("dominant7", frozenset({0, 4, 7, 10})),
    ("major7", frozenset({0, 4, 7, 11})),
    ("minor7", frozenset({0, 3, 7, 10})),
    ("minor_major7", frozenset({0, 3, 7, 11})),
    ("sixth", frozenset({0, 4, 7, 9})),
    ("minor6", frozenset({0, 3, 7, 9})),
    ("add9", frozenset({0, 2, 4, 7})),
    ("minor_add9", frozenset({0, 2, 3, 7})),
    ("ninth", frozenset({0, 2, 4, 7, 10})),
)


def _infer_root(expected_pitch_classes: set[int], bass_pc: int) -> int:
    if len(expected_pitch_classes) <= 1:
        return bass_pc
    best_root = bass_pc
    best_score = float("-inf")
    for candidate in expected_pitch_classes:
        intervals = frozenset((pc - candidate) % 12 for pc in expected_pitch_classes)
        for _name, template in _TEMPLATES:
            matched = len(intervals & template)
            missing = len(template - intervals)
            foreign = len(intervals - template)
            # Exact and near-exact families dominate. Bass is only a tiebreaker,
            # so inversions can still resolve to the actual harmonic root.
            score = matched * 2.0 - missing * 0.75 - foreign * 1.1
            score += 0.15 if candidate == bass_pc else 0.0
            score += 0.10 if 0 in intervals else 0.0
            if score > best_score:
                best_score = score
                best_root = candidate
    return best_root


def _role_for_interval(interval: int) -> str:
    if interval == 0:
        return "root"
    if interval in (3, 4):
        return "third"
    if interval in (6, 7, 8):
        return "fifth"
    if interval in (10, 11):
        return "seventh"
    if interval in (2, 5, 9):
        return "extension"
    return "colour"


def score_chord(
    expected_notes: list[ExpectedNote],
    verification: ChordVerification,
    *,
    hit_threshold: float,
    partial_threshold: float,
    max_extra_for_hit: int,
    strum_spread_ms: float | None = None,
    max_strum_spread_ms: float | None = None,
) -> ChordScore:
    if not expected_notes:
        return ChordScore(
            verdict="miss",
            score=0.0,
            root_pitch_class=0,
            bass_pitch_class=0,
            expected_pitch_classes=(),
            observed_pitch_classes=(),
            missing_critical_roles=("chord",),
            extra_pitch_classes=(),
            role_quality={},
            strum_spread_ms=strum_spread_ms,
        )

    expected_pcs = {note.midi % 12 for note in expected_notes}
    bass_note = min(expected_notes, key=lambda note: note.midi)
    bass_pc = bass_note.midi % 12
    root_pc = _infer_root(expected_pcs, bass_pc)

    confidence_by_pc = {pc: 0.0 for pc in expected_pcs}
    for expected, observed in zip(expected_notes, verification.notes):
        pc = expected.midi % 12
        if observed.is_pitch_present:
            confidence_by_pc[pc] = max(
                confidence_by_pc[pc],
                max(0.62, min(1.0, float(observed.confidence))),
            )
    for pc in expected_pcs:
        energy = float(verification.pitch_class_energy.get(pc, 0.0))
        if pc in verification.observed_pitch_classes:
            confidence_by_pc[pc] = max(confidence_by_pc[pc], max(0.58, energy))

    observed_pcs = set(verification.observed_pitch_classes)
    observed_pcs.update(pc for pc, confidence in confidence_by_pc.items() if confidence > 0.0)
    extras = tuple(sorted(observed_pcs - expected_pcs))

    role_weights = {
        "root": 1.35,
        "third": 1.60,
        "fifth": 0.72,
        "seventh": 1.50,
        "extension": 1.18,
        "colour": 1.0,
    }
    weighted_score = 0.0
    total_weight = 0.0
    role_values: dict[str, list[float]] = {}
    critical: list[tuple[str, int]] = []
    for pc in sorted(expected_pcs):
        role = _role_for_interval((pc - root_pc) % 12)
        weight = role_weights[role]
        if pc == bass_pc:
            weight += 0.42
        confidence = confidence_by_pc[pc]
        weighted_score += weight * confidence
        total_weight += weight
        role_values.setdefault(role, []).append(confidence)
        if pc == bass_pc:
            role_values.setdefault("bass", []).append(confidence)
        if role in ("root", "third", "seventh", "extension"):
            critical.append((role, pc))
        elif pc == bass_pc and bass_pc != root_pc:
            critical.append(("bass", pc))

    score = weighted_score / total_weight if total_weight > 0.0 else 0.0
    # Foreign pitch classes are penalised by their measured chroma energy.
    extra_penalty = 0.0
    for pc in extras:
        energy = float(verification.pitch_class_energy.get(pc, 1.0))
        extra_penalty += 0.08 + min(0.12, max(0.0, energy) * 0.12)
    score -= extra_penalty

    if (
        strum_spread_ms is not None
        and max_strum_spread_ms is not None
        and max_strum_spread_ms > 0.0
        and strum_spread_ms > max_strum_spread_ms
    ):
        over = (strum_spread_ms - max_strum_spread_ms) / max_strum_spread_ms
        score -= min(0.16, 0.05 + over * 0.08)

    score = max(0.0, min(1.0, score))
    missing = []
    for role, pc in critical:
        if confidence_by_pc.get(pc, 0.0) < 0.34 and role not in missing:
            missing.append(role)

    role_quality = {
        role: sum(values) / len(values)
        for role, values in role_values.items()
        if values
    }
    essential_present = confidence_by_pc.get(root_pc, 0.0) >= 0.34
    if bass_pc != root_pc:
        essential_present = essential_present or confidence_by_pc.get(bass_pc, 0.0) >= 0.34
    distinct_present = sum(1 for value in confidence_by_pc.values() if value >= 0.34)

    if (
        score >= hit_threshold
        and not missing
        and len(extras) <= max_extra_for_hit
    ):
        verdict = "hit"
    elif score >= partial_threshold and essential_present and distinct_present >= min(2, len(expected_pcs)):
        verdict = "partial"
    else:
        verdict = "miss"

    return ChordScore(
        verdict=verdict,
        score=score,
        root_pitch_class=root_pc,
        bass_pitch_class=bass_pc,
        expected_pitch_classes=tuple(sorted(expected_pcs)),
        observed_pitch_classes=tuple(sorted(observed_pcs)),
        missing_critical_roles=tuple(missing),
        extra_pitch_classes=extras,
        role_quality=role_quality,
        strum_spread_ms=strum_spread_ms,
    )
