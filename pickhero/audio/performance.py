"""Performance event data model for the articulation coaching layer.

This module defines the four load-bearing dataclasses every other module
imports:

- :class:`TechniqueSpec` — what the tab expects (frozen, immutable).
- :class:`TechniqueCandidate` — what the real-time detector heard on a note.
- :class:`TechniqueVerdict` — the after-take Judge's grade + coaching text.
- :class:`PerformanceEvent` — the per-note performance record (mutable while a
  note is being built frame-by-frame; the analyzer mutates ``verdicts`` in
  place after the take).

Signatures are load-bearing: many modules import these names directly, so the
field names and defaults below are part of the contract. See
``guitar-articulation-feedback-plan.md`` Step 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pickhero.tabs.timeline import NoteEvent


__all__ = [
    "TechniqueSpec",
    "TechniqueCandidate",
    "TechniqueVerdict",
    "PerformanceEvent",
]


@dataclass(frozen=True)
class TechniqueSpec:
    """What the tab expects for a note's technique.

    A single :class:`~pickhero.tabs.timeline.NoteEvent` can carry multiple
    TechniqueSpec entries (e.g. ``palm_mute`` + ``bend``). Phase-1 detector
    only emits at most one :class:`TechniqueCandidate` per note — compound
    tagging is Phase 2 — but the data model is a tuple now so later phases do
    not need a migration.
    """

    kind: str
    """One of ``bend``, ``vibrato``, ``slide``, ``hammer_on``, ``pull_off``,
    ``palm_mute``, ``harmonic``, ``dead_note``, ``normal``."""

    subtype: str | None = None
    """Bend: ``whole``|``half``|``quarter``|``1.5``|``2``|``pre``|``release``.
    Slide: ``up``|``down``|``legato``|``shift``|``slide_in_below``|
    ``slide_in_above``|``slide_out``. Harmonic: ``natural``|``artificial``|
    ``pinch``|``tapped``|``semi``. Palm_mute: ``None`` (degrees handled by
    metrics). dead_note: ``left_hand``|``right_hand`` (inferred Phase 2)."""

    direction: str | None = None
    """``up`` or ``down``. For slide, and hammer_on/pull_off the direction is
    resolved in :class:`~pickhero.matcher.NoteMatcher` from the neighbor pitch
    delta, not stored in the tab."""

    target_cents: float | None = None
    """Bend target relative to the fretted pitch, in cents."""

    start_fret: int | None = None
    end_fret: int | None = None
    """Slide endpoints, if known."""

    curve: tuple[tuple[float, float], ...] = ()
    """Bend/release shape as ``(time_ms_offset, cents)`` points. Comes straight
    from the Guitar Pro bend points (converted to cents)."""

    grace: bool = False
    tied_to_previous: bool = False
    """True for legato notes that share a pick with the previous note."""

    legato_group_id: int | None = None
    """Optional grouping key for a run of tied legato notes."""

    expected_sounding_midi: int | None = None
    """Harmonics: the MIDI note the harmonic sounds at, computed from the open
    string + node ratio at load time (Patch 5). ``None`` for non-harmonic specs
    or when the loader can't resolve it."""

    node_fret: int | None = None
    """Natural harmonic node fret (e.g. 12 for the octave node). Used to
    compute ``expected_sounding_midi`` and to disambiguate natural vs fretted."""


@dataclass(frozen=True)
class TechniqueCandidate:
    """A technique the real-time detector believes happened on a note.

    Phase-1 detector emits at most one candidate per note. The after-take
    Judge may still emit a separate vibrato verdict on a note that already has
    a bend candidate — that is two *verdicts*, not two real-time candidates.
    """

    kind: str
    confidence: float
    """0.0–1.0 detector confidence."""

    subtype: str | None = None
    target_cents: float | None = None
    """For bends: the detected target reached (real-time value). ``None`` if
    the tab target is unknown in real time — the Judge fills it from the
    expected note event."""

    detected_cents: float | None = None
    metrics: dict = field(default_factory=dict)
    """Technique-specific graded metrics (rate_hz, depth_cents,
    decay_halflife_ms, …)."""


@dataclass(frozen=True)
class TechniqueVerdict:
    """The Judge's grade for one technique on one note."""

    kind: str
    grade: str
    """``good``, ``ok``, ``weak``, ``missed``, or ``n/a``."""

    score: float
    """0.0–1.0."""

    metrics: dict = field(default_factory=dict)
    explanation: str = ""
    """Human-readable coaching sentence."""


@dataclass
class PerformanceEvent:
    """Per-note performance record.

    Built incrementally by :class:`~pickhero.audio.articulation.ArticulationDetector`
    frame-by-frame while a note is sounding; closed (``release_ms`` set) and
    pushed to the completed list when the next onset fires. The analyzer
    appends :class:`TechniqueVerdict` entries after the take.

    Mutable on purpose: the real-time detector appends curve points each
    frame, and the after-take Judge appends verdicts. Callers that need an
    immutable snapshot should copy the fields they care about.
    """

    onset_ms: float
    release_ms: float | None = None
    string_candidate: int | None = None
    expected_note_event: "NoteEvent" | None = None

    f0_curve: list[tuple[float, float, float]] = field(default_factory=list)
    """Per-frame ``(time_ms, Hz, cents_from_onset)``. Empty when no pitch was
    detected (silence / dead note)."""

    energy_envelope: list[tuple[float, float]] = field(default_factory=list)
    """Per-frame ``(time_ms, rms)``."""

    spectral_features: list[dict] = field(default_factory=list)
    """Per-onset-frame dicts: ``centroid``, ``flux``, ``flatness``, ``hnr``."""

    onset_features: dict = field(default_factory=dict)
    """Onset-frame features: ``pick_transient``, ``fret_transient``,
    ``noise_burst`` (and their strengths)."""

    technique_candidates: list[TechniqueCandidate] = field(default_factory=list)
    verdicts: list[TechniqueVerdict] = field(default_factory=list)

    midi_note: int | None = None
    confidence: float = 0.0

    event_kind: str = "pick_onset"
    """Taxonomy of the performed event. One of:
    ``pick_onset`` (default — a picked note), ``legato_transition`` (hammer-on /
    pull-off destination sounding without a pick), ``slide_landing`` (pitch
    arrived at the slide destination fret), ``bend_target`` (bend reached plateau),
    ``noise_gesture`` (dead-note / rake — no pitch), ``sustain_update`` (a held
    note's periodic refresh), ``release`` (note ending). Default preserves all
    current construction sites that don't specify a kind."""

    def upsert_technique_candidate(self, kind: str, confidence: float, **fields) -> None:
        """Add or merge a technique candidate. De-dups by ``kind``; later
        frames refresh ``confidence``/``metrics``/``detected_cents``/
        ``target_cents`` without appending duplicates.

        This replaces the priority-return chain in
        :class:`~pickhero.audio.articulation.ArticulationDetector.process`,
        allowing compound tags (e.g. bend + vibrato) on a single note.

        Judge A/B fix: index-based replacement (not list.remove, which fails
        when metrics dicts differ) + preserve existing subtype/detected_cents
        when not explicitly overridden (avoids field loss on refresh).
        """
        for idx, c in enumerate(self.technique_candidates):
            if c.kind == kind:
                # Replace in place by index, preserving fields not overridden.
                self.technique_candidates[idx] = TechniqueCandidate(
                    kind=kind,
                    confidence=confidence,
                    subtype=fields.get("subtype", c.subtype),
                    target_cents=fields.get("target_cents", c.target_cents),
                    detected_cents=fields.get("detected_cents", c.detected_cents),
                    metrics=fields.get("metrics", c.metrics),
                )
                return
        self.technique_candidates.append(
            TechniqueCandidate(
                kind=kind,
                confidence=confidence,
                subtype=fields.get("subtype"),
                target_cents=fields.get("target_cents"),
                detected_cents=fields.get("detected_cents"),
                metrics=fields.get("metrics", {}),
            )
        )
