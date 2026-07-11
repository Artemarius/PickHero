"""Tests for the performance event data model."""

import pytest
from dataclasses import FrozenInstanceError
from pickhero.audio.performance import (
    TechniqueSpec,
    TechniqueCandidate,
    TechniqueVerdict,
    PerformanceEvent,
)


class TestTechniqueSpec:
    """TechniqueSpec is a frozen dataclass describing what the tab expects."""

    def test_minimal_construction(self):
        """A TechniqueSpec can be built with just the kind."""
        spec = TechniqueSpec(kind="bend")
        assert spec.kind == "bend"
        assert spec.subtype is None
        assert spec.direction is None
        assert spec.target_cents is None
        assert spec.start_fret is None
        assert spec.end_fret is None
        assert spec.curve == ()
        assert spec.grace is False
        assert spec.tied_to_previous is False
        assert spec.legato_group_id is None
        assert spec.expected_sounding_midi is None
        assert spec.node_fret is None

    def test_all_fields_specified(self):
        """All TechniqueSpec fields can be set at construction."""
        spec = TechniqueSpec(
            kind="slide",
            subtype="up",
            direction="up",
            target_cents=100.0,
            start_fret=3,
            end_fret=5,
            curve=((0.0, 0.0), (50.0, 100.0)),
            grace=False,
            tied_to_previous=True,
            legato_group_id=42,
            expected_sounding_midi=64,
            node_fret=12,
        )
        assert spec.kind == "slide"
        assert spec.subtype == "up"
        assert spec.direction == "up"
        assert spec.target_cents == 100.0
        assert spec.start_fret == 3
        assert spec.end_fret == 5
        assert spec.curve == ((0.0, 0.0), (50.0, 100.0))
        assert spec.grace is False
        assert spec.tied_to_previous is True
        assert spec.legato_group_id == 42
        assert spec.expected_sounding_midi == 64
        assert spec.node_fret == 12

    def test_frozen_prevents_mutation(self):
        """Assigning to a TechniqueSpec field raises FrozenInstanceError."""
        spec = TechniqueSpec(kind="vibrato")
        with pytest.raises(FrozenInstanceError):
            spec.kind = "bend"

    def test_frozen_prevents_attribute_addition(self):
        """Setting a new attribute on a frozen dataclass raises FrozenInstanceError."""
        spec = TechniqueSpec(kind="harmonic")
        with pytest.raises(FrozenInstanceError):
            spec.extra = "should not work"


class TestTechniqueCandidate:
    """TechniqueCandidate is a frozen dataclass for a real-time detection
    result."""

    def test_minimal_construction(self):
        """A TechniqueCandidate requires only kind and confidence."""
        c = TechniqueCandidate(kind="bend", confidence=0.85)
        assert c.kind == "bend"
        assert c.confidence == 0.85
        assert c.subtype is None
        assert c.target_cents is None
        assert c.detected_cents is None
        assert c.metrics == {}

    def test_all_fields_specified(self):
        """All TechniqueCandidate fields can be set at construction."""
        c = TechniqueCandidate(
            kind="slide",
            confidence=0.92,
            subtype="down",
            target_cents=50.0,
            detected_cents=48.3,
            metrics={"rate_hz": 5.0, "depth_cents": 30.0},
        )
        assert c.kind == "slide"
        assert c.confidence == 0.92
        assert c.subtype == "down"
        assert c.target_cents == 50.0
        assert c.detected_cents == 48.3
        assert c.metrics == {"rate_hz": 5.0, "depth_cents": 30.0}

    def test_frozen_prevents_mutation(self):
        """Assigning to a TechniqueCandidate field raises FrozenInstanceError."""
        c = TechniqueCandidate(kind="vibrato", confidence=0.7)
        with pytest.raises(FrozenInstanceError):
            c.confidence = 0.9

    def test_metrics_defaults_to_empty_dict(self):
        """metrics defaults to a fresh empty dict per instance."""
        a = TechniqueCandidate(kind="bend", confidence=0.8)
        b = TechniqueCandidate(kind="slide", confidence=0.9)
        assert a.metrics == {}
        assert b.metrics == {}
        # Each instance gets its own default, not shared
        a.metrics["rate_hz"] = 3.0  # frozen, but we can mutate the dict
        assert "rate_hz" in a.metrics
        assert b.metrics == {}


class TestTechniqueVerdict:
    """TechniqueVerdict is a frozen dataclass for the Judge's grade."""

    def test_minimal_construction(self):
        """A TechniqueVerdict requires kind, grade, and score."""
        v = TechniqueVerdict(kind="bend", grade="good", score=0.85)
        assert v.kind == "bend"
        assert v.grade == "good"
        assert v.score == 0.85
        assert v.metrics == {}
        assert v.explanation == ""

    def test_all_fields_specified(self):
        """All TechniqueVerdict fields can be set at construction."""
        v = TechniqueVerdict(
            kind="vibrato",
            grade="ok",
            score=0.65,
            metrics={"rate_hz": 4.2},
            explanation="Vibrato rate is a bit slow, aim for 5+ Hz.",
        )
        assert v.kind == "vibrato"
        assert v.grade == "ok"
        assert v.score == 0.65
        assert v.metrics == {"rate_hz": 4.2}
        assert v.explanation == "Vibrato rate is a bit slow, aim for 5+ Hz."

    def test_frozen_prevents_mutation(self):
        """Assigning to a TechniqueVerdict field raises FrozenInstanceError."""
        v = TechniqueVerdict(kind="slide", grade="weak", score=0.4)
        with pytest.raises(FrozenInstanceError):
            v.score = 0.5


class TestPerformanceEvent:
    """PerformanceEvent is a mutable per-note performance record."""

    def test_minimal_construction(self):
        """A PerformanceEvent requires only onset_ms; defaults are used."""
        ev = PerformanceEvent(onset_ms=100.0)
        assert ev.onset_ms == 100.0
        assert ev.release_ms is None
        assert ev.string_candidate is None
        assert ev.expected_note_event is None
        assert ev.f0_curve == []
        assert ev.energy_envelope == []
        assert ev.spectral_features == []
        assert ev.onset_features == {}
        assert ev.technique_candidates == []
        assert ev.verdicts == []
        assert ev.midi_note is None
        assert ev.confidence == 0.0
        assert ev.event_kind == "pick_onset"

    def test_default_event_kind_is_pick_onset(self):
        """event_kind defaults to 'pick_onset' when not specified."""
        ev = PerformanceEvent(onset_ms=0.0)
        assert ev.event_kind == "pick_onset"

    def test_event_kind_override(self):
        """event_kind can be overridden at construction."""
        ev = PerformanceEvent(onset_ms=0.0, event_kind="legato_transition")
        assert ev.event_kind == "legato_transition"

    def test_mutable_fields(self):
        """PerformanceEvent fields can be mutated after construction."""
        ev = PerformanceEvent(onset_ms=0.0)
        ev.release_ms = 500.0
        ev.confidence = 0.95
        ev.midi_note = 60
        ev.event_kind = "sustain_update"
        assert ev.release_ms == 500.0
        assert ev.confidence == 0.95
        assert ev.midi_note == 60
        assert ev.event_kind == "sustain_update"

    def test_upsert_adds_new_candidate(self):
        """upsert_technique_candidate adds a candidate when the kind is not present."""
        ev = PerformanceEvent(onset_ms=0.0)
        ev.upsert_technique_candidate("bend", 0.85, detected_cents=120.0)
        assert len(ev.technique_candidates) == 1
        c = ev.technique_candidates[0]
        assert c.kind == "bend"
        assert c.confidence == 0.85
        assert c.detected_cents == 120.0
        assert c.subtype is None
        assert c.target_cents is None
        assert c.metrics == {}

    def test_upsert_updates_existing_candidate(self):
        """upsert_technique_candidate replaces the candidate when the kind exists."""
        ev = PerformanceEvent(onset_ms=0.0)
        ev.upsert_technique_candidate("bend", 0.85, detected_cents=120.0)
        assert ev.technique_candidates[0].confidence == 0.85

        ev.upsert_technique_candidate("bend", 0.95, detected_cents=125.0)
        assert len(ev.technique_candidates) == 1
        c = ev.technique_candidates[0]
        assert c.kind == "bend"
        assert c.confidence == 0.95
        assert c.detected_cents == 125.0

    def test_upsert_preserves_subtype_and_cents_on_update(self):
        """On update, existing subtype and detected_cents are preserved unless
        explicitly overridden."""
        ev = PerformanceEvent(onset_ms=0.0)
        ev.upsert_technique_candidate(
            "slide", 0.8, subtype="up", detected_cents=300.0
        )

        # Refresh with new confidence and metrics, but omit subtype and detected_cents
        ev.upsert_technique_candidate("slide", 0.9, metrics={"rate": 1.0})
        assert len(ev.technique_candidates) == 1
        c = ev.technique_candidates[0]
        assert c.kind == "slide"
        assert c.confidence == 0.9
        assert c.subtype == "up"  # preserved from existing
        assert c.detected_cents == 300.0  # preserved from existing
        assert c.metrics == {"rate": 1.0}
        assert c.target_cents is None

    def test_upsert_overrides_subtype_when_provided(self):
        """When subtype is explicitly provided on update, it overrides the existing value."""
        ev = PerformanceEvent(onset_ms=0.0)
        ev.upsert_technique_candidate("slide", 0.8, subtype="up")
        ev.upsert_technique_candidate("slide", 0.9, subtype="down")
        assert ev.technique_candidates[0].subtype == "down"

    def test_upsert_overrides_detected_cents_when_provided(self):
        """When detected_cents is explicitly provided on update, it overrides."""
        ev = PerformanceEvent(onset_ms=0.0)
        ev.upsert_technique_candidate("bend", 0.8, detected_cents=100.0)
        ev.upsert_technique_candidate("bend", 0.9, detected_cents=105.0)
        assert ev.technique_candidates[0].detected_cents == 105.0

    def test_upsert_multiple_kinds_coexist(self):
        """Multiple different technique kinds can coexist in the candidates list."""
        ev = PerformanceEvent(onset_ms=0.0)
        ev.upsert_technique_candidate("bend", 0.85)
        ev.upsert_technique_candidate("vibrato", 0.75)
        ev.upsert_technique_candidate("slide", 0.90)
        assert len(ev.technique_candidates) == 3
        kinds = [c.kind for c in ev.technique_candidates]
        assert kinds == ["bend", "vibrato", "slide"]

    def test_upsert_with_metrics(self):
        """upsert_technique_candidate accepts and stores metrics."""
        ev = PerformanceEvent(onset_ms=0.0)
        ev.upsert_technique_candidate(
            "vibrato",
            0.8,
            metrics={"rate_hz": 6.2, "depth_cents": 15.0},
        )
        c = ev.technique_candidates[0]
        assert c.metrics == {"rate_hz": 6.2, "depth_cents": 15.0}

    def test_upsert_updates_metrics_on_refresh(self):
        """When refreshing a candidate, metrics are replaced with the new value."""
        ev = PerformanceEvent(onset_ms=0.0)
        ev.upsert_technique_candidate(
            "vibrato", 0.8, metrics={"rate_hz": 6.2, "depth_cents": 15.0}
        )
        ev.upsert_technique_candidate("vibrato", 0.85, metrics={"rate_hz": 7.0})
        c = ev.technique_candidates[0]
        assert c.confidence == 0.85
        assert c.metrics == {"rate_hz": 7.0}  # replaced, not merged
        # Previously-preserved fields (none set on first call) are still None
        assert c.subtype is None
        assert c.target_cents is None
        assert c.detected_cents is None

    def test_upsert_preserves_metrics_when_not_provided_on_update(self):
        """When updating without passing metrics, the existing metrics are preserved."""
        ev = PerformanceEvent(onset_ms=0.0)
        ev.upsert_technique_candidate(
            "vibrato", 0.8, metrics={"rate_hz": 6.2, "depth_cents": 15.0}
        )
        # Update only confidence, no metrics passed
        ev.upsert_technique_candidate("vibrato", 0.85)
        c = ev.technique_candidates[0]
        assert c.confidence == 0.85
        assert c.metrics == {"rate_hz": 6.2, "depth_cents": 15.0}  # preserved

    def test_verdicts_list_mutable(self):
        """The verdicts list can be appended to after construction."""
        ev = PerformanceEvent(onset_ms=0.0)
        v = TechniqueVerdict(kind="bend", grade="good", score=0.9)
        ev.verdicts.append(v)
        assert len(ev.verdicts) == 1
        assert ev.verdicts[0].kind == "bend"

    def test_technique_candidates_list_independent(self):
        """Each PerformanceEvent has its own technique_candidates list."""
        ev1 = PerformanceEvent(onset_ms=0.0)
        ev2 = PerformanceEvent(onset_ms=100.0)
        ev1.upsert_technique_candidate("bend", 0.8)
        assert len(ev1.technique_candidates) == 1
        assert len(ev2.technique_candidates) == 0

    def test_default_factories_produce_fresh_mutable_objects(self):
        """List and dict defaults are fresh per instance, not shared."""
        ev1 = PerformanceEvent(onset_ms=0.0)
        ev2 = PerformanceEvent(onset_ms=100.0)
        ev1.onset_features["key"] = "val"
        ev1.f0_curve.append((0.0, 440.0, 0.0))
        assert ev2.onset_features == {}
        assert ev2.f0_curve == []
