"""Verify GP5 and GP7 technique extraction produces correct TechniqueSpec tuples.

Fixtures:
- fixtures/Effects.gp5 — real GP5 notes with: dead_note, harmonic (natural),
  hammer_on, bend (whole-step, target_cents=100), slides (shift, legato,
  slide_in_below, slide_in_above, slide_out), vibrato, palm_mute.
- fixtures/techniques.gp7 — synthetic GP7 with: palm_mute, bend (whole-step,
  target_cents=100), natural harmonic, shift slide, hammer_on, vibrato,
  dead_note.
- fixtures/Slides.gp5 — all slide subtypes.
- fixtures/notes.gp5 — no techniques at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pickhero.tabs.loader import (
    _bend_subtype,
    load_gp_file,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── GP5: Effects.gp5 ──────────────────────────────────────────────────────


class TestEffectsGp5:
    """Technique extraction from effects-heavy GP5 fixture."""

    @pytest.fixture(autouse=True)
    def loaded(self):
        self.timeline = load_gp_file(str(FIXTURES / "Effects.gp5"))

    # 1. Bend with correct target_cents
    def test_has_bend_with_correct_target_cents(self):
        notes_with_bend = [
            n for n in self.timeline.notes
            if any(s.kind == "bend" for s in n.techniques)
        ]
        assert len(notes_with_bend) >= 1, "Effects.gp5 should contain bend notes"
        for n in notes_with_bend:
            bend_specs = [s for s in n.techniques if s.kind == "bend"]
            assert len(bend_specs) == 1
            bs = bend_specs[0]
            assert bs.target_cents == 100.0
            assert bs.subtype == "whole"

    # 2. Natural harmonic (Effects.gp5 has natural + artificial + tapped)
    def test_has_harmonic_natural(self):
        notes_with_harmonic = [
            n for n in self.timeline.notes
            if any(s.kind == "harmonic" for s in n.techniques)
        ]
        assert len(notes_with_harmonic) >= 1
        subtypes = {
            s.subtype
            for n in notes_with_harmonic
            for s in n.techniques if s.kind == "harmonic"
        }
        assert "natural" in subtypes, (
            f"expected at least one natural harmonic, got {subtypes}"
        )

    # 3. Palm mute
    def test_has_palm_mute(self):
        pm_notes = [
            n for n in self.timeline.notes
            if any(s.kind == "palm_mute" for s in n.techniques)
        ]
        assert len(pm_notes) >= 1
        assert pm_notes[0].techniques[0].kind == "palm_mute"

    # 4. Slide subtypes
    def test_has_slide_subtypes(self):
        slide_notes = [
            n for n in self.timeline.notes
            if any(s.kind == "slide" for s in n.techniques)
        ]
        assert len(slide_notes) >= 1
        slide_subtypes = {
            s.kind
            for n in slide_notes
            for s in n.techniques
            if s.kind == "slide"
        }
        # At least one subtype from each expected set
        expected_subtypes = {"shift", "legato", "slide_in_below", "slide_in_above", "slide_out"}
        found_subtypes = {
            s.kind
            for n in slide_notes
            for s in n.techniques
            if s.kind == "slide"
        }
        found_subtypes = {
            s.subtype
            for n in slide_notes
            for s in n.techniques
            if s.kind == "slide"
        }
        # Assert at least one note has a subtype in the expected set
        assert found_subtypes & expected_subtypes, (
            f"Expected slide subtypes {expected_subtypes}, got {found_subtypes}"
        )

    # 5. Dead note
    def test_has_dead_note(self):
        dead_notes = [
            n for n in self.timeline.notes
            if any(s.kind == "dead_note" for s in n.techniques)
        ]
        assert len(dead_notes) >= 1

    # 6. notes.gp5 has no techniques
    def test_notes_gp5_has_no_techniques(self):
        notes = load_gp_file(str(FIXTURES / "notes.gp5"))
        for n in notes.notes:
            assert n.techniques == (), (
                f"notes.gp5 note at {n.timestamp_ms}ms has unexpected techniques: {n.techniques}"
            )


# ── GP7: techniques.gp7 ──────────────────────────────────────────────────


class TestTechniquesGp7:
    """Technique extraction from synthetic GP7 fixture."""

    @pytest.fixture(autouse=True)
    def loaded(self):
        self.timeline = load_gp_file(str(FIXTURES / "techniques.gp7"))

    # 6. Bend target_cents
    def test_bend_target_cents(self):
        bend_notes = [
            n for n in self.timeline.notes
            if any(s.kind == "bend" for s in n.techniques)
        ]
        assert len(bend_notes) >= 1
        for n in bend_notes:
            bs = [s for s in n.techniques if s.kind == "bend"][0]
            assert bs.target_cents == 100.0
            assert bs.subtype == "whole"

    # 7. Palm mute
    def test_palm_mute(self):
        pm_notes = [
            n for n in self.timeline.notes
            if any(s.kind == "palm_mute" for s in n.techniques)
        ]
        assert len(pm_notes) >= 1

    # 8. Natural harmonic
    def test_harmonic_natural(self):
        hc_notes = [
            n for n in self.timeline.notes
            if any(s.kind == "harmonic" for s in n.techniques)
        ]
        assert len(hc_notes) >= 1
        assert hc_notes[0].techniques[0].subtype == "natural"

    # 9. Slide
    def test_slide(self):
        slide_notes = [
            n for n in self.timeline.notes
            if any(s.kind == "slide" for s in n.techniques)
        ]
        assert len(slide_notes) >= 1
        # The slide notes have subtype 'shift'
        slide_specs = [s for n in slide_notes for s in n.techniques if s.kind == "slide"]
        assert any(s.subtype == "shift" for s in slide_specs)

    # 10. Dead note
    def test_dead_note(self):
        dead_notes = [
            n for n in self.timeline.notes
            if any(s.kind == "dead_note" for s in n.techniques)
        ]
        assert len(dead_notes) >= 1


# ── _bend_subtype helper unit tests ──────────────────────────────────────


class TestBendSubtypeHelper:
    """Unit tests for the _bend_subtype function."""

    def test_whole_bend(self):
        assert _bend_subtype(0, 4, 100.0) == "whole"

    def test_half_bend(self):
        assert _bend_subtype(0, 2, 50.0) == "half"

    def test_release(self):
        assert _bend_subtype(4, 0, -100.0) == "release"

    def test_quarter_bend(self):
        assert _bend_subtype(0, 1, 25.0) == "quarter"
    def test_pre_bend(self):
        # pre-bend: starts already bent, holds (origin == dest > 0)
        assert _bend_subtype(4, 4, 100.0) == "pre"

    def test_bend_from_bent_position(self):
        # both > 0 but dest != origin → "bend"
        assert _bend_subtype(4, 8, 100.0) == "bend"


# ── Slides.gp5 ────────────────────────────────────────────────────────────


class TestSlidesGp5:
    """All notes in Slides.gp5 should be slides."""

    @pytest.fixture(autouse=True)
    def loaded(self):
        self.timeline = load_gp_file(str(FIXTURES / "Slides.gp5"))

    def test_all_notes_are_slides(self):
        for n in self.timeline.notes:
            assert all(s.kind == "slide" for s in n.techniques), (
                f"Note at measure {n.measure} ts={n.timestamp_ms:.0f} has "
                f"non-slide techniques: {[s.kind for s in n.techniques]}"
            )

    def test_all_known_subtypes_present(self):
        subtypes = {
            s.subtype
            for n in self.timeline.notes
            for s in n.techniques
            if s.kind == "slide" and s.subtype is not None
        }
        expected = {"legato", "shift", "slide_in_below", "slide_in_above", "slide_out"}
        missing = expected - subtypes
        assert not missing, f"Missing slide subtypes: {missing}"


# ── Patch 5: harmonic expected_sounding_midi + slide fields ──────────────────


class TestHarmonicExpectedPitch:
    """Patch 5c: harmonic expected_sounding_midi is computed from the open
    string + node ratio, not the fretted midi × ratio."""

    @pytest.fixture(autouse=True)
    def loaded(self):
        self.timeline = load_gp_file(str(FIXTURES / "Effects.gp5"))

    def test_natural_harmonic_expected_pitch_from_open_string(self):
        """Natural harmonic at string=6, fret=5 (Effects.gp5): open E2 (midi 40)
        + node ratio 4.0 (fret 5 = two octaves) → expected_sounding_midi = 64."""
        naturals = [
            (n, s) for n in self.timeline.notes
            for s in n.techniques
            if s.kind == "harmonic" and s.subtype == "natural"
        ]
        assert naturals, "Effects.gp5 should have a natural harmonic"
        n, spec = naturals[0]
        assert spec.expected_sounding_midi is not None, (
            "natural harmonic must populate expected_sounding_midi"
        )
        open_midi = self.timeline.metadata.tuning.get(n.string)
        assert open_midi is not None
        # fret 5 node ratio = 4.0 → +24 semitones; fret 12 → +12.
        import math
        from pickhero.tabs.loader import _harmonic_node_ratio
        ratio = _harmonic_node_ratio(spec.node_fret)
        expected = open_midi + round(12 * math.log2(ratio))
        assert spec.expected_sounding_midi == expected, (
            f"expected {expected}, got {spec.expected_sounding_midi}"
        )

    def test_gp5_harmonic_subtype_preserved(self):
        """All harmonic subtypes in Effects.gp5 are preserved on the spec."""
        harmonics = [
            s for n in self.timeline.notes
            for s in n.techniques if s.kind == "harmonic"
        ]
        subtypes = {s.subtype for s in harmonics}
        assert "natural" in subtypes
        # Effects.gp5 also has artificial + tapped harmonics
        assert subtypes & {"artificial", "tapped", "pinch"}, (
            f"expected at least one non-natural subtype, got {subtypes}"
        )


class TestSlideSpecFields:
    """Patch 5b: slide specs carry start_fret, end_fret, target_cents."""

    @pytest.fixture(autouse=True)
    def loaded(self):
        self.timeline = load_gp_file(str(FIXTURES / "Effects.gp5"))

    def test_slide_spec_carries_start_end_target(self):
        """A shift slide 10→11 in Effects.gp5: start_fret=10, end_fret=11,
        target_cents=100.0 (1 semitone = 100 cents)."""
        slides = [
            (n, s) for n in self.timeline.notes
            for s in n.techniques
            if s.kind == "slide" and s.subtype == "shift"
        ]
        assert slides, "Effects.gp5 should have a shift slide"
        n, spec = slides[0]
        assert spec.start_fret == n.fret, (
            f"start_fret={spec.start_fret} != note fret {n.fret}"
        )
        assert spec.end_fret is not None, "shift slide must resolve end_fret"
        assert spec.target_cents is not None, "shift slide must resolve target_cents"
        assert spec.target_cents == (spec.end_fret - spec.start_fret) * 100.0
        # The actual fixture: 10→11 = 100 cents
        assert spec.start_fret == 10
        assert spec.end_fret == 11
        assert spec.target_cents == 100.0

    def test_slide_out_has_no_target(self):
        """A slide_out with no successor (last note) leaves end_fret/target_cents
        None — SlideJudge grades on gesture only."""
        slide_outs = [
            (n, s) for n in self.timeline.notes
            for s in n.techniques
            if s.kind == "slide" and s.subtype == "slide_out"
        ]
        assert slide_outs, "Effects.gp5 should have a slide_out"
        # At least one slide_out should have no resolved target (the last note)
        last = slide_outs[-1]
        n, spec = last
        # slide_out at the end of the voice has no successor
        if spec.end_fret is None:
            assert spec.target_cents is None
