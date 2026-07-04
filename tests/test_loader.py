"""Tests for pickhero.tabs.loader module."""

from pathlib import Path

import pytest

from pickhero.audio.note_utils import STANDARD_TUNING
from pickhero.tabs.loader import TempoMap, is_guitar_track, list_tracks, load_gp_file, _extract_articulation

FIXTURES = Path(__file__).parent / "fixtures"


class TestTempoMap:
    def test_single_tempo(self):
        tm = TempoMap(120)
        # 960 ticks = 1 quarter note at 120 BPM = 500ms
        assert tm.tick_to_ms(960) == pytest.approx(500.0)
        # 0 ticks = 0ms
        assert tm.tick_to_ms(0) == 0.0

    def test_duration_at_single_tempo(self):
        tm = TempoMap(120)
        # Quarter note (960 ticks) at 120 BPM = 500ms
        assert tm.duration_ticks_to_ms(960, at_tick=0) == pytest.approx(500.0)
        # Half note = 1000ms
        assert tm.duration_ticks_to_ms(1920, at_tick=0) == pytest.approx(1000.0)

    def test_tempo_change(self):
        tm = TempoMap(120)
        # Tempo changes to 240 at tick 960
        tm.add_change(960, 240)

        # At tick 960: all time is at 120 BPM → 500ms
        assert tm.tick_to_ms(960) == pytest.approx(500.0)

        # At tick 1920: 500ms (first quarter at 120) + 250ms (next quarter at 240) = 750ms
        assert tm.tick_to_ms(1920) == pytest.approx(750.0)

    def test_multiple_tempo_changes(self):
        tm = TempoMap(120)
        tm.add_change(960, 240)   # Change at tick 960
        tm.add_change(1920, 60)   # Change at tick 1920

        # tick 0→960 at 120 BPM: 500ms
        # tick 960→1920 at 240 BPM: 250ms
        # tick 1920→2880 at 60 BPM: 1000ms
        assert tm.tick_to_ms(2880) == pytest.approx(1750.0)

    def test_tempo_at_tick(self):
        tm = TempoMap(120)
        tm.add_change(960, 200)
        assert tm.tempo_at_tick(0) == 120
        assert tm.tempo_at_tick(959) == 120
        assert tm.tempo_at_tick(960) == 200
        assert tm.tempo_at_tick(5000) == 200

    def test_duration_uses_local_tempo(self):
        tm = TempoMap(120)
        tm.add_change(960, 240)

        # Before change: 120 BPM
        assert tm.duration_ticks_to_ms(960, at_tick=0) == pytest.approx(500.0)
        # After change: 240 BPM
        assert tm.duration_ticks_to_ms(960, at_tick=960) == pytest.approx(250.0)


class TestListTracks:
    def test_slides(self):
        tracks = list_tracks(FIXTURES / "Slides.gp5")
        assert len(tracks) == 1
        assert tracks[0]["is_guitar"] is True

    def test_notes(self):
        tracks = list_tracks(FIXTURES / "notes.gp5")
        assert len(tracks) == 1
        assert tracks[0]["is_guitar"] is True

    def test_effects(self):
        tracks = list_tracks(FIXTURES / "Effects.gp5")
        assert len(tracks) == 1
        assert tracks[0]["is_guitar"] is True

    def test_tie(self):
        tracks = list_tracks(FIXTURES / "Tie.gp5")
        assert len(tracks) == 1
        assert tracks[0]["is_guitar"] is True

    def test_demo_v5(self):
        tracks = list_tracks(FIXTURES / "Demo_v5.gp5")
        assert len(tracks) == 5
        guitar_tracks = [t for t in tracks if t["is_guitar"]]
        assert len(guitar_tracks) == 2
        assert guitar_tracks[0]["name"] == "Rhythm Guitar"
        assert guitar_tracks[1]["name"] == "Solo Guitar"

    def test_canon(self):
        tracks = list_tracks(FIXTURES / "canon.gp5")
        assert len(tracks) == 9
        guitar_tracks = [t for t in tracks if t["is_guitar"]]
        assert len(guitar_tracks) == 4
        # String ensemble (inst=49) and synth strings (inst=51) excluded
        non_guitar_names = {t["name"] for t in tracks if not t["is_guitar"]}
        assert "Low Bassy Sound" in non_guitar_names
        assert "High Soundy Thing" in non_guitar_names

    def test_percussion_excluded(self):
        tracks = list_tracks(FIXTURES / "Demo_v5.gp5")
        perc = [t for t in tracks if t["is_percussion"]]
        assert len(perc) == 1
        assert perc[0]["is_guitar"] is False


class TestLoadGPFile:
    def test_slides(self):
        tl = load_gp_file(FIXTURES / "Slides.gp5")
        assert len(tl) == 12
        assert tl.metadata.tempo == 120
        assert tl.metadata.tuning == STANDARD_TUNING

    def test_notes(self):
        tl = load_gp_file(FIXTURES / "notes.gp5")
        assert len(tl) == 28
        assert tl.metadata.tempo == 120

    def test_effects(self):
        tl = load_gp_file(FIXTURES / "Effects.gp5")
        assert len(tl) == 46
        assert tl.metadata.tempo == 120

    def test_tie(self):
        tl = load_gp_file(FIXTURES / "Tie.gp5")
        assert len(tl) == 11
        assert tl.metadata.tempo == 120

    def test_demo_v5_track0(self):
        tl = load_gp_file(FIXTURES / "Demo_v5.gp5", track_index=0)
        assert len(tl) == 729
        assert tl.metadata.tempo == 165
        assert tl.metadata.track_name == "Rhythm Guitar"

    def test_canon_track0(self):
        tl = load_gp_file(FIXTURES / "canon.gp5", track_index=0)
        assert len(tl) == 1489
        assert tl.metadata.tempo == 90
        assert tl.metadata.track_name == "Guitar Player"

    def test_standard_tuning(self):
        for fname in ["Slides.gp5", "notes.gp5", "Effects.gp5", "Tie.gp5"]:
            tl = load_gp_file(FIXTURES / fname)
            assert tl.metadata.tuning == STANDARD_TUNING, f"{fname} tuning mismatch"

    def test_timestamps_monotonic(self):
        for fname in [
            "Slides.gp5", "notes.gp5", "Effects.gp5", "Tie.gp5",
            "Demo_v5.gp5", "canon.gp5",
        ]:
            tl = load_gp_file(FIXTURES / fname)
            timestamps = [n.timestamp_ms for n in tl.notes]
            for i in range(1, len(timestamps)):
                assert timestamps[i] >= timestamps[i - 1], (
                    f"{fname}: timestamp[{i}]={timestamps[i]} < [{i-1}]={timestamps[i-1]}"
                )

    def test_canon_tempo_changes_affect_timestamps(self):
        """Verify that tempo changes produce different ms values than constant tempo."""
        tl = load_gp_file(FIXTURES / "canon.gp5", track_index=0)
        last_note = tl.notes[-1]

        # At constant 90 BPM, the last note would be much later
        # With tempo changes (faster sections), it's compressed
        # Constant 90 BPM for the same tick would give ~598s
        # With tempo changes it's ~322s
        assert last_note.timestamp_ms < 400_000  # well under constant-tempo value

    def test_auto_select_guitar_track(self):
        # canon.gp5 has guitar track at index 0
        tl = load_gp_file(FIXTURES / "canon.gp5")
        assert tl.metadata.track_name == "Guitar Player"

    def test_explicit_track_index(self):
        tl = load_gp_file(FIXTURES / "Demo_v5.gp5", track_index=1)
        assert tl.metadata.track_name == "Solo Guitar"

    def test_duration_positive(self):
        for fname in ["Slides.gp5", "notes.gp5", "canon.gp5"]:
            tl = load_gp_file(FIXTURES / fname)
            assert tl.duration_ms > 0
            for note in tl.notes:
                assert note.duration_ms > 0, f"{fname}: note with 0 duration"

    def test_slides_gp5_has_articulation(self):
        """Slides.gp5 should load notes with expected_articulation='slide'."""
        tl = load_gp_file(FIXTURES / "Slides.gp5")
        art_notes = [n for n in tl.notes if n.expected_articulation is not None]
        assert len(art_notes) > 0, "Slides.gp5 should have notes with articulations"
        # All articulation notes in Slides.gp5 are slides
        for note in art_notes:
            assert note.expected_articulation == "slide", (
                f"Expected 'slide', got '{note.expected_articulation}'"
            )

    def test_notes_gp5_no_articulation(self):
        """notes.gp5 should have no articulation effects (basic notes)."""
        tl = load_gp_file(FIXTURES / "notes.gp5")
        art_notes = [n for n in tl.notes if n.expected_articulation is not None]
        assert len(art_notes) == 0, f"notes.gp5 should have no articulations, got {len(art_notes)}"


class TestExtractArticulation:
    """Test the _extract_articulation helper function."""

    def _make_note(self, **effect_kwargs):
        """Create a mock note with effect attributes."""
        from unittest.mock import MagicMock
        note = MagicMock()
        note.effect = MagicMock()
        # Set defaults
        note.effect.hammer = False
        note.effect.palmMute = False
        note.effect.vibrato = False
        note.effect.letRing = False
        note.effect.staccato = False
        note.effect.ghostNote = False
        note.effect.accentuatedNote = False
        note.effect.slides = []
        note.effect.bend = None
        note.effect.harmonic = None
        # Apply overrides
        for k, v in effect_kwargs.items():
            setattr(note.effect, k, v)
        return note

    def test_normal_note_returns_none(self):
        note = self._make_note()
        assert _extract_articulation(note) is None

    def test_hammer_on(self):
        note = self._make_note(hammer=True)
        assert _extract_articulation(note) == "hammer_on"

    def test_palm_mute(self):
        note = self._make_note(palmMute=True)
        assert _extract_articulation(note) == "palm_mute"

    def test_vibrato(self):
        note = self._make_note(vibrato=True)
        assert _extract_articulation(note) == "vibrato"

    def test_bend(self):
        note = self._make_note(bend=object())  # non-None = bend present
        assert _extract_articulation(note) == "bend"

    def test_slide(self):
        note = self._make_note(slides=[object()])  # non-empty list
        assert _extract_articulation(note) == "slide"

    def test_harmonic(self):
        note = self._make_note(harmonic=object())  # non-None
        assert _extract_articulation(note) == "harmonic"

    def test_priority_harmonic_over_palm_mute(self):
        """Harmonic takes priority over palm mute."""
        note = self._make_note(harmonic=object(), palmMute=True)
        assert _extract_articulation(note) == "harmonic"

    def test_priority_palm_mute_over_bend(self):
        """Palm mute takes priority over bend."""
        note = self._make_note(palmMute=True, bend=object())
        assert _extract_articulation(note) == "palm_mute"
