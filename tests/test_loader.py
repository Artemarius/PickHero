"""Tests for pickhero.tabs.loader module."""

import io
import tempfile
import zipfile
from pathlib import Path

import pytest

from pickhero.audio.note_utils import STANDARD_TUNING
from pickhero.tabs.loader import (
    TempoMap,
    _extract_techniques,
    is_guitar_track,
    list_tracks,
    load_gp_file,
)

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
        assert len(guitar_tracks) == 3
        assert guitar_tracks[0]["name"] == "Rhythm Guitar"
        assert guitar_tracks[1]["name"] == "Solo Guitar"
        assert guitar_tracks[2]["name"] == "Bass"

    def test_canon(self):
        tracks = list_tracks(FIXTURES / "canon.gp5")
        assert len(tracks) == 9
        guitar_tracks = [t for t in tracks if t["is_guitar"]]
        assert len(guitar_tracks) == 5
        # String ensemble (inst=49) and synth strings (inst=51) excluded
        non_guitar_names = {t["name"] for t in tracks if not t["is_guitar"]}
        assert "Low Bassy Sound" in non_guitar_names
        assert "High Soundy Thing" in non_guitar_names

    def test_percussion_excluded(self):
        tracks = list_tracks(FIXTURES / "Demo_v5.gp5")
        perc = [t for t in tracks if t["is_percussion"]]
        assert len(perc) == 1
        assert perc[0]["is_guitar"] is False


class TestIsGuitarTrack:
    """Unit tests for the is_guitar_track heuristic."""

    def _make_track(self, strings: int, instrument: int, percussion: bool = False):
        track = type("Track", (), {})()
        track.strings = [type("String", (), {"value": 64 - i})() for i in range(strings)]
        track.channel = type("Channel", (), {})()
        track.channel.instrument = instrument
        track.channel.isPercussionChannel = percussion
        return track

    def test_standard_six_string_guitar(self):
        assert is_guitar_track(self._make_track(6, 29)) is True

    def test_seven_string_guitar_accepted(self):
        assert is_guitar_track(self._make_track(7, 30)) is True

    def test_four_string_bass_accepted(self):
        assert is_guitar_track(self._make_track(4, 34)) is True

    def test_eight_string_guitar_accepted(self):
        assert is_guitar_track(self._make_track(8, 31)) is True

    def test_three_string_rejected(self):
        assert is_guitar_track(self._make_track(3, 29)) is False

    def test_nine_string_rejected(self):
        assert is_guitar_track(self._make_track(9, 29)) is False

    def test_percussion_rejected(self):
        assert is_guitar_track(self._make_track(6, 30, percussion=True)) is False

    def test_piano_rejected(self):
        assert is_guitar_track(self._make_track(6, 1)) is False


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
        """Slides.gp5 should load notes with a slide technique."""
        tl = load_gp_file(FIXTURES / "Slides.gp5")
        tech_notes = [n for n in tl.notes if n.techniques]
        assert len(tech_notes) > 0, "Slides.gp5 should have notes with techniques"
        # All technique notes in Slides.gp5 are slides
        for note in tech_notes:
            kinds = [t.kind for t in note.techniques]
            assert "slide" in kinds, (
                f"Expected 'slide' in techniques, got {note.techniques}"
            )

    def test_notes_gp5_no_articulation(self):
        """notes.gp5 should have no technique effects (basic notes)."""
        tl = load_gp_file(FIXTURES / "notes.gp5")
        tech_notes = [n for n in tl.notes if n.techniques]
        assert len(tech_notes) == 0, f"notes.gp5 should have no techniques, got {len(tech_notes)}"

class TestExtractTechniques:
    """Test the _extract_techniques helper function."""

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
        # note.type.value defaults to 1 (normal)
        note.type = MagicMock()
        note.type.value = 1
        # Apply overrides
        for k, v in effect_kwargs.items():
            setattr(note.effect, k, v)
        return note

    def test_normal_note_returns_empty(self):
        note = self._make_note()
        assert _extract_techniques(note) == ()

    def test_hammer_on(self):
        note = self._make_note(hammer=True)
        specs = _extract_techniques(note)
        assert len(specs) == 1
        assert specs[0].kind == "hammer_on"
        assert specs[0].tied_to_previous is True

    def test_palm_mute(self):
        note = self._make_note(palmMute=True)
        assert _extract_techniques(note)[0].kind == "palm_mute"

    def test_vibrato(self):
        note = self._make_note(vibrato=True)
        assert _extract_techniques(note)[0].kind == "vibrato"

    def test_bend_whole_step(self):
        """A bend with dest value=4 (1 semitone) → whole-step bend, 100 cents."""
        from unittest.mock import MagicMock
        bend = MagicMock()
        point0 = MagicMock(position=0, value=0)
        point1 = MagicMock(position=6, value=4)
        point2 = MagicMock(position=12, value=4)
        bend.points = [point0, point1, point2]
        bend.semitoneLength = 1
        note = self._make_note(bend=bend)
        specs = _extract_techniques(note)
        assert len(specs) == 1
        assert specs[0].kind == "bend"
        assert specs[0].target_cents == 100.0
        assert specs[0].subtype == "whole"

    def test_slide(self):
        from unittest.mock import MagicMock
        slide_type = MagicMock()
        slide_type.value = 1  # shiftSlideTo
        note = self._make_note(slides=[slide_type])
        specs = _extract_techniques(note)
        assert len(specs) == 1
        assert specs[0].kind == "slide"
        assert specs[0].subtype == "shift"

    def test_harmonic_natural(self):
        from unittest.mock import MagicMock
        harmonic = MagicMock()
        harmonic.type = MagicMock()
        harmonic.type.value = 1  # natural
        note = self._make_note(harmonic=harmonic)
        specs = _extract_techniques(note)
        assert len(specs) == 1
        assert specs[0].kind == "harmonic"
        assert specs[0].subtype == "natural"

    def test_multiple_techniques_palm_mute_and_bend(self):
        """A note can carry multiple techniques (palm_mute + bend)."""
        from unittest.mock import MagicMock
        bend = MagicMock()
        point0 = MagicMock(position=0, value=0)
        point1 = MagicMock(position=6, value=4)
        bend.points = [point0, point1]
        bend.semitoneLength = 1
        note = self._make_note(palmMute=True, bend=bend)
        specs = _extract_techniques(note)
        kinds = [s.kind for s in specs]
        assert "palm_mute" in kinds
        assert "bend" in kinds

    def test_dead_note(self):
        """A dead note (note.type.value == 3) → dead_note spec."""
        note = self._make_note()
        note.type.value = 3
        specs = _extract_techniques(note)
        assert any(s.kind == "dead_note" for s in specs)


class TestGp7AndGp6:
    """Tests for GP7/GP8 ZIP and GP6 BCFZ/BCFS loaders.

    Fixtures are synthetic; they exercise the XML parsing path that is shared
    by GP6, GP7, and GP8 formats. String numbering matches the GP5 parser:
    string 1 = high E, string 6 = low E.
    """

    def test_gp7_simple_load(self):
        path = FIXTURES / "simple.gp7"
        timeline = load_gp_file(path)
        assert timeline.metadata.title == "GP7 Test"
        # Fixture has 2 measures of 4 beats each
        assert len(timeline.notes) == 8
        # Notes should be parsed with the correct strings and frets.
        # The first note is fret 0 on gp7_string 5 (low E) -> our string 6.
        notes = sorted(timeline.notes, key=lambda n: n.timestamp_ms)
        assert notes[0].fret == 0
        assert notes[0].string == 6  # low E
        # The second note is fret 2 on gp7_string 4 (A) -> our string 5.
        assert notes[1].fret == 2
        assert notes[1].string == 5

    def test_gp7_track_selection(self):
        path = FIXTURES / "simple.gp7"
        timeline = load_gp_file(path, track_index=0)
        assert timeline.metadata.track_name == "Electric Guitar"
        assert timeline.metadata.track_index == 0

        timeline_bass = load_gp_file(path, track_index=1)
        assert timeline_bass.metadata.track_name == "Bass"
        assert timeline_bass.metadata.track_index == 1

    def test_gp7_list_tracks(self):
        tracks = list_tracks(FIXTURES / "simple.gp7")
        assert len(tracks) == 2
        assert tracks[0]["name"] == "Electric Guitar"
        assert tracks[0]["is_guitar"] is True
        assert tracks[1]["name"] == "Bass"

    def test_gp6_simple_load(self):
        path = FIXTURES / "simple.gpx"
        timeline = load_gp_file(path)
        assert timeline.metadata.title == "GP6 Test"
        assert timeline.metadata.track_name == "Acoustic Guitar"
        assert len(timeline.notes) == 1
        note = timeline.notes[0]
        assert note.fret == 3
        # gp7_string 5 (low E) -> our string 6
        assert note.string == 6

    def test_gp7_tempo_change(self):
        """A tempo automation in bar 1 must change the timing of later notes."""
        import io, zipfile

        gpif = '''<?xml version="1.0" encoding="UTF-8"?>
<GPIF version="1.0">
  <Score><Title>Tempo</Title><Artist>T</Artist><Album>A</Album></Score>
  <Rhythms>
    <Rhythm id="0"><NoteValue>Quarter</NoteValue></Rhythm>
  </Rhythms>
  <Notes>
    <Note id="0"><Properties><Property name="Fret"><Fret>0</Fret></Property><Property name="String"><String>5</String></Property></Properties></Note>
  </Notes>
  <Beats><Beat id="0"><Rhythm ref="0"/><Notes>0</Notes></Beat></Beats>
  <Voices><Voice id="0"><Beats>0</Beats></Voice></Voices>
  <Bars><Bar id="0"><Voices>0</Voices></Bar><Bar id="1"><Voices>0</Voices></Bar></Bars>
  <Tracks>
    <Track id="0">
      <Name>Guitar</Name>
      <Properties><Property name="Tuning"><Pitches>64 59 55 50 45 40</Pitches></Property></Properties>
      <MIDI><Program>27</Program></MIDI>
    </Track>
  </Tracks>
  <MasterTrack>
    <Automations>
      <Automation><Type>Tempo</Type><Bar>1</Bar><Value>240 2</Value></Automation>
    </Automations>
  </MasterTrack>
  <MasterBars>
    <MasterBar><Time>4/4</Time><Bars>0</Bars></MasterBar>
    <MasterBar><Time>4/4</Time><Bars>0</Bars></MasterBar>
  </MasterBars>
</GPIF>'''
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("Content/score.gpif", gpif)

        with tempfile.NamedTemporaryFile(suffix=".gp7", delete=False) as f:
            f.write(buf.getvalue())
            tmp_path = Path(f.name)
        try:
            timeline = load_gp_file(tmp_path)
            # Measure 0 at 120 BPM: 4 quarter notes * 500ms = 2000ms
            # Measure 1 at 240 BPM: 4 quarter notes * 250ms = 1000ms
            # The single note in measure 1 must start at 2000ms.
            notes = sorted(timeline.notes, key=lambda n: n.timestamp_ms)
            assert notes[0].timestamp_ms == pytest.approx(0.0)
            assert notes[1].timestamp_ms == pytest.approx(2000.0)
        finally:
            tmp_path.unlink(missing_ok=True)
