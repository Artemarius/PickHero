"""End-to-end smoke tests for the parse -> timeline -> match pipeline.

These tests load real tab fixtures (GP5 and GP7/GP6) and verify that the
matcher correctly scores synthetic detected notes. They don't require audio
hardware or a running PyGame display.
"""

from pathlib import Path

import pytest

from pickhero.audio.detector import DetectedNote
from pickhero.audio.input import TimestampedNote
from pickhero.matcher import MatchType, NoteMatcher
from pickhero.tabs.loader import load_gp_file


FIXTURES = Path(__file__).parent / "fixtures"


def _detected(midi_note: int, timestamp_ms: float) -> TimestampedNote:
    """Build a synthetic detected note with an onset."""
    return TimestampedNote(
        note=DetectedNote(
            midi_note=midi_note,
            frequency=440.0,
            confidence=0.95,
            name="N",
            is_onset=True,
        ),
        timestamp_ms=timestamp_ms,
    )


def _matcher_for(path: Path) -> NoteMatcher:
    """Load a tab file and return a fresh matcher for its guitar track."""
    timeline = load_gp_file(path)
    return NoteMatcher(timeline, timing_window_ms=100.0)


class TestGp5Integration:
    def test_canon_rock_first_note_hit(self):
        """Loading canon.gp5 and playing the first note registers a HIT."""
        matcher = _matcher_for(FIXTURES / "canon.gp5")

        # First guitar note in the fixture: MIDI 78 at 22000ms.
        detected = [_detected(78, 22_000.0)]
        matcher.process_detected_notes(detected, 22_050.0)

        assert matcher.hits >= 1


class TestGp7Gp6Integration:
    def test_gp7_first_note_hit(self):
        """Loading simple.gp7 and playing the first note registers a HIT."""
        matcher = _matcher_for(FIXTURES / "simple.gp7")

        # First note in simple.gp7: MIDI 40 (low E open) at 0ms.
        detected = [_detected(40, 0.0)]
        matcher.process_detected_notes(detected, 50.0)

        assert matcher.hits >= 1

    def test_gp6_first_note_hit(self):
        """Loading simple.gpx and playing the first note registers a HIT."""
        matcher = _matcher_for(FIXTURES / "simple.gpx")

        # First note in simple.gpx: fret 3 on gp7_string 5 (low E) -> MIDI 43.
        detected = [_detected(43, 0.0)]
        matcher.process_detected_notes(detected, 50.0)

        assert matcher.hits >= 1

    def test_gp7_tempo_change_affects_timing(self):
        """A tab with a tempo change must place later notes at adjusted times."""
        import io
        import tempfile
        import zipfile

        gpif = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<GPIF version="1.0">'
            '<Score><Title>Tempo</Title><Artist>T</Artist><Album>A</Album></Score>'
            '<Rhythms><Rhythm id="0"><NoteValue>Quarter</NoteValue></Rhythm></Rhythms>'
            '<Notes>'
            '<Note id="0"><Properties><Property name="Fret"><Fret>0</Fret></Property><Property name="String"><String>5</String></Property></Properties></Note>'
            '<Note id="1"><Properties><Property name="Fret"><Fret>2</Fret></Property><Property name="String"><String>4</String></Property></Properties></Note>'
            '</Notes>'
            '<Beats>'
            '<Beat id="0"><Rhythm ref="0"/><Notes>0</Notes></Beat>'
            '<Beat id="1"><Rhythm ref="0"/><Notes>1</Notes></Beat>'
            '</Beats>'
            '<Voices><Voice id="0"><Beats>0 1</Beats></Voice></Voices>'
            '<Bars><Bar id="0"><Voices>0</Voices></Bar><Bar id="1"><Voices>0</Voices></Bar></Bars>'
            '<Tracks>'
            '<Track id="0"><Name>Guitar</Name>'
            '<Properties><Property name="Tuning"><Pitches>64 59 55 50 45 40</Pitches></Property></Properties>'
            '<MIDI><Program>27</Program></MIDI>'
            '</Track>'
            '</Tracks>'
            '<MasterTrack>'
            '<Automations><Automation><Type>Tempo</Type><Bar>1</Bar><Value>240 2</Value></Automation></Automations>'
            '</MasterTrack>'
            '<MasterBars>'
            '<MasterBar><Time>4/4</Time><Bars>0</Bars></MasterBar>'
            '<MasterBar><Time>4/4</Time><Bars>1</Bars></MasterBar>'
            '</MasterBars>'
            '</GPIF>'
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("Content/score.gpif", gpif)

        with tempfile.NamedTemporaryFile(suffix=".gp7", delete=False) as f:
            f.write(buf.getvalue())
            tmp_path = Path(f.name)

        try:
            matcher = _matcher_for(tmp_path)
            # Measure 0 at 120 BPM: first quarter-note at 0ms, second at 500ms.
            # Measure 1 at 240 BPM: first quarter-note at 2000ms, second at 2250ms.
            matcher.process_detected_notes([_detected(40, 0.0)], 50.0)
            matcher.process_detected_notes([_detected(47, 2_250.0)], 2_300.0)

            assert matcher.hits == 2
        finally:
            tmp_path.unlink(missing_ok=True)
