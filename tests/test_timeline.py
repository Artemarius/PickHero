"""Tests for pickhero.tabs.timeline module."""

import pytest

from pickhero.tabs.timeline import NoteEvent, SongMetadata, Timeline


class TestNoteEvent:
    def test_creation(self):
        note = NoteEvent(
            timestamp_ms=1000.0, duration_ms=500.0, midi_note=64, string=1, fret=0
        )
        assert note.timestamp_ms == 1000.0
        assert note.duration_ms == 500.0
        assert note.midi_note == 64
        assert note.string == 1
        assert note.fret == 0

    def test_end_ms(self):
        note = NoteEvent(
            timestamp_ms=1000.0, duration_ms=500.0, midi_note=64, string=1, fret=0
        )
        assert note.end_ms == 1500.0

    def test_immutable(self):
        note = NoteEvent(
            timestamp_ms=1000.0, duration_ms=500.0, midi_note=64, string=1, fret=0
        )
        with pytest.raises(AttributeError):
            note.timestamp_ms = 2000.0

    def test_invalid_timestamp(self):
        with pytest.raises(ValueError, match="timestamp_ms"):
            NoteEvent(
                timestamp_ms=-1.0, duration_ms=500.0, midi_note=64, string=1, fret=0
            )

    def test_invalid_duration(self):
        with pytest.raises(ValueError, match="duration_ms"):
            NoteEvent(
                timestamp_ms=0.0, duration_ms=-1.0, midi_note=64, string=1, fret=0
            )

    def test_invalid_midi_note_low(self):
        with pytest.raises(ValueError, match="midi_note"):
            NoteEvent(
                timestamp_ms=0.0, duration_ms=500.0, midi_note=-1, string=1, fret=0
            )

    def test_invalid_midi_note_high(self):
        with pytest.raises(ValueError, match="midi_note"):
            NoteEvent(
                timestamp_ms=0.0, duration_ms=500.0, midi_note=128, string=1, fret=0
            )

    def test_invalid_string_low(self):
        with pytest.raises(ValueError, match="string"):
            NoteEvent(
                timestamp_ms=0.0, duration_ms=500.0, midi_note=64, string=0, fret=0
            )

    def test_invalid_string_high(self):
        with pytest.raises(ValueError, match="string"):
            NoteEvent(
                timestamp_ms=0.0, duration_ms=500.0, midi_note=64, string=13, fret=0
            )

    def test_extended_string_allowed(self):
        note = NoteEvent(
            timestamp_ms=0.0, duration_ms=500.0, midi_note=35, string=7, fret=0
        )
        assert note.string == 7

    def test_zero_duration_allowed(self):
        note = NoteEvent(
            timestamp_ms=0.0, duration_ms=0.0, midi_note=64, string=1, fret=0
        )
        assert note.end_ms == 0.0


class TestSongMetadata:
    def test_defaults(self):
        meta = SongMetadata()
        assert meta.title == ""
        assert meta.artist == ""
        assert meta.tempo == 120
        assert meta.tuning == {}
        assert meta.num_strings == 6

    def test_custom_values(self):
        tuning = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40, 7: 35}
        meta = SongMetadata(
            title="Test Song", artist="Artist", tempo=140, tuning=tuning,
            num_strings=7,
        )
        assert meta.title == "Test Song"
        assert meta.tempo == 140
        assert meta.tuning[1] == 64
        assert meta.num_strings == 7


class TestTimeline:
    def _make_note(self, timestamp_ms, duration_ms=500.0, string=1, fret=0):
        return NoteEvent(
            timestamp_ms=timestamp_ms,
            duration_ms=duration_ms,
            midi_note=64,
            string=string,
            fret=fret,
        )

    def test_empty_timeline(self):
        tl = Timeline([])
        assert len(tl) == 0
        assert tl.duration_ms == 0.0
        assert tl.get_notes_in_range(0, 1000) == []
        assert tl.get_active_notes_at_time(500) == []

    def test_sorting(self):
        notes = [self._make_note(2000), self._make_note(500), self._make_note(1000)]
        tl = Timeline(notes)
        assert len(tl) == 3
        assert tl.notes[0].timestamp_ms == 500.0
        assert tl.notes[1].timestamp_ms == 1000.0
        assert tl.notes[2].timestamp_ms == 2000.0

    def test_duration(self):
        notes = [
            self._make_note(0, duration_ms=200),
            self._make_note(1000, duration_ms=500),
        ]
        tl = Timeline(notes)
        assert tl.duration_ms == 1500.0  # 1000 + 500

    def test_get_notes_in_range(self):
        notes = [
            self._make_note(100),
            self._make_note(500),
            self._make_note(1000),
            self._make_note(1500),
            self._make_note(2000),
        ]
        tl = Timeline(notes)
        # Range [500, 1500) — includes 500 and 1000, not 1500
        result = tl.get_notes_in_range(500, 1500)
        assert len(result) == 2
        assert result[0].timestamp_ms == 500.0
        assert result[1].timestamp_ms == 1000.0

    def test_get_notes_in_range_empty(self):
        notes = [self._make_note(100), self._make_note(2000)]
        tl = Timeline(notes)
        result = tl.get_notes_in_range(500, 1000)
        assert result == []

    def test_get_active_notes_at_time(self):
        # Note from 0-500ms, another from 800-1300ms
        notes = [
            self._make_note(0, duration_ms=500),
            self._make_note(800, duration_ms=500),
        ]
        tl = Timeline(notes)

        # At time 250, window 100: only first note active
        result = tl.get_active_notes_at_time(250, window_ms=100)
        assert len(result) == 1
        assert result[0].timestamp_ms == 0.0

        # At time 450, window 100: first note ends at 500, window is [350, 550]
        result = tl.get_active_notes_at_time(450, window_ms=100)
        assert len(result) == 1

        # At time 700, window 150: window is [550, 850], second note starts at 800
        result = tl.get_active_notes_at_time(700, window_ms=150)
        assert len(result) == 1
        assert result[0].timestamp_ms == 800.0

    def test_get_active_notes_overlap(self):
        # Two overlapping notes
        notes = [
            self._make_note(0, duration_ms=1000),
            self._make_note(500, duration_ms=1000),
        ]
        tl = Timeline(notes)
        result = tl.get_active_notes_at_time(750, window_ms=0)
        assert len(result) == 2

    def test_seek_and_get_next(self):
        notes = [
            self._make_note(100),
            self._make_note(200),
            self._make_note(300),
            self._make_note(400),
        ]
        tl = Timeline(notes)

        tl.seek(0)
        result = tl.get_next_notes(2)
        assert len(result) == 2
        assert result[0].timestamp_ms == 100.0
        assert result[1].timestamp_ms == 200.0

        result = tl.get_next_notes(1)
        assert len(result) == 1
        assert result[0].timestamp_ms == 300.0

    def test_seek_past_end(self):
        notes = [self._make_note(100)]
        tl = Timeline(notes)
        tl.seek(9999)
        result = tl.get_next_notes(5)
        assert result == []

    def test_seek_to_middle(self):
        notes = [self._make_note(100), self._make_note(500), self._make_note(1000)]
        tl = Timeline(notes)
        tl.seek(500)
        result = tl.get_next_notes(10)
        assert len(result) == 2
        assert result[0].timestamp_ms == 500.0

    def test_repr(self):
        meta = SongMetadata(title="My Song")
        tl = Timeline([self._make_note(0)], metadata=meta)
        assert "My Song" in repr(tl)
        assert "1 notes" in repr(tl)

    def test_len(self):
        notes = [self._make_note(i * 100) for i in range(10)]
        tl = Timeline(notes)
        assert len(tl) == 10

    def test_active_cursor_forward_playback(self):
        """Cursor should advance during forward playback, not scan from 0."""
        notes = [self._make_note(i * 100, duration_ms=50) for i in range(100)]
        tl = Timeline(notes)
        # Query forward — should find the right notes
        result = tl.get_active_notes_at_time(500, window_ms=50)
        assert len(result) >= 1
        # Cursor should have advanced past early notes
        assert tl._active_cursor > 0

    def test_active_cursor_backward_seek_resets(self):
        """Backward query should reset cursor and still return correct results."""
        notes = [
            self._make_note(100, duration_ms=50),
            self._make_note(500, duration_ms=50),
            self._make_note(1000, duration_ms=50),
        ]
        tl = Timeline(notes)
        # Forward: query at 1000
        tl.get_active_notes_at_time(1000, window_ms=50)
        assert tl._active_cursor > 0
        # Backward: query at 100 — should reset and still work
        result = tl.get_active_notes_at_time(100, window_ms=50)
        assert len(result) >= 1
        assert notes[0] in result or result[0].timestamp_ms == 100

    def test_active_cursor_sustained_notes(self):
        """Long sustained notes that started before the window are still returned."""
        notes = [
            self._make_note(0, duration_ms=2000, string=1),  # sustained
            self._make_note(500, duration_ms=50, string=2),
        ]
        tl = Timeline(notes)
        # Query at 600 — sustained note (0-2000) should still be active
        result = tl.get_active_notes_at_time(600, window_ms=50)
        ts = [n.timestamp_ms for n in result]
        assert 0 in ts  # sustained note still active