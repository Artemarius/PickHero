"""Tests for DatasetImporter ABC and group_simultaneous_notes."""

from __future__ import annotations

from pathlib import Path

import pytest

from pickhero.datasets.base import DatasetImporter
from pickhero.datasets.grouping import group_simultaneous_notes
from pickhero.datasets.schema import ClipEvent, ClipExpectedNote


# ── DatasetImporter (ABC) tests ──────────────────────────────────────────


class TestDatasetImporterABC:
    """DatasetImporter is an abstract base class with one abstract method."""

    def test_cannot_instantiate_directly(self):
        """Instantiation of the ABC itself must raise TypeError."""
        with pytest.raises(TypeError):
            DatasetImporter()  # type: ignore[abstract]

    def test_subclass_with_scan_can_instantiate(self):
        """A concrete subclass implementing scan() must be instantiable."""

        class Importer(DatasetImporter):
            name = "test"

            def scan(self, path: str | Path) -> list[ClipEvent]:
                return []

        importer = Importer()
        assert importer.name == "test"
        assert importer.scan("/some/path") == []

    def test_subclass_missing_scan_raises_typeerror(self):
        """A subclass that does not override the abstract scan() must raise."""

        class BadImporter(DatasetImporter):
            name = "bad"

        with pytest.raises(TypeError):
            BadImporter()  # type: ignore[abstract]

    def test_audio_files_returns_wav_and_flac(self, tmp_path):
        """_audio_files must return sorted .wav and .flac paths recursively."""
        (tmp_path / "sub").mkdir()
        wav = tmp_path / "sub" / "a.wav"
        flac = tmp_path / "sub" / "b.flac"
        wav.touch()
        flac.touch()
        (tmp_path / "readme.txt").touch()

        class Importer(DatasetImporter):
            name = "test"

            def scan(self, path: str | Path) -> list[ClipEvent]:
                return []

        importer = Importer()
        result = importer._audio_files(tmp_path)
        assert result == [wav, flac]

    def test_audio_files_empty_directory(self, tmp_path):
        """_audio_files must return an empty list when no audio files exist."""

        class Importer(DatasetImporter):
            name = "test"

            def scan(self, path: str | Path) -> list[ClipEvent]:
                return []

        importer = Importer()
        assert importer._audio_files(tmp_path) == []

    def test_name_class_attribute_default(self):
        """The name attribute must default to empty string."""
        assert DatasetImporter.name == ""


# ── group_simultaneous_notes tests ───────────────────────────────────────


def _make_note(
    *,
    clip_id: str = "test/note",
    source: str = "test",
    start_s: float = 0.0,
    end_s: float = 1.0,
    midi: int = 60,
    technique: str = "pick",
    confidence: float = 1.0,
    audio_path: str = "/audio/test.wav",
    string: int | None = None,
    fret: int | None = None,
    metadata: dict[str, str] | None = None,
) -> ClipEvent:
    """Build a single-note ClipEvent with convenient defaults."""
    return ClipEvent(
        clip_id=clip_id,
        source=source,
        start_s=start_s,
        end_s=end_s,
        midi=midi,
        technique=technique,
        confidence=confidence,
        audio_path=audio_path,
        string=string,
        fret=fret,
        metadata=metadata or {},
    )
def _make_chord(
    *,
    clip_id: str = "test/chord",
    source: str = "test",
    start_s: float = 0.0,
    end_s: float = 1.0,
    notes: tuple[ClipExpectedNote, ...] = (),
    technique: str = "pick",
    confidence: float = 1.0,
    audio_path: str = "/audio/test.wav",
) -> ClipEvent:
    """Build a chord ClipEvent (midi is None) with convenient defaults."""
    return ClipEvent(
        clip_id=clip_id,
        source=source,
        start_s=start_s,
        end_s=end_s,
        midi=None,
        notes=notes,
        technique=technique,
        confidence=confidence,
        audio_path=audio_path,
    )


class TestGroupSimultaneousNotes:
    """group_simultaneous_notes merges same-onset notes into chords."""

    def test_empty_list(self):
        """An empty input must return an empty list."""
        assert group_simultaneous_notes([]) == []

    def test_single_note(self):
        """A single note must be returned unchanged as one group."""
        note = _make_note()
        result = group_simultaneous_notes([note])
        assert len(result) == 1
        assert result[0] == note
        assert result[0].midi == 60
        assert result[0].notes == ()

    def test_notes_at_same_timestamp_grouped(self):
        """Notes with the same start_s must be merged into one chord event."""
        notes = [
            _make_note(midi=60, string=1),
            _make_note(midi=64, string=2),
            _make_note(midi=67, string=3),
        ]
        result = group_simultaneous_notes(notes)
        assert len(result) == 1
        chord = result[0]
        assert chord.midi is None
        assert len(chord.notes) == 3
        assert chord.midi_notes == frozenset({60, 64, 67})

    def test_notes_at_different_timestamps_separate(self):
        """Notes far apart in time must remain separate single-note events."""
        notes = [
            _make_note(clip_id="a", start_s=0.0, midi=60),
            _make_note(clip_id="b", start_s=1.0, midi=64),
            _make_note(clip_id="c", start_s=2.0, midi=67),
        ]
        result = group_simultaneous_notes(notes)
        assert len(result) == 3
        assert [e.midi for e in result] == [60, 64, 67]

    def test_mixed_timestamps_correct_grouping(self):
        """Multiple groups and isolated singles must each be handled correctly."""
        notes = [
            _make_note(clip_id="a1", start_s=0.0, midi=60, string=1),
            _make_note(clip_id="a2", start_s=0.0, midi=64, string=2),
            _make_note(clip_id="b", start_s=1.5, midi=67),
            _make_note(clip_id="c1", start_s=3.0, midi=69, string=1),
            _make_note(clip_id="c2", start_s=3.0, midi=72, string=2),
            _make_note(clip_id="d", start_s=5.0, midi=76),
        ]
        result = group_simultaneous_notes(notes)

        # Expected: chord at 0.0, single at 1.5, chord at 3.0, single at 5.0
        assert len(result) == 4
        assert result[0].midi is None
        assert result[0].midi_notes == frozenset({60, 64})
        assert result[1].midi == 67
        assert result[1].notes == ()
        assert result[2].midi is None
        assert result[2].midi_notes == frozenset({69, 72})
        assert result[3].midi == 76

    def test_same_onset_different_durations(self):
        """Notes at the same onset but with different end_s must be merged and use max end_s."""
        notes = [
            _make_note(clip_id="a", start_s=0.0, end_s=0.5, midi=60),
            _make_note(clip_id="b", start_s=0.0, end_s=2.0, midi=64),
            _make_note(clip_id="c", start_s=0.0, end_s=1.0, midi=67),
        ]
        result = group_simultaneous_notes(notes)
        assert len(result) == 1
        chord = result[0]
        assert chord.midi is None
        # end_s must be the maximum across the bucket
        assert chord.end_s == 2.0
        # start_s must be the minimum
        assert chord.start_s == 0.0
        assert chord.midi_notes == frozenset({60, 64, 67})

    def test_tolerance_window_groups_near_notes(self):
        """Notes within tolerance_s of each other must be grouped together."""
        notes = [
            _make_note(clip_id="a", start_s=0.000, midi=60),
            _make_note(clip_id="b", start_s=0.010, midi=64),  # within default 0.012
            _make_note(clip_id="c", start_s=0.015, midi=67),  # outside 0.012 from a
        ]
        result = group_simultaneous_notes(notes)
        # a and b are within tolerance, c is outside
        assert len(result) == 2
        assert result[0].midi is None
        assert result[0].midi_notes == frozenset({60, 64})
        assert result[1].midi == 67

    def test_different_audio_paths_not_grouped(self):
        """Notes on different audio_paths must not be grouped together."""
        notes = [
            _make_note(clip_id="a", start_s=0.0, midi=60, audio_path="/a.wav"),
            _make_note(clip_id="b", start_s=0.0, midi=64, audio_path="/b.wav"),
        ]
        result = group_simultaneous_notes(notes)
        assert len(result) == 2
        assert result[0].midi == 60
        assert result[1].midi == 64

    def test_different_sources_not_grouped(self):
        """Notes from different sources must not be grouped together."""
        notes = [
            _make_note(clip_id="a", start_s=0.0, midi=60, source="A"),
            _make_note(clip_id="b", start_s=0.0, midi=64, source="B"),
        ]
        result = group_simultaneous_notes(notes)
        assert len(result) == 2

    def test_different_techniques_not_grouped(self):
        """Notes with different techniques must not be grouped together."""
        notes = [
            _make_note(clip_id="a", start_s=0.0, midi=60, technique="pick"),
            _make_note(clip_id="b", start_s=0.0, midi=64, technique="bend"),
        ]
        result = group_simultaneous_notes(notes)
        assert len(result) == 2

    def test_chord_events_pass_through_unchanged(self):
        """Existing chord events (midi is None) must pass through unmodified."""
        chord = _make_chord(
            clip_id="existing/chord",
            notes=(ClipExpectedNote(midi=60), ClipExpectedNote(midi=64)),
        )
        result = group_simultaneous_notes([chord])
        assert len(result) == 1
        assert result[0] == chord
        assert result[0].midi is None
        assert result[0].clip_id == "existing/chord"

    def test_mixed_single_and_chord_events(self):
        """Chord events must pass through while singles are grouped."""
        chord = _make_chord(
            clip_id="existing/chord",
            notes=(ClipExpectedNote(midi=60), ClipExpectedNote(midi=64)),
            start_s=0.0,
        )
        singles = [
            _make_note(clip_id="a", start_s=0.0, midi=67, string=3),
            _make_note(clip_id="b", start_s=0.0, midi=72, string=4),
        ]
        result = group_simultaneous_notes([chord, *singles])
        assert len(result) == 2
        # The existing chord is in the result
        assert any(e.clip_id == "existing/chord" for e in result)
        # The singles became a chord — clip_id inherits first note's id with ":chord" suffix
        chord2 = [e for e in result if e.clip_id == "a:chord"]
        assert len(chord2) == 1
        assert chord2[0].midi_notes == frozenset({67, 72})
    def test_custom_tolerance(self):
        """The tolerance_s parameter must be adjustable."""
        notes = [
            _make_note(clip_id="a", start_s=0.000, midi=60),
            _make_note(clip_id="b", start_s=0.050, midi=64),
        ]
        # Default tolerance (0.012) does not group these
        assert len(group_simultaneous_notes(notes)) == 2
        # Wider tolerance does group them
        result = group_simultaneous_notes(notes, tolerance_s=0.1)
        assert len(result) == 1
        assert result[0].midi_notes == frozenset({60, 64})

    def test_chord_inherits_min_confidence(self):
        """The chord confidence must be the minimum among its notes."""
        notes = [
            _make_note(clip_id="a", start_s=0.0, midi=60, confidence=0.8),
            _make_note(clip_id="b", start_s=0.0, midi=64, confidence=0.5),
        ]
        result = group_simultaneous_notes(notes)
        assert len(result) == 1
        assert result[0].confidence == 0.5

    def test_chord_clip_id_suffix(self):
        """Chord clip_ids must end with ':chord' suffix."""
        notes = [
            _make_note(clip_id="test/note_a", start_s=0.0, midi=60),
            _make_note(clip_id="test/note_b", start_s=0.0, midi=64),
        ]
        result = group_simultaneous_notes(notes)
        assert len(result) == 1
        assert result[0].clip_id.endswith(":chord")

    def test_notes_sorted_by_string_in_chord(self):
        """Chord notes must be sorted by string number (ascending)."""
        notes = [
            _make_note(clip_id="a", start_s=0.0, midi=67, string=3),
            _make_note(clip_id="b", start_s=0.0, midi=64, string=2),
            _make_note(clip_id="c", start_s=0.0, midi=60, string=1),
        ]
        result = group_simultaneous_notes(notes)
        assert len(result) == 1
        assert result[0].notes == (
            ClipExpectedNote(midi=60, string=1, fret=None),
            ClipExpectedNote(midi=64, string=2, fret=None),
            ClipExpectedNote(midi=67, string=3, fret=None),
        )

    def test_result_sorted_by_audio_path_then_start_s(self):
        """The final grouped list must be sorted by (audio_path, start_s, clip_id)."""
        notes = [
            _make_note(clip_id="b", start_s=0.0, midi=64, audio_path="/b.wav"),
            _make_note(clip_id="a", start_s=1.0, midi=60, audio_path="/a.wav"),
        ]
        result = group_simultaneous_notes(notes)
        assert len(result) == 2
        assert result[0].audio_path == "/a.wav"
        assert result[1].audio_path == "/b.wav"

    def test_preserves_metadata(self):
        """Metadata from the first note in a bucket must be preserved."""
        notes = [
            _make_note(clip_id="a", start_s=0.0, midi=60, metadata={"key": "value"}),
            _make_note(clip_id="b", start_s=0.0, midi=64),
        ]
        result = group_simultaneous_notes(notes)
        assert len(result) == 1
        assert result[0].metadata == {"key": "value"}
