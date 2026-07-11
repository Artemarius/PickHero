"""Tests for dataset schema, registry, and importers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pickhero.datasets import ClipEvent, DatasetRegistry
from pickhero.datasets.schema import ClipExpectedNote
from pickhero.datasets.goat_importer import GoatImporter


class TestClipEvent:
    """Validation tests for the unified schema."""

    def test_single_note_event(self):
        """A single-note event can be created and frozen."""
        event = ClipEvent(
            clip_id="test/1",
            source="GOAT",
            start_s=0.0,
            end_s=1.0,
            midi=64,
            notes=(),
            string=1,
            fret=5,
            technique="normal",
            confidence=1.0,
            audio_path="/tmp/test.wav",
        )
        assert event.midi == 64
        assert event.midi_notes is None

    def test_chord_event(self):
        """A chord event preserves per-note identity and exposes midi_notes."""
        event = ClipEvent(
            clip_id="test/2",
            source="GuitarSet",
            start_s=0.5,
            end_s=1.5,
            midi=None,
            notes=(
                ClipExpectedNote(midi=60),
                ClipExpectedNote(midi=64),
                ClipExpectedNote(midi=67),
            ),
            string=None,
            fret=None,
            technique="normal",
            confidence=0.9,
            audio_path="/tmp/test.wav",
        )
        assert event.midi is None
        assert event.notes
        assert event.midi_notes == frozenset({60, 64, 67})

    def test_event_rejects_both_midi_and_notes(self):
        """Validation must reject ambiguous events."""
        with pytest.raises(ValueError):
            ClipEvent(
                clip_id="bad",
                source="GOAT",
                start_s=0.0,
                end_s=1.0,
                midi=64,
                notes=(ClipExpectedNote(midi=64),),
                string=None,
                fret=None,
                technique="normal",
                confidence=1.0,
                audio_path="/tmp/test.wav",
            )

    def test_event_rejects_missing_pitches(self):
        """Validation must reject events with no pitch info."""
        with pytest.raises(ValueError):
            ClipEvent(
                clip_id="bad",
                source="GOAT",
                start_s=0.0,
                end_s=1.0,
                midi=None,
                notes=(),
                string=None,
                fret=None,
                technique="normal",
                confidence=1.0,
                audio_path="/tmp/test.wav",
            )


class TestGoatImporter:
    """Importer tests using temporary fixtures."""

    def test_empty_path_returns_empty_list(self):
        importer = GoatImporter()
        assert importer.scan("/does/not/exist") == []

    def test_jsonl_without_audio_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "clip.jsonl"
            jsonl.write_text(json.dumps({"midi": 64, "start": 0.0, "end": 1.0}))
            importer = GoatImporter()
            assert importer.scan(tmp) == []

    def test_jsonl_with_audio_imports_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "clip.wav"
            audio.write_bytes(b"RIFF" + b"\x00" * 100)  # fake WAV header
            jsonl = Path(tmp) / "clip.jsonl"
            jsonl.write_text(
                json.dumps({
                    "midi": 64,
                    "start": 0.1,
                    "end": 0.5,
                    "string": 2,
                    "fret": 5,
                    "technique": "bend",
                    "confidence": 0.95,
                })
                + "\n"
            )
            importer = GoatImporter()
            events = importer.scan(tmp)
            assert len(events) == 1
            assert events[0].midi == 64
            assert events[0].technique == "bend"
            assert events[0].confidence == pytest.approx(0.95)

    def test_jsonl_chord_imports_notes(self):
        """GOAT chord events populate the notes tuple."""
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "clip.wav"
            audio.write_bytes(b"RIFF" + b"\x00" * 100)
            jsonl = Path(tmp) / "clip.jsonl"
            jsonl.write_text(
                json.dumps({
                    "midi_notes": [60, 64, 67],
                    "start": 0.0,
                    "end": 1.0,
                    "string": 3,
                    "fret": 0,
                })
                + "\n"
            )
            events = GoatImporter().scan(tmp)
            assert len(events) == 1
            assert events[0].midi is None
            assert events[0].midi_notes == frozenset({60, 64, 67})
            assert len(events[0].notes) == 3


class TestDatasetRegistry:
    """Registry tests covering caching and filtering."""

    def test_registry_uses_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            registry = DatasetRegistry(cache_dir=cache)
            assert registry._cache_dir == cache

    def test_registry_scans_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = DatasetRegistry(
                dataset_paths={"GOAT": tmp},
                cache_dir=tmp,
            )
            events = registry.scan_datasets()
            assert events == []

    def test_registry_round_trips_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = DatasetRegistry(cache_dir=tmp)
            event = ClipEvent(
                clip_id="test/1",
                source="GOAT",
                start_s=0.0,
                end_s=1.0,
                midi=64,
                notes=(),
                string=None,
                fret=None,
                technique="normal",
                confidence=1.0,
                audio_path="/tmp/x.wav",
            )
            registry._write_cache([event])
            loaded = registry._read_cache()
            assert len(loaded) == 1
            assert loaded[0].clip_id == "test/1"
            assert loaded[0].midi == 64

    def test_registry_round_trips_chord_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = DatasetRegistry(cache_dir=tmp)
            event = ClipEvent(
                clip_id="test/2",
                source="GOAT",
                start_s=0.0,
                end_s=1.0,
                midi=None,
                notes=(
                    ClipExpectedNote(midi=60, string=3, fret=0),
                    ClipExpectedNote(midi=64, string=2, fret=0),
                ),
                string=None,
                fret=None,
                technique="chord",
                confidence=1.0,
                audio_path="/tmp/y.wav",
            )
            registry._write_cache([event])
            loaded = registry._read_cache()
            assert len(loaded) == 1
            assert loaded[0].midi_notes == frozenset({60, 64})
            assert [n.midi for n in loaded[0].notes] == [60, 64]

    def test_load_events_filters_by_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = DatasetRegistry(cache_dir=tmp)
            registry._write_cache([
                ClipEvent(
                    clip_id="a",
                    source="GOAT",
                    start_s=0.0,
                    end_s=1.0,
                    midi=60,
                    notes=(),
                    string=None,
                    fret=None,
                    technique="normal",
                    confidence=1.0,
                    audio_path="/tmp/a.wav",
                ),
                ClipEvent(
                    clip_id="b",
                    source="IDMT",
                    start_s=0.0,
                    end_s=1.0,
                    midi=62,
                    notes=(),
                    string=None,
                    fret=None,
                    technique="normal",
                    confidence=1.0,
                    audio_path="/tmp/b.wav",
                ),
            ])
            goat = registry.load_events({"source": "GOAT"})
            assert len(goat) == 1
            assert goat[0].source == "GOAT"

    def test_registry_reads_legacy_v1_cache(self):
        """Old caches with midi_notes still load."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = DatasetRegistry(cache_dir=tmp)
            registry._cache_file.write_text(
                json.dumps({
                    "clip_id": "legacy/1",
                    "source": "GOAT",
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "midi": None,
                    "midi_notes": [60, 64],
                    "string": None,
                    "fret": None,
                    "technique": "normal",
                    "confidence": 1.0,
                    "audio_path": "/tmp/legacy.wav",
                })
                + "\n"
            )
            loaded = registry._read_cache()
            assert len(loaded) == 1
            assert loaded[0].midi_notes == frozenset({60, 64})
            assert [n.midi for n in loaded[0].notes] == [60, 64]


class TestIdmtImporter:
    """Importer tests for IDMT XML annotations."""

    def test_empty_path_returns_empty_list(self):
        from pickhero.datasets.idmt_importer import IdmtImporter

        importer = IdmtImporter()
        assert importer.scan("/does/not/exist") == []

    def test_xml_with_audio_imports_event(self):
        from pickhero.datasets.idmt_importer import IdmtImporter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "audio").mkdir()
            (root / "audio" / "sample.wav").write_bytes(b"RIFF" + b"\x00" * 100)
            (root / "annotation").mkdir()
            (root / "annotation" / "sample.xml").write_text(
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
                "<instrumentRecording>\n"
                "  <globalParameter>\n"
                "    <audioFileName>sample.wav</audioFileName>\n"
                "  </globalParameter>\n"
                "  <transcription>\n"
                "    <event>\n"
                "      <pitch>40</pitch>\n"
                "      <onsetSec>0.1</onsetSec>\n"
                "      <offsetSec>0.3</offsetSec>\n"
                "      <fretNumber>0</fretNumber>\n"
                "      <stringNumber>1</stringNumber>\n"
                "      <excitationStyle>PK</excitationStyle>\n"
                "      <expressionStyle>BN</expressionStyle>\n"
                "    </event>\n"
                "  </transcription>\n"
                "</instrumentRecording>\n"
            )
            events = IdmtImporter().scan(root)
            assert len(events) == 1
            assert events[0].midi == 40
            assert events[0].start_s == pytest.approx(0.1)
            assert events[0].end_s == pytest.approx(0.3)
            # IDMT string 1 is low E, schema string 1 is high E.
            assert events[0].string == 6
            assert events[0].fret == 0
            assert events[0].technique == "bend"


class TestGuitarSetImporter:
    """Importer tests for GuitarSet JAMS annotations."""

    def test_empty_path_returns_empty_list(self):
        from pickhero.datasets.guitarset_importer import GuitarSetImporter

        importer = GuitarSetImporter()
        assert importer.scan("/does/not/exist") == []

    def test_jams_with_mix_audio_imports_event(self):
        from pickhero.datasets.guitarset_importer import GuitarSetImporter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample_mix.wav").write_bytes(b"RIFF" + b"\x00" * 100)
            jams = {
                "file_metadata": {
                    "title": "sample",
                    "duration": 1.0,
                    "identifiers": {},
                    "jams_version": "0.3.1",
                },
                "annotations": [
                    {
                        "namespace": "note_midi",
                        "data": [
                            {
                                "time": 0.0,
                                "duration": 0.2,
                                "value": 64.0,
                                "confidence": None,
                            }
                        ],
                    }
                ],
                "sandbox": {},
            }
            (root / "sample.jams").write_text(json.dumps(jams))
            events = GuitarSetImporter().scan(root)
            assert len(events) == 1
            assert events[0].midi == 64
            assert events[0].start_s == pytest.approx(0.0)
            assert events[0].end_s == pytest.approx(0.2)
            # Single annotation is low-E -> schema string 6.
            assert events[0].string == 6


class TestGuitarTechsImporter:
    """Importer tests for Guitar-TECHS MIDI annotations."""

    def test_empty_path_returns_empty_list(self):
        from pickhero.datasets.guitar_techs_importer import GuitarTechsImporter

        importer = GuitarTechsImporter()
        assert importer.scan("/does/not/exist") == []

    def test_midi_with_micamp_audio_imports_event(self):
        mido = pytest.importorskip("mido")
        from pickhero.datasets.guitar_techs_importer import GuitarTechsImporter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "P1_singlenotes").mkdir()
            (root / "P1_singlenotes" / "audio" / "micamp").mkdir(parents=True)
            (root / "P1_singlenotes" / "midi").mkdir(parents=True)
            (root / "P1_singlenotes" / "audio" / "micamp" / "micamp_sample.wav").write_bytes(
                b"RIFF" + b"\x00" * 100
            )
            mid = mido.MidiFile(type=0)
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("track_name", name="e", time=0))
            track.append(mido.Message("note_on", note=64, velocity=100, time=0))
            track.append(mido.Message("note_off", note=64, velocity=0, time=480))
            track.append(mido.MetaMessage("end_of_track", time=0))
            mid.tracks.append(track)
            mid.save(str(root / "P1_singlenotes" / "midi" / "midi_sample.mid"))
            events = GuitarTechsImporter().scan(root)
            assert len(events) == 1
            assert events[0].midi == 64
            assert events[0].string == 1
            assert events[0].audio_path.endswith("micamp_sample.wav")
