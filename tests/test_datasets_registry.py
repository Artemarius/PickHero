"""Tests for DatasetRegistry: construction, scanning, serialization, caching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pickhero.datasets import ClipEvent, DatasetRegistry
from pickhero.datasets.schema import ClipExpectedNote


class TestDatasetRegistryConstruction:
    """Registry creation and lazy importer loading."""

    def test_constructor_defaults(self, tmp_path):
        """Registry with only cache_dir sets sensible defaults."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        assert registry._cache_dir == tmp_path
        assert registry._cache_file == tmp_path / "events.jsonl"
        assert registry._dataset_paths == {}
        assert registry._importers is None  # lazy
        assert tmp_path.is_dir()  # constructor creates the dir

    def test_constructor_with_dataset_paths(self, tmp_path):
        """Constructor accepts initial dataset path mapping."""
        paths = {"GOAT": "/data/goat", "IDMT": "/data/idmt"}
        registry = DatasetRegistry(dataset_paths=paths, cache_dir=tmp_path)
        assert registry._dataset_paths["GOAT"] == Path("/data/goat")
        assert registry._dataset_paths["IDMT"] == Path("/data/idmt")
        assert len(registry._dataset_paths) == 2

    def test_constructor_empty_paths(self, tmp_path):
        """Constructor handles None dataset_paths."""
        registry = DatasetRegistry(dataset_paths=None, cache_dir=tmp_path)
        assert registry._dataset_paths == {}

    def test_constructor_default_cache_dir(self, tmp_path):
        """When cache_dir is None, defaults to ~/.pickhero/datasets."""
        import os
        from unittest.mock import patch as mock_patch

        expected = Path.home() / ".pickhero" / "datasets"
        with mock_patch("pathlib.Path.mkdir"):
            registry = DatasetRegistry()
        assert registry._cache_dir == expected
        assert registry._cache_file == expected / "events.jsonl"

    def test_load_importers_returns_cached(self, tmp_path):
        """_load_importers caches after first call."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        first = registry._load_importers()
        second = registry._load_importers()
        assert second is first  # same dict object

    def test_load_importers_has_expected_keys(self, tmp_path):
        """_load_importers loads all four known importers."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        importers = registry._load_importers()
        assert set(importers.keys()) == {"GOAT", "Guitar-TECHS", "GuitarSet", "IDMT"}

    def test_load_importers_are_importer_instances(self, tmp_path):
        """Each importer is a DatasetImporter subclass instance."""
        from pickhero.datasets.base import DatasetImporter

        registry = DatasetRegistry(cache_dir=tmp_path)
        for imp in registry._load_importers().values():
            assert isinstance(imp, DatasetImporter)


class TestDatasetRegistrySetPath:
    """set_path updates the dataset path mapping."""

    def test_set_path_string(self, tmp_path):
        """set_path accepts a string path."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        registry.set_path("GOAT", "/custom/goat")
        assert registry._dataset_paths["GOAT"] == Path("/custom/goat")

    def test_set_path_pathlib(self, tmp_path):
        """set_path accepts a Path object."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        p = tmp_path / "guitarset"
        registry.set_path("GuitarSet", p)
        assert registry._dataset_paths["GuitarSet"] == p

    def test_set_path_overwrites(self, tmp_path):
        """set_path overwrites an existing source path."""
        registry = DatasetRegistry(
            dataset_paths={"GOAT": "/old/path"},
            cache_dir=tmp_path,
        )
        registry.set_path("GOAT", "/new/path")
        assert registry._dataset_paths["GOAT"] == Path("/new/path")

    def test_set_path_new_source(self, tmp_path):
        """set_path adds a source not in the original mapping."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        registry.set_path("IDMT", "/data/idmt")
        assert "IDMT" in registry._dataset_paths


class TestDatasetRegistryScanDatasets:
    """scan_datasets behavior with various path states."""

    def test_scan_no_paths(self, tmp_path):
        """With no configured paths, scan returns empty list."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        assert registry.scan_datasets() == []

    def test_scan_nonexistent_path(self, tmp_path):
        """With a nonexistent dataset directory, scan returns empty list."""
        registry = DatasetRegistry(
            dataset_paths={"GOAT": "/nonexistent/goat_12345"},
            cache_dir=tmp_path,
        )
        events = registry.scan_datasets()
        assert events == []

    def test_scan_empty_directory(self, tmp_path):
        """With an existing-but-empty dataset dir, scan returns empty list."""
        empty_dir = tmp_path / "goat_empty"
        empty_dir.mkdir()
        registry = DatasetRegistry(
            dataset_paths={"GOAT": empty_dir},
            cache_dir=tmp_path,
        )
        events = registry.scan_datasets()
        assert events == []

    def test_scan_unknown_source_skipped(self, tmp_path):
        """A source with no matching importer is silently skipped."""
        registry = DatasetRegistry(
            dataset_paths={"UnknownSource": tmp_path},
            cache_dir=tmp_path,
        )
        events = registry.scan_datasets()
        assert events == []

    def test_scan_deduplicates_by_clip_id(self, tmp_path):
        """Duplicate clip_ids across sources are deduplicated (first wins)."""
        # To test dedup we need controlled output from multiple importers.
        # Use mocks to return events with the same clip_id from different sources.
        from unittest.mock import MagicMock, patch

        mock_goat = MagicMock()
        mock_goat.scan.return_value = [
            ClipEvent(
                clip_id="dup/1",
                source="GOAT",
                start_s=0.0, end_s=1.0, midi=60, notes=(),
                string=None, fret=None, technique="normal",
                confidence=1.0, audio_path="/a.wav",
            ),
        ]
        mock_techs = MagicMock()
        mock_techs.scan.return_value = [
            ClipEvent(
                clip_id="dup/1",  # same id
                source="Guitar-TECHS",
                start_s=0.5, end_s=1.5, midi=62, notes=(),
                string=None, fret=None, technique="slide",
                confidence=0.9, audio_path="/b.wav",
            ),
        ]
        with patch.object(
            DatasetRegistry,
            "_load_importers",
            return_value={"GOAT": mock_goat, "Guitar-TECHS": mock_techs},
        ):
            registry = DatasetRegistry(
                dataset_paths={"GOAT": "/d1", "Guitar-TECHS": "/d2"},
                cache_dir=tmp_path,
            )
            events = registry.scan_datasets()
        assert len(events) == 1
        assert events[0].source == "GOAT"  # first-encountered wins
        assert events[0].midi == 60

    def test_scan_writes_cache(self, tmp_path):
        """scan_datasets also triggers _write_cache."""
        from unittest.mock import MagicMock, patch

        mock_imp = MagicMock()
        mock_imp.scan.return_value = [
            ClipEvent(
                clip_id="cache_test/1",
                source="GOAT",
                start_s=0.0, end_s=1.0, midi=64, notes=(),
                string=None, fret=None, technique="normal",
                confidence=1.0, audio_path="/x.wav",
            ),
        ]
        with patch.object(
            DatasetRegistry,
            "_load_importers",
            return_value={"GOAT": mock_imp},
        ):
            registry = DatasetRegistry(
                dataset_paths={"GOAT": "/d"},
                cache_dir=tmp_path,
            )
            assert not registry._cache_file.exists()
            registry.scan_datasets()
        # Cache file was created
        assert registry._cache_file.exists()
        content = registry._cache_file.read_text()
        assert "cache_test/1" in content


class TestDatasetRegistryEventSerialization:
    """_event_to_dict / _event_from_dict round-trip."""

    def _round_trip(self, registry: DatasetRegistry, event: ClipEvent) -> ClipEvent:
        """Helper: convert event to dict and back."""
        return registry._event_from_dict(registry._event_to_dict(event))

    def test_round_trip_single_note(self, tmp_path):
        """Single-note event survives dict round-trip."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        original = ClipEvent(
            clip_id="rt/single_note",
            source="GOAT",
            start_s=0.0, end_s=1.5,
            midi=72,
            notes=(),
            string=1, fret=12,
            technique="bend",
            confidence=0.85,
            audio_path="/audio/test.wav",
            metadata={"artist": "test", "genre": "rock"},
        )
        restored = self._round_trip(registry, original)
        assert restored.clip_id == original.clip_id
        assert restored.source == original.source
        assert restored.start_s == original.start_s
        assert restored.end_s == original.end_s
        assert restored.midi == original.midi
        assert restored.notes == original.notes
        assert restored.string == original.string
        assert restored.fret == original.fret
        assert restored.technique == original.technique
        assert restored.confidence == pytest.approx(original.confidence)
        assert restored.audio_path == original.audio_path
        assert restored.metadata == original.metadata

    def test_round_trip_chord(self, tmp_path):
        """Chord event with multiple notes round-trips correctly."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        original = ClipEvent(
            clip_id="rt/chord_event",
            source="GuitarSet",
            start_s=1.0, end_s=2.0,
            midi=None,
            notes=(
                ClipExpectedNote(midi=60, string=3, fret=0, role="root"),
                ClipExpectedNote(midi=64, string=2, fret=1, role="third"),
                ClipExpectedNote(midi=67, string=1, fret=2, role="fifth"),
            ),
            string=None,
            fret=None,
            technique="chord",
            confidence=0.95,
            audio_path="/audio/chord.wav",
            metadata={"quality": "major", "inversion": "root"},
        )
        restored = self._round_trip(registry, original)
        assert restored.clip_id == original.clip_id
        assert restored.midi is None
        assert len(restored.notes) == 3
        for rn, on in zip(restored.notes, original.notes):
            assert rn.midi == on.midi
            assert rn.string == on.string
            assert rn.fret == on.fret
            assert rn.role == on.role
        assert restored.metadata == original.metadata

    def test_round_trip_optional_none(self, tmp_path):
        """Fields set to None survive the dict round-trip as None."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        original = ClipEvent(
            clip_id="rt/opt_none",
            source="IDMT",
            start_s=0.0, end_s=0.5,
            midi=42,
            notes=(),
            string=None,
            fret=None,
            technique="normal",
            confidence=1.0,
            audio_path="/e.wav",
        )
        restored = self._round_trip(registry, original)
        assert restored.string is None
        assert restored.fret is None
        assert restored.metadata == {}

    def test_round_trip_chord_no_role(self, tmp_path):
        """Chord notes with None role round-trip correctly."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        original = ClipEvent(
            clip_id="rt/no_role",
            source="GuitarSet",
            start_s=0.0, end_s=1.0,
            midi=None,
            notes=(
                ClipExpectedNote(midi=60, string=3, fret=0, role=None),
                ClipExpectedNote(midi=64, string=1, fret=5, role=None),
            ),
            string=None,
            fret=None,
            technique="chord",
            confidence=1.0,
            audio_path="/f.wav",
        )
        restored = self._round_trip(registry, original)
        assert restored.notes[0].role is None
        assert restored.notes[1].role is None

    def test_round_trip_empty_notes(self, tmp_path):
        """Single-note event with empty notes tuple round-trips."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        original = ClipEvent(
            clip_id="rt/empty_notes",
            source="GOAT",
            start_s=0.0, end_s=1.0,
            midi=48,
            notes=(),
            string=6, fret=0,
            technique="normal",
            confidence=1.0,
            audio_path="/g.wav",
        )
        restored = self._round_trip(registry, original)
        assert restored.notes == ()
        assert restored.midi == 48

    def test_round_trip_empty_metadata(self, tmp_path):
        """Empty metadata round-trips to empty dict."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        original = ClipEvent(
            clip_id="rt/empty_meta",
            source="GOAT",
            start_s=0.0, end_s=1.0,
            midi=60,
            notes=(),
            string=None, fret=None,
            technique="normal",
            confidence=1.0,
            audio_path="/h.wav",
            metadata={},
        )
        restored = self._round_trip(registry, original)
        assert restored.metadata == {}

    def test_schema_version_in_dict(self, tmp_path):
        """_event_to_dict includes schema_version."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        event = ClipEvent(
            clip_id="v/1", source="GOAT",
            start_s=0.0, end_s=1.0, midi=60, notes=(),
            string=None, fret=None, technique="normal",
            confidence=1.0, audio_path="/x.wav",
        )
        data = registry._event_to_dict(event)
        assert data["schema_version"] == 3

    def test_event_from_dict_v2_compat(self, tmp_path):
        """_event_from_dict handles schema_version 2 (same format as 3)."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        data = {
            "schema_version": 2,
            "clip_id": "v2/test",
            "source": "GOAT",
            "start_s": 0.0,
            "end_s": 1.0,
            "midi": 65,
            "notes": [],
            "string": None,
            "fret": None,
            "technique": "normal",
            "confidence": 1.0,
            "audio_path": "/v2.wav",
            "metadata": {},
        }
        event = registry._event_from_dict(data)
        assert event.clip_id == "v2/test"
        assert event.midi == 65

    def test_event_from_dict_legacy_v1(self, tmp_path):
        """_event_from_dict handles legacy v1 schema with midi_notes."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        data = {
            "clip_id": "v1/test",
            "source": "GOAT",
            "start_s": 0.0,
            "end_s": 1.0,
            "midi": None,
            "midi_notes": [60, 64, 67],
            "string": None,
            "fret": None,
            "technique": "chord",
            "confidence": 1.0,
            "audio_path": "/v1.wav",
            "metadata": {},
        }
        event = registry._event_from_dict(data)
        assert event.midi is None
        assert len(event.notes) == 3
        assert [n.midi for n in event.notes] == [60, 64, 67]
        assert all(n.string is None for n in event.notes)
        assert all(n.fret is None for n in event.notes)

    def test_event_from_dict_v1_missing_schema_version(self, tmp_path):
        """Dicts without schema_version are treated as v1."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        data = {
            "clip_id": "nov/test",
            "source": "GOAT",
            "start_s": 0.0,
            "end_s": 1.0,
            "midi": None,
            "midi_notes": [72],
            "string": None,
            "fret": None,
            "technique": "normal",
            "confidence": 1.0,
            "audio_path": "/nov.wav",
            "metadata": {},
        }
        event = registry._event_from_dict(data)
        assert len(event.notes) == 1
        assert event.notes[0].midi == 72


class TestDatasetRegistryCacheIO:
    """_read_cache and _write_cache behavior."""

    def test_read_cache_missing_file(self, tmp_path):
        """_read_cache with no cache file returns empty list."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        assert not registry._cache_file.exists()
        assert registry._read_cache() == []

    def test_write_cache_creates_file(self, tmp_path):
        """_write_cache creates the cache file with JSONL content."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        event = ClipEvent(
            clip_id="wc/1", source="GOAT",
            start_s=0.0, end_s=1.0, midi=60, notes=(),
            string=None, fret=None, technique="normal",
            confidence=1.0, audio_path="/a.wav",
        )
        registry._write_cache([event])
        assert registry._cache_file.is_file()
        content = registry._cache_file.read_text().strip()
        assert len(content.split("\n")) == 1

    def test_write_cache_overwrites(self, tmp_path):
        """_write_cache replaces existing cache content."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        # Write first event
        e1 = ClipEvent(
            clip_id="ov/1", source="GOAT",
            start_s=0.0, end_s=1.0, midi=60, notes=(),
            string=None, fret=None, technique="normal",
            confidence=1.0, audio_path="/a.wav",
        )
        registry._write_cache([e1])
        # Write second, replacing first
        e2 = ClipEvent(
            clip_id="ov/2", source="IDMT",
            start_s=0.0, end_s=1.0, midi=62, notes=(),
            string=None, fret=None, technique="slide",
            confidence=0.9, audio_path="/b.wav",
        )
        registry._write_cache([e2])
        loaded = registry._read_cache()
        assert len(loaded) == 1
        assert loaded[0].clip_id == "ov/2"

    def test_write_cache_jsonl_format(self, tmp_path):
        """Cache file uses one JSON line per event."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        e1 = ClipEvent(
            clip_id="fmt/1", source="GOAT",
            start_s=0.0, end_s=1.0, midi=60, notes=(),
            string=None, fret=None, technique="normal",
            confidence=1.0, audio_path="/a.wav",
        )
        e2 = ClipEvent(
            clip_id="fmt/2", source="GuitarSet",
            start_s=0.5, end_s=1.5, midi=None,
            notes=(ClipExpectedNote(midi=60, string=3, fret=0),),
            string=None, fret=None, technique="chord",
            confidence=1.0, audio_path="/b.wav",
        )
        registry._write_cache([e1, e2])
        lines = registry._cache_file.read_text().strip().split("\n")
        assert len(lines) == 2
        d1 = json.loads(lines[0])
        assert d1["clip_id"] == "fmt/1"
        assert d1["schema_version"] == 3
        d2 = json.loads(lines[1])
        assert d2["clip_id"] == "fmt/2"
        assert len(d2["notes"]) == 1

    def test_read_cache_skips_empty_lines(self, tmp_path):
        """_read_cache handles blank lines gracefully."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        event = ClipEvent(
            clip_id="bl/1", source="GOAT",
            start_s=0.0, end_s=1.0, midi=64, notes=(),
            string=None, fret=None, technique="normal",
            confidence=1.0, audio_path="/x.wav",
        )
        registry._write_cache([event])
        # Append extra blank lines
        content = registry._cache_file.read_text()
        registry._cache_file.write_text(content + "\n\n  \n")
        loaded = registry._read_cache()
        assert len(loaded) == 1
        assert loaded[0].clip_id == "bl/1"

    def test_read_cache_multiple_events(self, tmp_path):
        """_read_cache restores all written events in order."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        events = [
            ClipEvent(
                clip_id=f"me/{i}", source="GOAT",
                start_s=float(i), end_s=float(i + 1),
                midi=60 + i, notes=(),
                string=None, fret=None, technique="normal",
                confidence=1.0, audio_path=f"/{i}.wav",
            )
            for i in range(5)
        ]
        registry._write_cache(events)
        loaded = registry._read_cache()
        assert len(loaded) == 5
        for orig, got in zip(events, loaded):
            assert got.clip_id == orig.clip_id
            assert got.midi == orig.midi

    def test_write_then_read_metadata(self, tmp_path):
        """Metadata dict round-trips through cache."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        event = ClipEvent(
            clip_id="meta/1", source="GOAT",
            start_s=0.0, end_s=1.0, midi=60, notes=(),
            string=None, fret=None, technique="normal",
            confidence=1.0, audio_path="/m.wav",
            metadata={"key1": "val1", "key2": "val2"},
        )
        registry._write_cache([event])
        loaded = registry._read_cache()
        assert loaded[0].metadata == {"key1": "val1", "key2": "val2"}


class TestDatasetRegistryLoadEvents:
    """load_events filtering behavior."""

    def test_load_no_cache(self, tmp_path):
        """With no cache file, load_events returns empty list."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        assert registry.load_events() == []
        assert registry.load_events({"source": "GOAT"}) == []

    def test_load_no_filter(self, tmp_path):
        """With no filter, load_events returns all cached events."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        events = [
            ClipEvent(
                clip_id=f"nf/{i}", source="GOAT",
                start_s=0.0, end_s=1.0,
                midi=60 + i, notes=(),
                string=None, fret=None, technique="normal",
                confidence=1.0, audio_path=f"/{i}.wav",
            )
            for i in range(3)
        ]
        registry._write_cache(events)
        result = registry.load_events()
        assert len(result) == 3

    def test_load_filter_source(self, tmp_path):
        """load_events filters events by source."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        e1 = ClipEvent(
            clip_id="fs/1", source="GOAT",
            start_s=0.0, end_s=1.0, midi=60, notes=(),
            string=None, fret=None, technique="normal",
            confidence=1.0, audio_path="/a.wav",
        )
        e2 = ClipEvent(
            clip_id="fs/2", source="IDMT",
            start_s=0.0, end_s=1.0, midi=62, notes=(),
            string=None, fret=None, technique="slide",
            confidence=0.9, audio_path="/b.wav",
        )
        registry._write_cache([e1, e2])
        goat = registry.load_events({"source": "GOAT"})
        assert len(goat) == 1
        assert goat[0].clip_id == "fs/1"
        assert goat[0].source == "GOAT"
        idmt = registry.load_events({"source": "IDMT"})
        assert len(idmt) == 1
        assert idmt[0].source == "IDMT"

    def test_load_filter_source_no_match(self, tmp_path):
        """Filtering by source with no match returns empty list."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        event = ClipEvent(
            clip_id="nm/1", source="GOAT",
            start_s=0.0, end_s=1.0, midi=60, notes=(),
            string=None, fret=None, technique="normal",
            confidence=1.0, audio_path="/a.wav",
        )
        registry._write_cache([event])
        result = registry.load_events({"source": "NonExistent"})
        assert result == []

    def test_load_filter_technique(self, tmp_path):
        """load_events filters events by technique."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        e1 = ClipEvent(
            clip_id="ft/1", source="GOAT",
            start_s=0.0, end_s=1.0, midi=60, notes=(),
            string=None, fret=None, technique="normal",
            confidence=1.0, audio_path="/a.wav",
        )
        e2 = ClipEvent(
            clip_id="ft/2", source="GOAT",
            start_s=1.0, end_s=2.0, midi=62, notes=(),
            string=None, fret=None, technique="slide",
            confidence=0.9, audio_path="/b.wav",
        )
        registry._write_cache([e1, e2])
        slides = registry.load_events({"technique": "slide"})
        assert len(slides) == 1
        assert slides[0].technique == "slide"

    def test_load_filter_midi_single_note(self, tmp_path):
        """load_events filters single-note events by midi."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        e1 = ClipEvent(
            clip_id="fm/1", source="GOAT",
            start_s=0.0, end_s=1.0, midi=60, notes=(),
            string=None, fret=None, technique="normal",
            confidence=1.0, audio_path="/a.wav",
        )
        e2 = ClipEvent(
            clip_id="fm/2", source="GOAT",
            start_s=1.0, end_s=2.0, midi=62, notes=(),
            string=None, fret=None, technique="normal",
            confidence=1.0, audio_path="/b.wav",
        )
        registry._write_cache([e1, e2])
        result = registry.load_events({"midi": 60})
        assert len(result) == 1
        assert result[0].midi == 60

    def test_load_filter_midi_chord(self, tmp_path):
        """load_events filters chord events by midi in notes."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        e1 = ClipEvent(
            clip_id="fc/1", source="GuitarSet",
            start_s=0.0, end_s=1.0, midi=None,
            notes=(ClipExpectedNote(midi=60), ClipExpectedNote(midi=64)),
            string=None, fret=None, technique="chord",
            confidence=1.0, audio_path="/a.wav",
        )
        e2 = ClipEvent(
            clip_id="fc/2", source="GuitarSet",
            start_s=1.0, end_s=2.0, midi=None,
            notes=(ClipExpectedNote(midi=62), ClipExpectedNote(midi=67)),
            string=None, fret=None, technique="chord",
            confidence=1.0, audio_path="/b.wav",
        )
        registry._write_cache([e1, e2])
        result = registry.load_events({"midi": 64})
        assert len(result) == 1
        assert result[0].clip_id == "fc/1"

    def test_load_filter_combined(self, tmp_path):
        """load_events supports combined source + technique filter."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        events = [
            ClipEvent(
                clip_id=f"cm/{i}", source=src,
                start_s=0.0, end_s=1.0, midi=60 + i, notes=(),
                string=None, fret=None, technique=tech,
                confidence=1.0, audio_path=f"/{i}.wav",
            )
            for i, (src, tech) in enumerate([
                ("GOAT", "normal"),
                ("GOAT", "slide"),
                ("IDMT", "normal"),
                ("IDMT", "slide"),
            ])
        ]
        registry._write_cache(events)
        f = {"source": "GOAT", "technique": "slide"}
        result = registry.load_events(f)
        assert len(result) == 1
        assert result[0].clip_id == "cm/1"

    def test_load_filter_empty_dict(self, tmp_path):
        """An empty filter dict returns all cached events (no filtering)."""
        registry = DatasetRegistry(cache_dir=tmp_path)
        e1 = ClipEvent(
            clip_id="ed/1", source="GOAT",
            start_s=0.0, end_s=1.0, midi=60, notes=(),
            string=None, fret=None, technique="normal",
            confidence=1.0, audio_path="/a.wav",
        )
        registry._write_cache([e1])
        result = registry.load_events({})
        assert len(result) == 1
        assert result[0].clip_id == "ed/1"
