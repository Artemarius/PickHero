"""Dataset registry: manage paths, normalize events, cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pickhero.datasets.schema import ClipEvent, ClipExpectedNote

if TYPE_CHECKING:
    from pickhero.datasets.base import DatasetImporter


class DatasetRegistry:
    """Manages paths to downloaded datasets and normalized event caches."""

    def __init__(
        self,
        dataset_paths: dict[str, str | Path] | None = None,
        cache_dir: str | Path | None = None,
    ):
        self._dataset_paths: dict[str, Path] = {
            name: Path(p) for name, p in (dataset_paths or {}).items()
        }
        if cache_dir is None:
            cache_dir = Path.home() / ".pickhero" / "datasets"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / "events.jsonl"
        self._importers: dict[str, DatasetImporter] | None = None

    def _load_importers(self) -> dict[str, DatasetImporter]:
        """Lazy-load importers to avoid importing optional dependencies early."""
        if self._importers is not None:
            return self._importers
        from pickhero.datasets.goat_importer import GoatImporter
        from pickhero.datasets.guitar_techs_importer import GuitarTechsImporter
        from pickhero.datasets.guitarset_importer import GuitarSetImporter
        from pickhero.datasets.idmt_importer import IdmtImporter

        self._importers = {
            "GOAT": GoatImporter(),
            "Guitar-TECHS": GuitarTechsImporter(),
            "GuitarSet": GuitarSetImporter(),
            "IDMT": IdmtImporter(),
        }
        return self._importers

    def set_path(self, source: str, path: str | Path) -> None:
        """Point ``source`` to its downloaded dataset directory."""
        self._dataset_paths[source] = Path(path)

    def scan_datasets(self) -> list[ClipEvent]:
        """Run all configured importers, deduplicate, and write cache."""
        events: list[ClipEvent] = []
        seen: set[str] = set()
        importers = self._load_importers()
        for source, path in self._dataset_paths.items():
            importer = importers.get(source)
            if importer is None:
                continue
            for event in importer.scan(path):
                if event.clip_id not in seen:
                    seen.add(event.clip_id)
                    events.append(event)
        self._write_cache(events)
        return events

    def _write_cache(self, events: list[ClipEvent]) -> None:
        """Write normalized events to the JSONL cache."""
        with self._cache_file.open("w") as f:
            for event in events:
                f.write(json.dumps(self._event_to_dict(event)) + "\n")

    def _read_cache(self) -> list[ClipEvent]:
        """Read normalized events from the JSONL cache."""
        if not self._cache_file.exists():
            return []
        events: list[ClipEvent] = []
        with self._cache_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(self._event_from_dict(json.loads(line)))
        return events

    @staticmethod
    def _event_to_dict(event: ClipEvent) -> dict:
        return {
            "schema_version": 3,
            "clip_id": event.clip_id,
            "source": event.source,
            "start_s": event.start_s,
            "end_s": event.end_s,
            "midi": event.midi,
            "notes": [
                {
                    "midi": n.midi,
                    "string": n.string,
                    "fret": n.fret,
                    "role": n.role,
                }
                for n in event.notes
            ],
            "string": event.string,
            "fret": event.fret,
            "technique": event.technique,
            "confidence": event.confidence,
            "audio_path": event.audio_path,
            "metadata": dict(event.metadata),
        }

    @staticmethod
    def _event_from_dict(data: dict) -> ClipEvent:
        schema_version = data.get("schema_version", 1)
        if schema_version >= 2:
            notes = tuple(
                ClipExpectedNote(
                    midi=int(n["midi"]),
                    string=int(n["string"]) if n.get("string") is not None else None,
                    fret=int(n["fret"]) if n.get("fret") is not None else None,
                    role=n.get("role"),
                )
                for n in data.get("notes", [])
            )
        else:
            # Legacy v1 cache used a flat list of MIDI ints under "midi_notes".
            notes = tuple(
                ClipExpectedNote(midi=int(x))
                for x in data.get("midi_notes", [])
            )
        return ClipEvent(
            clip_id=data["clip_id"],
            source=data["source"],
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            midi=int(data["midi"]) if data.get("midi") is not None else None,
            notes=notes,
            string=int(data["string"]) if data.get("string") is not None else None,
            fret=int(data["fret"]) if data.get("fret") is not None else None,
            technique=str(data["technique"]),
            confidence=float(data["confidence"]),
            audio_path=str(data["audio_path"]),
            metadata={
                str(key): str(value)
                for key, value in data.get("metadata", {}).items()
            } if isinstance(data.get("metadata", {}), dict) else {},
        )

    def load_events(self, filter: dict | None = None) -> list[ClipEvent]:
        """Read from cache with optional source/technique/midi filters."""
        events = self._read_cache()
        if not filter:
            return events
        source_filter = filter.get("source")
        technique_filter = filter.get("technique")
        midi_filter = filter.get("midi")
        result: list[ClipEvent] = []
        for event in events:
            if source_filter and event.source != source_filter:
                continue
            if technique_filter and event.technique != technique_filter:
                continue
            if midi_filter is not None:
                midis = {event.midi} if event.midi is not None else set(event.midi_notes or ())
                if midi_filter not in midis:
                    continue
            result.append(event)
        return result
