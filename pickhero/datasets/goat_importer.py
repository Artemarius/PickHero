"""GOAT dataset importer.

GOAT provides tab+audio paired JSONL + WAV electric-guitar DI data.
Importer expects files like ``annotations.jsonl`` alongside WAV clips.
"""

from __future__ import annotations

import json
from pathlib import Path

from pickhero.datasets.base import DatasetImporter
from pickhero.datasets.schema import ClipEvent, ClipExpectedNote

class GoatImporter(DatasetImporter):
    """Importer for the GOAT dataset."""

    name = "GOAT"

    def scan(self, path: str | Path) -> list[ClipEvent]:
        root = Path(path)
        if not root.exists():
            return []
        events: list[ClipEvent] = []
        for jsonl in sorted(root.rglob("*.jsonl")):
            audio = jsonl.with_suffix(".wav")
            if not audio.exists():
                audio = jsonl.with_suffix(".flac")
            if not audio.exists():
                continue
            try:
                events.extend(self._parse_jsonl(jsonl, audio))
            except Exception:
                # Malformed annotation file — skip.
                continue
        return events

    def _parse_jsonl(self, jsonl: Path, audio: Path) -> list[ClipEvent]:
        events: list[ClipEvent] = []
        stem = jsonl.stem
        with jsonl.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                midi = obj.get("midi")
                midi_notes = obj.get("midi_notes")
                if isinstance(midi_notes, (list, tuple)):
                    notes = tuple(
                        ClipExpectedNote(midi=int(m)) for m in midi_notes
                    )
                else:
                    notes = ()
                string = int(obj["string"]) if "string" in obj else None
                fret = int(obj["fret"]) if "fret" in obj else None
                metadata = {
                    str(key): str(obj[key])
                    for key in (
                        "player",
                        "guitar",
                        "pickup",
                        "interface",
                        "tone",
                        "tuning",
                        "instrument",
                        "audio_variant",
                    )
                    if obj.get(key) not in (None, "")
                }
                events.append(ClipEvent(
                    clip_id=f"goat/{stem}/{i}",
                    source=self.name,
                    start_s=float(obj.get("start", 0.0)),
                    end_s=float(obj.get("end", 0.0)),
                    midi=int(midi) if isinstance(midi, (int, float)) else None,
                    notes=notes,
                    string=string,
                    fret=fret,
                    technique=str(obj.get("technique", "normal")),
                    confidence=float(obj.get("confidence", 1.0)),
                    audio_path=str(audio),
                    metadata=metadata,
                ))
        return events
