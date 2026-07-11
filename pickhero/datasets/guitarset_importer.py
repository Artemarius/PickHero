"""GuitarSet dataset importer.

GuitarSet ships hexaphonic audio with JAMS annotations.  This importer parses
the ``note_midi`` annotations (one per string) from each JAMS file and emits normalized note or simultaneous chord events.
"""

from __future__ import annotations

import json
from pathlib import Path

from pickhero.datasets.base import DatasetImporter
from pickhero.datasets.grouping import group_simultaneous_notes
from pickhero.datasets.schema import ClipEvent


class GuitarSetImporter(DatasetImporter):
    """Importer for the GuitarSet dataset."""

    name = "GuitarSet"

    def scan(self, path: str | Path) -> list[ClipEvent]:
        root = Path(path)
        if not root.exists():
            return []
        events: list[ClipEvent] = []
        for jams in sorted(root.rglob("*.jams")):
            audio = jams.with_suffix(".wav")
            if not audio.exists():
                # GuitarSet audio files use a _mix suffix.
                audio = jams.parent / (jams.stem + "_mix.wav")
            if not audio.exists():
                continue
            try:
                events.extend(self._parse_jams(jams, audio))
            except Exception:
                continue
        return events

    @staticmethod
    def _note_midi_annotations(data: dict) -> list[tuple[int, dict]]:
        """Return note_midi observations paired with their string index.

        JAMS stores annotations as either a list of observations or a dict of
        aligned arrays.  This handles both forms.  GuitarSet's ``note_midi``
        annotations are ordered low-E (index 1) to high-E (index 6).
        """
        results: list[tuple[int, dict]] = []
        string_index = 0
        for ann in data.get("annotations", []):
            if ann.get("namespace") != "note_midi":
                continue
            string_index += 1
            raw = ann.get("data", [])
            if isinstance(raw, list):
                for obs in raw:
                    results.append((string_index, obs))
            elif isinstance(raw, dict):
                times = raw.get("time", [])
                durations = raw.get("duration", [])
                values = raw.get("value", [])
                confidences = raw.get("confidence", [])
                for t, d, v, c in zip(times, durations, values, confidences):
                    results.append((string_index, {
                        "time": t,
                        "duration": d,
                        "value": v,
                        "confidence": c,
                    }))
        return results

    def _parse_jams(self, jams: Path, audio: Path) -> list[ClipEvent]:
        events: list[ClipEvent] = []
        with jams.open() as f:
            data = json.load(f)
        metadata = self._metadata(data, jams, audio)
        for i, (string_index, obs) in enumerate(self._note_midi_annotations(data)):
            midi_float = obs.get("value")
            if midi_float is None:
                continue
            midi = int(round(midi_float))
            if not (20 <= midi <= 100):
                continue
            start = float(obs.get("time", 0.0))
            duration = float(obs.get("duration", 0.0))
            # Annotation order is low-E (index 1) to high-E (index 6); convert
            # to schema convention where string 1 = high E.
            string = 7 - string_index if 1 <= string_index <= 6 else None
            events.append(ClipEvent(
                clip_id=f"guitarset/{jams.stem}/{i}",
                source=self.name,
                start_s=start,
                end_s=start + duration,
                midi=midi,
                notes=(),
                string=string,
                fret=None,
                technique="normal",
                confidence=1.0,
                audio_path=str(audio),
                metadata=dict(metadata),
            ))
        return group_simultaneous_notes(events)

    @staticmethod
    def _metadata(data: dict, jams: Path, audio: Path) -> dict[str, str]:
        metadata = {
            "instrument": "acoustic_guitar",
            "audio_variant": "mix" if audio.stem.endswith("_mix") else "mono",
        }
        # GuitarSet file names begin with the performer identifier. Keep the
        # raw identifier so split tooling can group a player without guessing
        # demographic information.
        performer = jams.stem.split("_", 1)[0].strip()
        if performer:
            metadata["player"] = performer
        file_metadata = data.get("file_metadata", {})
        if isinstance(file_metadata, dict):
            for source_key, target_key in (
                ("title", "title"),
                ("artist", "artist"),
                ("release", "session"),
            ):
                value = file_metadata.get(source_key)
                if value not in (None, ""):
                    metadata[target_key] = str(value)
        return metadata
