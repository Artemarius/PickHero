"""IDMT-SMT-Guitar dataset importer.

The dataset ships one WAV file per recording with a matching XML annotation file
that lists discrete note events (pitch, onset/offset, string, fret, playing
style).  This importer parses those XML files and emits
:class:`~pickhero.datasets.schema.ClipEvent` objects.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from pickhero.datasets.base import DatasetImporter
from pickhero.datasets.schema import ClipEvent


# IDMT stores stringNumber 1=low E; the schema uses 1=high E.
_IDMT_STRING_TO_SCHEMA = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}

_TECHNIQUE_MAP = {
    "PK": "normal",
    "FS": "normal",
    "NO": "normal",
    "BN": "bend",
    "VI": "vibrato",
    "SL": "slide",
    "PM": "palm_mute",
    "HM": "harmonic",
    "PH": "pinch_harmonic",
}


class IdmtImporter(DatasetImporter):
    """Importer for the IDMT-SMT-Guitar dataset."""

    name = "IDMT"

    def scan(self, path: str | Path) -> list[ClipEvent]:
        root = Path(path)
        if not root.exists():
            return []
        events: list[ClipEvent] = []
        for xml_file in sorted(root.rglob("*.xml")):
            try:
                events.extend(self._parse_xml(xml_file))
            except Exception:
                continue
        return events

    def _resolve_audio(self, xml_file: Path, audio_name: str | None) -> Path | None:
        """Resolve the WAV file referenced by an XML annotation."""
        if audio_name:
            candidate = xml_file.parent.parent / "audio" / audio_name
            if candidate.exists():
                return candidate
        candidate = xml_file.with_suffix(".wav")
        if candidate.exists():
            return candidate
        candidate = xml_file.parent.parent / "audio" / xml_file.with_suffix(".wav").name
        if candidate.exists():
            return candidate
        return None

    def _parse_xml(self, xml_file: Path) -> list[ClipEvent]:
        events: list[ClipEvent] = []
        tree = ET.parse(xml_file)
        root = tree.getroot()

        audio_name = None
        audio_file_el = root.find(".//audioFileName")
        if audio_file_el is not None and audio_file_el.text:
            audio_name = audio_file_el.text.strip()
        audio = self._resolve_audio(xml_file, audio_name)
        if audio is None:
            return events

        recording_metadata = {"instrument": "electric_guitar"}
        for xml_tag, metadata_key in (
            ("player", "player"),
            ("guitar", "guitar"),
            ("pickup", "pickup"),
            ("tuning", "tuning"),
        ):
            element = root.find(f".//{xml_tag}")
            if element is not None and element.text and element.text.strip():
                recording_metadata[metadata_key] = element.text.strip()

        for i, event in enumerate(root.findall(".//event")):
            pitch_el = event.find("pitch")
            onset_el = event.find("onsetSec")
            offset_el = event.find("offsetSec")
            fret_el = event.find("fretNumber")
            string_el = event.find("stringNumber")
            excitation_el = event.find("excitationStyle")
            expression_el = event.find("expressionStyle")
            if pitch_el is None or onset_el is None:
                continue
            try:
                midi = int(pitch_el.text or "")
                onset = float(onset_el.text or "0.0")
                offset = float(offset_el.text or "0.0") if offset_el is not None else onset + 0.5
                fret = int(fret_el.text or "-1") if fret_el is not None else None
                idmt_string = int(string_el.text or "-1") if string_el is not None else None
            except ValueError:
                continue
            if not (20 <= midi <= 100):
                continue

            string = _IDMT_STRING_TO_SCHEMA.get(idmt_string) if idmt_string is not None else None
            expression_code = expression_el.text if expression_el is not None else ""
            excitation_code = excitation_el.text if excitation_el is not None else ""
            technique = (
                _TECHNIQUE_MAP.get(expression_code)
                or _TECHNIQUE_MAP.get(excitation_code)
                or "normal"
            )

            events.append(ClipEvent(
                clip_id=f"idmt/{audio.stem}/{i}",
                source=self.name,
                start_s=onset,
                end_s=offset,
                midi=midi,
                notes=(),
                string=string,
                fret=fret if fret >= 0 else None,
                technique=technique,
                confidence=1.0,
                audio_path=str(audio),
                metadata=dict(recording_metadata),
            ))
        return events
