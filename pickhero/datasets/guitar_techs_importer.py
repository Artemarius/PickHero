"""Guitar-TECHS dataset importer.

Guitar-TECHS provides DI + amp/mic audio and per-string MIDI labels for
electric-guitar techniques.  This importer parses the MIDI files and pairs them
with the ``audio/micamp/*.wav`` recordings.
"""

from __future__ import annotations

from pathlib import Path

from pickhero.datasets.base import DatasetImporter
from pickhero.datasets.grouping import group_simultaneous_notes
from pickhero.datasets.schema import ClipEvent


_TECHNIQUE_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("palm", "mute"), "palm_mute"),
    (("palmmute",), "palm_mute"),
    (("pinch", "harmonic"), "pinch_harmonic"),
    (("natural", "harmonic"), "harmonic"),
    (("harmonic",), "harmonic"),
    (("hammer", "on"), "hammer_on"),
    (("hammeron",), "hammer_on"),
    (("pull", "off"), "pull_off"),
    (("pulloff",), "pull_off"),
    (("vibrato",), "vibrato"),
    (("bend",), "bend"),
    (("slide",), "slide"),
    (("dead", "note"), "dead_note"),
    (("deadnote",), "dead_note"),
    (("tapping",), "tap"),
    (("tap",), "tap"),
)


def _technique_from_path(path: Path) -> str:
    """Infer a concrete articulation from folder and file tokens."""
    normalized = " ".join(
        part.lower().replace("_", " ").replace("-", " ")
        for part in path.parts[-5:]
    )
    words = set(normalized.split())
    compact = normalized.replace(" ", "")
    for tokens, technique in _TECHNIQUE_PATTERNS:
        if len(tokens) == 1:
            token = tokens[0]
            if token in words or token in compact:
                return technique
        elif all(token in words for token in tokens):
            return technique
    return "normal"



class GuitarTechsImporter(DatasetImporter):
    """Importer for the Guitar-TECHS dataset."""

    name = "Guitar-TECHS"

    def scan(self, path: str | Path) -> list[ClipEvent]:
        root = Path(path)
        if not root.exists():
            return []
        events: list[ClipEvent] = []
        for midi_file in sorted(root.rglob("*.mid")):
            if "__MACOSX" in midi_file.parts or midi_file.name.startswith("._"):
                continue
            audio = self._find_audio(midi_file)
            if audio is None:
                continue
            try:
                events.extend(self._parse_midi(midi_file, audio))
            except Exception:
                continue
        return events

    def _find_audio(self, midi_file: Path) -> Path | None:
        """Pair a MIDI file with its micamp WAV recording."""
        name = midi_file.stem.replace("midi_", "")
        audio_dir = midi_file.parent.parent / "audio" / "micamp"
        candidate = audio_dir / f"micamp_{name}.wav"
        return candidate if candidate.exists() else None

    def _parse_midi(self, midi_file: Path, audio: Path) -> list[ClipEvent]:
        """Parse note_on/note_off events from a type-1 MIDI file."""
        try:
            import mido
        except Exception as exc:
            raise ImportError(
                "Guitar-TECHS importer requires the optional 'mido' package."
            ) from exc

        events: list[ClipEvent] = []
        mid = mido.MidiFile(str(midi_file))
        technique = _technique_from_path(midi_file)
        tempo = 500000  # default 120 BPM microseconds per quarter
        ticks_per_beat = mid.ticks_per_beat or 960

        for track_idx, track in enumerate(mid.tracks):
            active: dict[int, tuple[float, int]] = {}
            absolute_ticks = 0
            note_idx = 0
            for msg in track:
                absolute_ticks += msg.time
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                elif msg.type == "note_on" and msg.velocity > 0:
                    active[msg.note] = (absolute_ticks, msg.velocity)
                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    if msg.note not in active:
                        continue
                    start_ticks, _ = active.pop(msg.note)
                    start_s = mido.tick2second(start_ticks, ticks_per_beat, tempo)
                    end_s = mido.tick2second(absolute_ticks, ticks_per_beat, tempo)
                    if end_s <= start_s:
                        continue
                    track_name = track.name.strip() if track.name else ""
                    string_map = {"e": 1, "B": 2, "G": 3, "D": 4, "A": 5, "E": 6}
                    string = string_map.get(track_name)
                    clip_id = f"guitar_techs/{audio.stem}/{track_idx}_{note_idx}"
                    events.append(ClipEvent(
                        clip_id=clip_id,
                        source=self.name,
                        start_s=start_s,
                        end_s=end_s,
                        midi=msg.note,
                        notes=(),
                        string=string,
                        fret=None,
                        technique=technique,
                        confidence=1.0,
                        audio_path=str(audio),
                        metadata={
                            "instrument": "electric_guitar",
                            "audio_variant": "micamp",
                            "technique_group": technique,
                            "collection": midi_file.parent.parent.name,
                        },
                    ))
                    note_idx += 1
        return group_simultaneous_notes(events)
