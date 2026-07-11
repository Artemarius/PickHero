"""Guitar Pro file loader.

Parses GP3/GP4/GP5 files via pyguitarpro, and GP7/GP8 files (ZIP+XML) via
stdlib xml.etree. Builds note timelines for both formats.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import guitarpro

from pickhero.audio.midi_playback import (
    NOTE_ON, NOTE_OFF, PROGRAM_CHANGE, BackingTrack, MidiEvent,
)
from pickhero.tabs import gpx
from pickhero.audio.performance import TechniqueSpec
from pickhero.tabs.timeline import MeasureInfo, NoteEvent, SongMetadata, Timeline

# GP tick resolution: 960 ticks = 1 quarter note
TICKS_PER_QUARTER = 960

# MIDI program numbers for guitar/bass instruments (General MIDI)
GUITAR_INSTRUMENT_MIN = 24  # Nylon String Guitar
GUITAR_INSTRUMENT_MAX = 39  # Electric Bass (pick), also covers Synth Bass

class TempoMap:
    """Maps absolute tick positions to milliseconds, handling tempo changes."""

    def __init__(self, initial_tempo: int):
        self._changes: list[tuple[int, int]] = [(0, initial_tempo)]

    def add_change(self, tick: int, tempo: int) -> None:
        self._changes.append((tick, tempo))
        self._changes.sort(key=lambda x: x[0])

    def tick_to_ms(self, tick: int) -> float:
        """Convert absolute tick position to milliseconds.

        Accumulates time through all tempo changes up to the target tick.
        A tempo change at the target tick does NOT affect the time up to it.
        """
        ms = 0.0
        prev_tick = 0
        prev_bpm = self._changes[0][1]

        for change_tick, change_bpm in self._changes[1:]:
            if change_tick >= tick:
                break
            delta = change_tick - prev_tick
            ms += (delta / TICKS_PER_QUARTER) * (60_000 / prev_bpm)
            prev_tick = change_tick
            prev_bpm = change_bpm

        delta = tick - prev_tick
        ms += (delta / TICKS_PER_QUARTER) * (60_000 / prev_bpm)
        return ms

    def tempo_at_tick(self, tick: int) -> int:
        """Return the active tempo (BPM) at a given tick position."""
        bpm = self._changes[0][1]
        for change_tick, change_bpm in self._changes:
            if change_tick > tick:
                break
            bpm = change_bpm
        return bpm

    def duration_ticks_to_ms(self, ticks: int, at_tick: int) -> float:
        """Convert a relative duration (in ticks) to ms using the tempo at at_tick."""
        bpm = self.tempo_at_tick(at_tick)
        return (ticks / TICKS_PER_QUARTER) * (60_000 / bpm)


def is_guitar_track(track: guitarpro.Track) -> bool:
    """Check if a track is a guitar or bass track.

    Accepts 4-8 stringed fretted instruments in the guitar/bass MIDI
    program range (24-39), excluding percussion.
    """
    return (
        4 <= len(track.strings) <= 8
        and GUITAR_INSTRUMENT_MIN <= track.channel.instrument <= GUITAR_INSTRUMENT_MAX
        and not track.channel.isPercussionChannel
    )


def _build_tempo_map(song: guitarpro.Song) -> TempoMap:
    """Scan first track for tempo changes and build a TempoMap."""
    tempo_map = TempoMap(song.tempo)
    if not song.tracks:
        return tempo_map
    track = song.tracks[0]
    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                mtc = beat.effect.mixTableChange
                if mtc and mtc.tempo:
                    tempo_map.add_change(beat.start, mtc.tempo.value)
    return tempo_map


def _extract_tuning(track: guitarpro.Track) -> dict[int, int]:
    """Extract string tuning as {string_number: midi_note}."""
    return {i + 1: s.value for i, s in enumerate(track.strings)}


def _extract_techniques(
    note,
    string: int | None = None,
    fret: int | None = None,
    tuning: dict[int, int] | None = None,
    next_fret: int | None = None,
) -> tuple[TechniqueSpec, ...]:
    """Map pyguitarpro note effects to TechniqueSpec tuples.

    A single note can carry multiple techniques (e.g. palm_mute + bend). Phase-1
    detector emits at most one TechniqueCandidate per note, but the data model
    is a tuple so compound tagging (Phase 2) needs no migration.

    Bend values are in 1/4-semitone units (value=4 -> 1 semitone = 100 cents),
    so target_cents = dest * 25.0. Harmonic type is a pyguitarpro HarmonicType
    int enum. Slides is a list of SlideType enums (not a bitmask).

    ``string``/``fret``/``tuning`` populate harmonic expected_sounding_midi
    (Patch 5c) from the open string + node ratio. ``next_fret`` populates
    slide start_fret/end_fret/target_cents (Patch 5b) — the destination fret
    is the next note's fret on the same string (pyguitarpro convention).
    """
    eff = note.effect
    specs: list[TechniqueSpec] = []
    if eff.harmonic is not None:
        # Bug fix: check hasattr on eff.harmonic, not eff.
        ht = eff.harmonic.type if hasattr(eff.harmonic, "type") else None
        subtype = _harmonic_subtype(ht)
        expected_midi, node_fret = _harmonic_expected_midi(
            subtype, string, fret, tuning,
        )
        specs.append(TechniqueSpec(
            kind="harmonic", subtype=subtype,
            expected_sounding_midi=expected_midi, node_fret=node_fret,
        ))
    if eff.palmMute:
        specs.append(TechniqueSpec(kind="palm_mute"))
    if eff.bend is not None:
        pts = [(float(p.position), float(p.value)) for p in eff.bend.points]
        origin = pts[0][1] if pts else 0.0
        dest = pts[-1][1] if pts else 0.0
        # 1/4-semitone units: value * 25 = cents (value=4 -> 100 cents = 1 semitone)
        target_cents = dest * 25.0
        subtype = _bend_subtype(origin, dest, target_cents)
        semitone_length = getattr(eff.bend, "semitoneLength", None) or 1
        specs.append(TechniqueSpec(
            kind="bend", subtype=subtype, target_cents=target_cents,
            curve=tuple((p[0] * semitone_length, p[1] * 25.0) for p in pts),
        ))
    if eff.slides:
        subtype = _slide_subtype(eff.slides)
        start_fret = fret if fret is not None else note.value
        end_fret = next_fret
        slide_target_cents: float | None = None
        if end_fret is not None:
            slide_target_cents = float((end_fret - start_fret) * 100)
        specs.append(TechniqueSpec(
            kind="slide", subtype=subtype,
            start_fret=start_fret, end_fret=end_fret,
            target_cents=slide_target_cents,
        ))
    if eff.hammer:
        # pyguitarpro encodes both hammer-ons and pull-offs as hammer=True.
        # Direction is resolved in the matcher (Step 9) from the neighbor
        # pitch delta; store a hammer_on marker with tied_to_previous=True.
        specs.append(TechniqueSpec(kind="hammer_on", tied_to_previous=True))
    if eff.vibrato:
        specs.append(TechniqueSpec(kind="vibrato"))
    # dead note: note.type.value == 3 (kept by _extract_notes, not dropped)
    if getattr(note, "type", None) is not None and note.type.value == 3:
        specs.append(TechniqueSpec(kind="dead_note"))
    return tuple(specs)


def _harmonic_expected_midi(
    subtype: str,
    string: int | None,
    fret: int | None,
    tuning: dict[int, int] | None,
) -> tuple[int | None, int | None]:
    """Compute the sounding MIDI note for a harmonic from the open string.

    Natural: open_midi + 12*log2(node_ratio). Artificial/pinch/tapped: fall
    back to open_midi + 12 (one octave up) when pyguitarpro doesn't carry a
    pitch. Returns (expected_sounding_midi, node_fret).
    """
    if string is None or fret is None or not tuning:
        return None, None
    open_midi = tuning.get(string)
    if open_midi is None:
        return None, None
    node_fret = fret if subtype == "natural" else None
    ratio = _harmonic_node_ratio(fret) if subtype == "natural" else 2.0
    if ratio <= 0:
        ratio = 2.0
    import math
    expected = open_midi + round(12 * math.log2(ratio))
    return expected, node_fret


def _harmonic_node_ratio(fret: int) -> float:
    """String-length ratio for common natural harmonic node frets."""
    if fret == 12:
        return 2.0   # octave
    if fret in (7, 19):
        return 3.0   # octave + fifth
    if fret in (5, 24):
        return 4.0   # two octaves
    if fret in (9, 16):
        return 5.0  # maj third + two octaves
    if fret in (3, 4):
        return 2.0
    return 2.0


# pyguitarpro HarmonicType enum -> our subtype string
_HARMONIC_SUBTYPES = {
    1: "natural", 2: "artificial", 3: "pinch", 4: "tapped", 5: "semi", 6: "feedback",
}


def _harmonic_subtype(ht) -> str:
    """Map a pyguitarpro HarmonicType (int enum or str) to our subtype."""
    if ht is None:
        return "natural"
    val = getattr(ht, "value", ht)
    if isinstance(val, int):
        return _HARMONIC_SUBTYPES.get(val, "natural")
    if isinstance(val, str):
        low = val.lower()
        if "pinch" in low: return "pinch"
        if "artificial" in low: return "artificial"
        if "tap" in low: return "tapped"
        if "semi" in low: return "semi"
        return "natural"
    return "natural"


def _bend_subtype(origin: float, dest: float, target_cents: float) -> str:
    """Classify a bend by its origin/dest values (1/4-semitone units)."""
    if origin > 0 and dest == 0:
        return "release"
    if origin > 0 and dest > 0:
        # pre-bend: starts already bent, holds; "bend": bent from origin to higher dest
        return "pre" if dest == origin else "bend"
    # origin == 0, dest > 0: standard upward bend; classify by target_cents
    abs_cents = abs(target_cents)
    if abs_cents < 30:
        return "quarter"
    if abs_cents < 60:
        return "half"
    if abs_cents < 90:
        return "1.5"
    if abs_cents < 150:
        return "whole"
    return "2"


# pyguitarpro SlideType enum value -> our subtype string.
# SlideType.intoFromBelow=-1, intoFromAbove=-2, shiftSlideTo=1,
# legatoSlideTo=2, outDownwards=3, outUpwards=4.
_SLIDE_SUBTYPES = {
-1: "slide_in_below", -2: "slide_in_above",
    1: "shift", 2: "legato", 3: "slide_out", 4: "slide_out",
}


def _slide_subtype(slides) -> str:
    """Map a list of pyguitarpro SlideType enums to a single subtype string.

    Precedence: shift > legato > slide_out > slide_in_* for ambiguous masks.
    """
    seen = set()
    for s in slides:
        val = getattr(s, "value", s)
        if isinstance(val, str):
            low = val.lower()
            if "shift" in low: val = 1
            elif "legato" in low: val = 2
            elif "outdown" in low or "out_down" in low: val = 3
            elif "outup" in low or "out_up" in low: val = 4
            elif "below" in low: val = -1
            elif "above" in low: val = -2
            else: continue
        seen.add(val)
    for v in (1, 2, 3, 4, -1, -2):
        if v in seen:
            return _SLIDE_SUBTYPES[v]
    return "shift"


# ── GP7/GP8 (score.gpif) technique property helpers ────────────────────────

# GPIF HarmonicType text → our subtype
_GPIF_HARMONIC_SUBTYPES = {
    "natural": "natural", "artificial": "artificial", "pinch": "pinch",
    "tap": "tapped", "tapped": "tapped", "feedback": "feedback", "semi": "semi",
}

# GPIF Slide flags → our subtype (flag names from the slundi/guitarpro schema)
_GPIF_SLIDE_SUBTYPES = {
    "shift": "shift", "legato": "legato", "slidedown": "slide_out",
    "slideup": "slide_out", "outdownwards": "slide_out", "outupwards": "slide_out",
    "intofrombelow": "slide_in_below", "intofromabove": "slide_in_above",
    "slideout": "slide_out",
}


def _gpif_harmonic_subtype(text: str) -> str:
    """Map a GPIF HarmonicType text value to our subtype string."""
    low = (text or "").strip().lower()
    if low in _GPIF_HARMONIC_SUBTYPES:
        return _GPIF_HARMONIC_SUBTYPES[low]
    if "pinch" in low: return "pinch"
    if "artificial" in low: return "artificial"
    if "tap" in low: return "tapped"
    if "feedback" in low: return "feedback"
    if "semi" in low: return "semi"
    return "natural"


def _gpif_slide_subtype(flags: list[str]) -> str:
    """Map GPIF slide flag names to a single subtype string.

    Precedence: shift > legato > slide_out > slide_in_* for ambiguous masks.
    """
    seen = set()
    for f in flags:
        low = f.strip().lower()
        for key, sub in _GPIF_SLIDE_SUBTYPES.items():
            if key in low:
                seen.add(sub)
                break
    for sub in ("shift", "legato", "slide_out", "slide_in_below", "slide_in_above"):
        if sub in seen:
            return sub
    return "shift"


def _gpif_bend_spec(n: ET.Element) -> TechniqueSpec | None:
    """Build a bend TechniqueSpec from a GPIF Note element's bend properties.

    GPIF stores bend origin/destination as float attrs on
    ``<Property name="BendDestinationValue">`` / ``BendOriginValue``.
    Values are in 1/4-semitone units (matching the GP5 convention). The target
    is the absolute bend height above the fretted note, not merely the delta
    from a pre-bent origin; using ``dest - origin`` under-graded pre-bend-and-
    bend passages and made their authored curve disagree with the target.
    """
    origin = None
    dest = None
    for prop in n.findall(".//Property"):
        pname = prop.get("name", "")
        if pname == "BendOriginValue":
            try:
                origin = float((prop.text or "0").strip())
            except (TypeError, ValueError):
                origin = 0.0
        elif pname == "BendDestinationValue":
            try:
                dest = float((prop.text or "0").strip())
            except (TypeError, ValueError):
                dest = 0.0
    if dest is None:
        return None
    if origin is None:
        origin = 0.0
    target_cents = dest * 25.0
    subtype = _bend_subtype(origin, dest, target_cents)
    return TechniqueSpec(
        kind="bend", subtype=subtype, target_cents=target_cents,
        curve=((0.0, origin * 25.0), (1.0, dest * 25.0)),
    )

def _extract_notes(track: guitarpro.Track, tempo_map: TempoMap) -> list[NoteEvent]:
    """Extract all playable notes from a track."""
    tuning = _extract_tuning(track)
    notes = []
    for measure_idx, measure in enumerate(track.measures):
        for voice in measure.voices:
            beats = voice.beats
            for beat_idx, beat in enumerate(beats):
                timestamp_ms = tempo_map.tick_to_ms(beat.start)
                duration_ms = tempo_map.duration_ticks_to_ms(
                    beat.duration.time, beat.start
                )
                for note in beat.notes:
                    # Keep normal (1), dead (3), and tie (2) notes.
                    # Tie notes enter the state machine at PITCHED (no onset
                    # required, sustain from parent note).
                    if note.type.value not in (1, 2, 3):
                        continue
                    is_tie = note.type.value == 2
                    # Slide destination: the next note on the same string in the
                    # next beat of this voice (pyguitarpro encodes slide dest
                    # as the following note's fret). None if no successor.
                    next_fret = _find_next_fret_on_string(
                        beats, beat_idx, note.string,
                    )
                    notes.append(
                        NoteEvent(
                            timestamp_ms=timestamp_ms,
                            duration_ms=duration_ms,
                            midi_note=note.realValue,
                            string=note.string,
                            fret=note.value,
                            measure=measure_idx,
                            pick_required=not is_tie,
                            techniques=_extract_techniques(
                                note,
                                string=note.string,
                                fret=note.value,
                                tuning=tuning,
                                next_fret=next_fret,
                            ),
                        )
                    )
    return notes


def _find_next_fret_on_string(
    beats: list, beat_idx: int, string: int,
) -> int | None:
    """Look ahead in the beat list for the next note on the same string.

    pyguitarpro encodes a slide's destination as the next note's fret on the
    same string in the following beat. Returns None at the end of the voice or
    if no same-string note follows.
    """
    for nb in beats[beat_idx + 1:]:
        for n in nb.notes:
            if n.type.value not in (1, 3):
                continue
            if n.string == string:
                return n.value
    return None


def _extract_measures(track: guitarpro.Track, tempo_map: TempoMap) -> list[MeasureInfo]:
    """Extract measure time ranges from a track."""
    measures = []
    for idx, measure in enumerate(track.measures):
        # Each measure has a header with start tick. We compute start/end from beats.
        beats_in_measure = []
        for voice in measure.voices:
            for beat in voice.beats:
                beats_in_measure.append(beat.start)
                beats_in_measure.append(beat.start + beat.duration.time)

        if beats_in_measure:
            start_tick = min(beats_in_measure)
            end_tick = max(beats_in_measure)
            start_ms = tempo_map.tick_to_ms(start_tick)
            end_ms = tempo_map.tick_to_ms(end_tick)
        else:
            # Empty measure — use previous end or 0
            if measures:
                start_ms = measures[-1].end_ms
            else:
                start_ms = 0.0
            end_ms = start_ms

        measures.append(MeasureInfo(index=idx, start_ms=start_ms, end_ms=end_ms))
    return measures


def list_tracks(path: str | Path) -> list[dict]:
    """List all tracks in a GP file with metadata.

    Supports GP3/GP4/GP5 via pyguitarpro and GP6/GP7/GP8 via score.gpif XML.
    """
    if _is_gp7_file(path) or _is_gpx_file(path):
        # Load the GPIF XML and list tracks without building a full timeline.
        if _is_gp7_file(path):
            with zipfile.ZipFile(str(path)) as zf:
                gpif = zf.read("Content/score.gpif").decode("utf-8")
        else:
            gpif_bytes = gpx.extract_score_gpif(path)
            if gpif_bytes is None:
                return []
            gpif = gpif_bytes.decode("utf-8", errors="replace")
        root = ET.fromstring(gpif)
        track_list = _parse_gpif_track_list(root)
        return [
            {
                "index": i,
                "name": ti["name"],
                "strings": ti["num_strings"],
                "instrument": ti["midi_program"],
                "is_percussion": ti["is_percussion"],
                "is_guitar": ti["is_guitar"],
            }
            for i, ti in enumerate(track_list)
        ]
    song = guitarpro.parse(str(path))
    tracks = []
    for i, track in enumerate(song.tracks):
        tracks.append(
            {
                "index": i,
                "name": track.name,
                "strings": len(track.strings),
                "instrument": track.channel.instrument,
                "is_percussion": track.channel.isPercussionChannel,
                "is_guitar": is_guitar_track(track),
            }
        )
    return tracks


def _is_gp7_file(path: str | Path) -> bool:
    """Check if a file is GP7/GP8 format (ZIP with Content/score.gpif)."""
    try:
        return zipfile.is_zipfile(str(path))
    except OSError:
        return False


def _is_gpx_file(path: str | Path) -> bool:
    """Check if a file is GP6 format (BCFZ/BCFS container)."""
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        return header in (b"BCFZ", b"BCFS")
    except OSError:
        return False


def _parse_gpif_track_list(root: ET.Element) -> list[dict]:
    """Parse the <Tracks> element of a score.gpif XML tree.

    Returns a list of track info dicts used by both _parse_gpif_xml and
    list_tracks. The dict layout matches the one returned by list_tracks.
    """
    track_list: list[dict] = []
    tracks_el = root.find("Tracks")
    if tracks_el is None:
        return track_list
    for t in tracks_el.findall("Track"):
        tuning_pitches: list[int] = []
        for prop in t.findall(".//Property"):
            if prop.get("name") == "Tuning":
                raw = prop.findtext("Pitches", "")
                tuning_pitches = [int(x) for x in raw.split() if x]
        midi_prog = -1
        try:
            midi_prog = int(t.findtext(".//MIDI/Program", "-1"))
        except ValueError:
            pass
        is_drum = (midi_prog == 1024 or
                   t.findtext(".//InstrumentSet", "") == "drums")
        track_list.append({
            "id": t.get("id", ""),
            "name": t.findtext("Name", "").strip(),
            "tuning": tuning_pitches,
            "midi_program": midi_prog,
            "num_strings": len(tuning_pitches),
            "is_guitar": (
                4 <= len(tuning_pitches) <= 8
                and GUITAR_INSTRUMENT_MIN <= midi_prog <= GUITAR_INSTRUMENT_MAX
            ),
            "is_percussion": is_drum,
        })
    return track_list


def _load_gpx_file(path: str | Path, track_index: int | None = None) -> Timeline:
    """Load a GP6 file (BCFZ/BCFS compressed container with score.gpif XML)."""
    gpif_bytes = gpx.extract_score_gpif(path)
    if gpif_bytes is None:
        raise ValueError(f"Could not find score.gpif in GP6 file: {path}")
    gpif = gpif_bytes.decode("utf-8", errors="replace")
    return _parse_gpif_xml(gpif, track_index)


# ── GP7/GP8 loader ──────────────────────────────────────────────────────────

# Rhythm note value → duration in quarter notes
_GP7_NOTE_VALUES = {
    "Whole": 4.0, "Half": 2.0, "Quarter": 1.0, "Eighth": 0.5,
    "16th": 0.25, "32nd": 0.125, "64th": 0.0625,
}


def _load_gp7_file(path: str | Path, track_index: int | None = None) -> Timeline:
    """Load a GP7/GP8 file (ZIP containing Content/score.gpif XML)."""
    with zipfile.ZipFile(str(path)) as zf:
        gpif = zf.read("Content/score.gpif").decode("utf-8")
    return _parse_gpif_xml(gpif, track_index)


def _parse_gpif_xml(gpif: str, track_index: int | None = None) -> Timeline:
    """Parse a ``score.gpif`` XML string into a Timeline.

    Shared by the GP7/GP8 (ZIP container) and GP6 (BCFZ/BCFS container)
    loaders — both formats use the same GPIF XML schema.
    """
    root = ET.fromstring(gpif)

    # ── Parse lookup tables ──────────────────────────────────────────────

    # Rhythms: id → duration in quarter notes
    rhythms: dict[str, float] = {}
    rhythms_el = root.find("Rhythms")
    if rhythms_el is not None:
        for r in rhythms_el.findall("Rhythm"):
            rid = r.get("id", "")
            nv = r.findtext("NoteValue", "Quarter")
            base = _GP7_NOTE_VALUES.get(nv, 1.0)
            dot = r.find("AugmentationDot")
            if dot is not None:
                dot_count = int(dot.get("count", "1"))
                if dot_count == 1:
                    base *= 1.5
                elif dot_count >= 2:
                    base *= 1.75
            tuplet = r.find("PrimaryTuplet")
            if tuplet is not None:
                num = int(tuplet.get("num", "1"))
                den = int(tuplet.get("den", "1"))
                if num > 0:
                    base = base * den / num
            rhythms[rid] = base

    # Notes: id → (fret, gp7_string, [TechniqueSpec, ...])
    # (gp7_string is 0-indexed, 0=low E). Dead/Muted notes are kept and tagged
    # with a dead_note spec so the matcher can grade them as percussive targets.
    notes_map: dict[str, tuple[int, int, list[TechniqueSpec]]] = {}
    notes_el = root.find("Notes")
    if notes_el is not None:
        for n in notes_el.findall("Note"):
            nid = n.get("id", "")
            fret = None
            string_val = None
            techniques: list[TechniqueSpec] = []
            for prop in n.findall(".//Property"):
                pname = prop.get("name", "")
                enabled = prop.get("enable") == "true"
                if pname == "Fret":
                    try:
                        fret = int(prop.findtext("Fret", "0"))
                    except ValueError:
                        pass
                elif pname == "String":
                    try:
                        raw = prop.findtext("String", "0")
                        f = float(raw)
                        if f != int(f):
                            continue  # fractional = drum/percussion, skip
                        string_val = int(f)
                    except ValueError:
                        pass
                elif pname in ("Dead", "Muted"):
                    if enabled:
                        techniques.append(TechniqueSpec(kind="dead_note"))
                elif pname == "PalmMuted":
                    if enabled:
                        techniques.append(TechniqueSpec(kind="palm_mute"))
                elif pname == "Vibrato":
                    if enabled:
                        techniques.append(TechniqueSpec(kind="vibrato"))
                elif pname == "HopoOrigin" or pname == "HopoDestination":
                    if enabled:
                        # hammer/pull marker; direction resolved in the matcher
                        techniques.append(
                            TechniqueSpec(kind="hammer_on", tied_to_previous=True)
                        )
                elif pname in ("BendOriginValue", "BendDestinationValue"):
                    # Bend: collect origin/dest; final spec built when dest seen
                    pass  # handled below in a second pass over the note element
                elif pname == "Slide":
                    if enabled:
                        # Slide flags as a whitespace-separated list of types
                        flags_text = prop.findtext("Flags", "") or prop.text or ""
                        flags = flags_text.split()
                        subtype = _gpif_slide_subtype(flags)
                        techniques.append(TechniqueSpec(kind="slide", subtype=subtype))
                elif pname == "HarmonicType":
                    ht_text = (prop.findtext("HarmonicType", "")
                               or prop.text or "").strip()
                    subtype = _gpif_harmonic_subtype(ht_text)
                    techniques.append(TechniqueSpec(kind="harmonic", subtype=subtype))
            # Second pass for bend (needs both origin and dest values).
            bend_spec = _gpif_bend_spec(n)
            if bend_spec is not None:
                techniques.append(bend_spec)
            if fret is not None and string_val is not None:
                notes_map[nid] = (fret, string_val, techniques)

    # Beats: id → (rhythm_ref, [note_ids])
    beats_map: dict[str, tuple[str, list[str]]] = {}
    beats_el = root.find("Beats")
    if beats_el is not None:
        for b in beats_el.findall("Beat"):
            bid = b.get("id", "")
            rhythm_ref_el = b.find("Rhythm")
            rhythm_ref = rhythm_ref_el.get("ref", "") if rhythm_ref_el is not None else ""
            notes_text = b.findtext("Notes", "").strip()
            note_ids = notes_text.split() if notes_text else []
            beats_map[bid] = (rhythm_ref, note_ids)

    # Voices: id → [beat_ids]
    voices_map: dict[str, list[str]] = {}
    voices_el = root.find("Voices")
    if voices_el is not None:
        for v in voices_el.findall("Voice"):
            vid = v.get("id", "")
            beats_text = v.findtext("Beats", "").strip()
            voices_map[vid] = beats_text.split() if beats_text else []

    # Bars: id → [voice_ids]
    bars_map: dict[str, list[str]] = {}
    bars_el = root.find("Bars")
    if bars_el is not None:
        for bar in bars_el.findall("Bar"):
            bid = bar.get("id", "")
            voices_text = bar.findtext("Voices", "").strip()
            bars_map[bid] = voices_text.split() if voices_text else []

    # ── Parse tracks ─────────────────────────────────────────────────────

    track_list = _parse_gpif_track_list(root)

    # ── Select track ─────────────────────────────────────────────────────

    if track_index is not None:
        selected_index = track_index
    else:
        selected_index = 0
        for i, ti in enumerate(track_list):
            if ti["is_guitar"]:
                selected_index = i
                break

    sel = track_list[selected_index]
    tuning = sel["tuning"]  # [high_E, B, G, D, A, low_E] (0-indexed)
    num_strings = sel["num_strings"] or 6

    # Build our tuning dict using the same convention as GP3/GP4/GP5:
    # string 1 = high E (top lane), string 6 = low E (bottom lane).
    tuning_dict: dict[int, int] = {}
    for gp7_idx, midi_val in enumerate(tuning):
        our_string = gp7_idx + 1  # 0→1, 1→2, ..., 5→6
        tuning_dict[our_string] = midi_val

    # ── Parse tempos ─────────────────────────────────────────────────────

    tempos: dict[int, float] = {}  # master_bar_index → BPM
    for auto in root.findall(".//MasterTrack/Automations/Automation"):
        if auto.findtext("Type") == "Tempo":
            bar_idx = int(auto.findtext("Bar", "0"))
            val = auto.findtext("Value", "120 2")
            bpm = float(val.split()[0])
            tempos[bar_idx] = bpm

    initial_bpm = tempos.get(0, 120.0)

    # ── Walk MasterBars → extract notes and measures ─────────────────────

    master_bars = root.findall(".//MasterBar")
    current_bpm = initial_bpm
    current_ms = 0.0
    note_events: list[NoteEvent] = []
    measure_infos: list[MeasureInfo] = []

    for mb_idx, mb in enumerate(master_bars):
        if mb_idx in tempos:
            current_bpm = tempos[mb_idx]

        time_sig = mb.findtext("Time", "4/4")
        parts = time_sig.split("/")
        ts_num = int(parts[0]) if len(parts) == 2 else 4
        ts_den = int(parts[1]) if len(parts) == 2 else 4

        measure_start_ms = current_ms
        measure_duration_ms = ts_num * (4.0 / ts_den) * (60_000.0 / current_bpm)

        # Get bar IDs for this master bar (one per track)
        bar_ids_text = mb.findtext("Bars", "").strip()
        bar_ids = bar_ids_text.split()

        if selected_index < len(bar_ids):
            bar_id = bar_ids[selected_index]
            voice_ids = bars_map.get(bar_id, [])

            # Walk voices → beats → notes
            for vid in voice_ids:
                if vid == "-1":
                    continue
                beat_ids = voices_map.get(vid, [])
                beat_pos_ms = current_ms

                for bid in beat_ids:
                    rhythm_ref, note_ids = beats_map.get(bid, ("", []))
                    dur_quarters = rhythms.get(rhythm_ref, 1.0)
                    dur_ms = dur_quarters * (60_000.0 / current_bpm)

                    for nid in note_ids:
                        if nid not in notes_map:
                            continue
                        fret, gp7_string, note_techniques = notes_map[nid]
                        our_string = gp7_string + 1  # match GP5 convention: 1=high E
                        if not 1 <= our_string <= 6:
                            continue
                        if gp7_string < len(tuning):
                            midi_note = tuning[gp7_string] + fret
                        else:
                            midi_note = fret  # fallback

                        if not 0 <= midi_note <= 127:
                            continue

                        note_events.append(NoteEvent(
                            timestamp_ms=beat_pos_ms,
                            duration_ms=dur_ms,
                            midi_note=midi_note,
                            string=our_string,
                            fret=fret,
                            measure=mb_idx,
                            techniques=tuple(note_techniques),
                        ))

                    beat_pos_ms += dur_ms

        measure_infos.append(MeasureInfo(
            index=mb_idx,
            start_ms=measure_start_ms,
            end_ms=measure_start_ms + measure_duration_ms,
        ))
        current_ms += measure_duration_ms

    # ── Build metadata and timeline ──────────────────────────────────────

    title = root.findtext(".//Score/Title", "").strip()
    artist = root.findtext(".//Score/Artist", "").strip()
    album = root.findtext(".//Score/Album", "").strip()

    metadata = SongMetadata(
        title=title,
        artist=artist,
        album=album,
        track_name=sel["name"],
        tempo=int(initial_bpm),
        tuning=tuning_dict,
        num_strings=num_strings,
        track_index=selected_index,
    )

    return Timeline(note_events, metadata, measures=measure_infos)


def _extract_gp7_backing_track(
    path: str | Path,
    exclude_track_indices: set[int] | None = None,
) -> BackingTrack:
    """Extract non-guitar tracks from a GP7/GP8 file as MIDI events."""
    with zipfile.ZipFile(str(path)) as zf:
        gpif = zf.read("Content/score.gpif").decode("utf-8")

    root = ET.fromstring(gpif)

    # Reuse the same lookup tables as _load_gp7_file
    # Rhythms
    rhythms: dict[str, float] = {}
    rhythms_el = root.find("Rhythms")
    if rhythms_el is not None:
        for r in rhythms_el.findall("Rhythm"):
            rid = r.get("id", "")
            nv = r.findtext("NoteValue", "Quarter")
            base = _GP7_NOTE_VALUES.get(nv, 1.0)
            dot = r.find("AugmentationDot")
            if dot is not None:
                dot_count = int(dot.get("count", "1"))
                if dot_count == 1:
                    base *= 1.5
                elif dot_count >= 2:
                    base *= 1.75
            tuplet = r.find("PrimaryTuplet")
            if tuplet is not None:
                num = int(tuplet.get("num", "1"))
                den = int(tuplet.get("den", "1"))
                if num > 0:
                    base = base * den / num
            rhythms[rid] = base

    # Notes: id → (fret, gp7_string)
    notes_map: dict[str, tuple[int, int]] = {}
    notes_el = root.find("Notes")
    if notes_el is not None:
        for n in notes_el.findall("Note"):
            nid = n.get("id", "")
            fret = None
            string_val = None
            is_dead_or_muted = False
            for prop in n.findall(".//Property"):
                pname = prop.get("name", "")
                if pname in ("Dead", "Muted"):
                    if prop.get("enable") == "true":
                        is_dead_or_muted = True
                        break
                if pname == "Fret":
                    try:
                        fret = int(prop.findtext("Fret", "0"))
                    except ValueError:
                        pass
                elif pname == "String":
                    try:
                        raw = prop.findtext("String", "0")
                        f = float(raw)
                        string_val = int(round(f))
                    except ValueError:
                        pass
            if is_dead_or_muted:
                continue
            if fret is not None and string_val is not None:
                notes_map[nid] = (fret, string_val)

    # Beats
    beats_map: dict[str, tuple[str, list[str]]] = {}
    beats_el = root.find("Beats")
    if beats_el is not None:
        for b in beats_el.findall("Beat"):
            bid = b.get("id", "")
            rhythm_ref_el = b.find("Rhythm")
            rhythm_ref = rhythm_ref_el.get("ref", "") if rhythm_ref_el is not None else ""
            notes_text = b.findtext("Notes", "").strip()
            beats_map[bid] = (rhythm_ref, notes_text.split() if notes_text else [])

    # Voices
    voices_map: dict[str, list[str]] = {}
    voices_el = root.find("Voices")
    if voices_el is not None:
        for v in voices_el.findall("Voice"):
            vid = v.get("id", "")
            beats_text = v.findtext("Beats", "").strip()
            voices_map[vid] = beats_text.split() if beats_text else []

    # Bars
    bars_map: dict[str, list[str]] = {}
    bars_el = root.find("Bars")
    if bars_el is not None:
        for bar in bars_el.findall("Bar"):
            bid = bar.get("id", "")
            voices_text = bar.findtext("Voices", "").strip()
            bars_map[bid] = voices_text.split() if voices_text else []

    # Tracks
    track_list: list[dict] = []
    tracks_el = root.find("Tracks")
    if tracks_el is not None:
        for t in tracks_el.findall("Track"):
            tuning_pitches: list[int] = []
            for prop in t.findall(".//Property"):
                if prop.get("name") == "Tuning":
                    raw = prop.findtext("Pitches", "")
                    tuning_pitches = [int(x) for x in raw.split() if x]
            midi_prog = -1
            try:
                midi_prog = int(t.findtext(".//MIDI/Program", "-1"))
            except ValueError:
                pass
            track_list.append({
                "tuning": tuning_pitches,
                "midi_program": midi_prog,
                "num_strings": len(tuning_pitches),
                "is_guitar": (
                    len(tuning_pitches) == 6
                    and GUITAR_INSTRUMENT_MIN <= midi_prog <= GUITAR_INSTRUMENT_MAX
                ),
            })

    if exclude_track_indices is None:
        exclude_track_indices = {
            i for i, ti in enumerate(track_list) if ti["is_guitar"]
        }

    # Tempos
    tempos: dict[int, float] = {}
    for auto in root.findall(".//MasterTrack/Automations/Automation"):
        if auto.findtext("Type") == "Tempo":
            bar_idx = int(auto.findtext("Bar", "0"))
            val = auto.findtext("Value", "120 2")
            tempos[bar_idx] = float(val.split()[0])

    initial_bpm = tempos.get(0, 120.0)

    # Walk master bars and extract MIDI events for non-excluded tracks
    master_bars = root.findall(".//MasterBar")
    events: list[MidiEvent] = []

    # Program changes at t=0
    for i, ti in enumerate(track_list):
        if i in exclude_track_indices:
            continue
        if ti["midi_program"] >= 0:
            channel = i % 16
            events.append(MidiEvent(
                timestamp_ms=0.0,
                channel=channel,
                event_type=PROGRAM_CHANGE,
                data1=ti["midi_program"],
                data2=0,
            ))

    current_bpm = initial_bpm
    current_ms = 0.0

    for mb_idx, mb in enumerate(master_bars):
        if mb_idx in tempos:
            current_bpm = tempos[mb_idx]

        time_sig = mb.findtext("Time", "4/4")
        parts = time_sig.split("/")
        ts_num = int(parts[0]) if len(parts) == 2 else 4
        ts_den = int(parts[1]) if len(parts) == 2 else 4

        bar_ids_text = mb.findtext("Bars", "").strip()
        bar_ids = bar_ids_text.split()

        for track_idx, bar_id in enumerate(bar_ids):
            if track_idx in exclude_track_indices:
                continue
            if track_idx >= len(track_list):
                continue

            ti = track_list[track_idx]
            tuning = ti["tuning"]
            channel = track_idx % 16

            voice_ids = bars_map.get(bar_id, [])
            for vid in voice_ids:
                if vid == "-1":
                    continue
                beat_ids = voices_map.get(vid, [])
                beat_pos_ms = current_ms

                for bid in beat_ids:
                    rhythm_ref, note_ids = beats_map.get(bid, ("", []))
                    dur_quarters = rhythms.get(rhythm_ref, 1.0)
                    dur_ms = dur_quarters * (60_000.0 / current_bpm)

                    for nid in note_ids:
                        if nid not in notes_map:
                            continue
                        fret, gp7_string = notes_map[nid]
                        if gp7_string < len(tuning):
                            midi_note = tuning[gp7_string] + fret
                        else:
                            midi_note = fret
                        midi_note = max(0, min(127, midi_note))

                        events.append(MidiEvent(
                            timestamp_ms=beat_pos_ms,
                            channel=channel,
                            event_type=NOTE_ON,
                            data1=midi_note,
                            data2=80,
                        ))
                        events.append(MidiEvent(
                            timestamp_ms=beat_pos_ms + dur_ms,
                            channel=channel,
                            event_type=NOTE_OFF,
                            data1=midi_note,
                            data2=0,
                        ))

                    beat_pos_ms += dur_ms

        measure_dur = ts_num * (4.0 / ts_den) * (60_000.0 / current_bpm)
        current_ms += measure_dur

    return BackingTrack(events)


def load_gp_file(path: str | Path, track_index: int | None = None) -> Timeline:
    """Load a GP file and return a Timeline.

    Args:
        path: Path to GP3/GP4/GP5/GP6/GP7/GP8 file.
        track_index: Explicit track index to load. If None, auto-selects
                     the first guitar track (or track 0 as fallback).
    """
    if _is_gp7_file(path):
        return _load_gp7_file(path, track_index)
    if _is_gpx_file(path):
        return _load_gpx_file(path, track_index)

    song = guitarpro.parse(str(path))
    tempo_map = _build_tempo_map(song)

    # Select track
    if track_index is not None:
        selected_index = track_index
        track = song.tracks[track_index]
    else:
        selected_index = 0
        track = song.tracks[0]  # default fallback
        for i, t in enumerate(song.tracks):
            if is_guitar_track(t):
                selected_index = i
                track = t
                break

    notes = _extract_notes(track, tempo_map)
    measures = _extract_measures(track, tempo_map)
    metadata = SongMetadata(
        title=song.title or "",
        artist=song.artist or "",
        album=song.album or "",
        track_name=track.name or "",
        tempo=song.tempo,
        tuning=_extract_tuning(track),
        num_strings=len(track.strings),
        track_index=selected_index,
    )

    return Timeline(notes, metadata, measures=measures)


def extract_backing_track(
    path: str | Path,
    exclude_track_indices: set[int] | None = None,
) -> BackingTrack:
    """Extract non-guitar tracks from a GP file as MIDI events.

    Args:
        path: Path to GP3/GP4/GP5/GP7/GP8 file.
        exclude_track_indices: Track indices to exclude. If None, auto-excludes
                               all guitar tracks.

    Returns:
        BackingTrack with note_on, note_off, and program_change events.
    """
    if _is_gp7_file(path):
        return _extract_gp7_backing_track(path, exclude_track_indices)
    if _is_gpx_file(path):
        # GP6 backing tracks are not implemented yet; the GPIF schema stores
        # them differently than GP7 and a dedicated parser is not in scope.
        return BackingTrack([])

    song = guitarpro.parse(str(path))
    tempo_map = _build_tempo_map(song)

    if exclude_track_indices is None:
        exclude_track_indices = {
            i for i, t in enumerate(song.tracks) if is_guitar_track(t)
        }

    events: list[MidiEvent] = []

    for i, track in enumerate(song.tracks):
        if i in exclude_track_indices:
            continue

        # Determine MIDI channel: percussion on channel 9, others use GP channel
        if track.channel.isPercussionChannel:
            channel = 9
        else:
            channel = (track.channel.channel - 1) % 16  # GP is 1-indexed

        # Program change at t=0 for non-percussion tracks
        if not track.channel.isPercussionChannel:
            events.append(MidiEvent(
                timestamp_ms=0.0,
                channel=channel,
                event_type=PROGRAM_CHANGE,
                data1=track.channel.instrument,
                data2=0,
            ))

        # Extract note events
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    timestamp_ms = tempo_map.tick_to_ms(beat.start)
                    duration_ms = tempo_map.duration_ticks_to_ms(
                        beat.duration.time, beat.start,
                    )

                    for note in beat.notes:
                        if note.type.value not in (1, 2, 3):
                            continue

                        velocity = note.velocity if note.velocity > 0 else 80
                        midi_note = note.realValue

                        events.append(MidiEvent(
                            timestamp_ms=timestamp_ms,
                            channel=channel,
                            event_type=NOTE_ON,
                            data1=midi_note,
                            data2=velocity,
                        ))
                        events.append(MidiEvent(
                            timestamp_ms=timestamp_ms + duration_ms,
                            channel=channel,
                            event_type=NOTE_OFF,
                            data1=midi_note,
                            data2=0,
                        ))

    return BackingTrack(events)
