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
from pickhero.tabs.timeline import MeasureInfo, NoteEvent, SongMetadata, Timeline

# GP tick resolution: 960 ticks = 1 quarter note
TICKS_PER_QUARTER = 960

# MIDI program numbers for guitar instruments (General MIDI)
GUITAR_INSTRUMENT_MIN = 24  # Nylon String Guitar
GUITAR_INSTRUMENT_MAX = 31  # Guitar Harmonics


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
    """Check if a track is a guitar track.

    Criteria: 6 strings, MIDI instrument 24-31, not percussion.
    """
    return (
        len(track.strings) == 6
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


def _extract_notes(track: guitarpro.Track, tempo_map: TempoMap) -> list[NoteEvent]:
    """Extract all playable notes from a track."""
    notes = []
    for measure_idx, measure in enumerate(track.measures):
        for voice in measure.voices:
            for beat in voice.beats:
                timestamp_ms = tempo_map.tick_to_ms(beat.start)
                duration_ms = tempo_map.duration_ticks_to_ms(
                    beat.duration.time, beat.start
                )
                for note in beat.notes:
                    # Keep normal (1) and dead (3), skip rest (0) and tie (2)
                    if note.type.value not in (1, 3):
                        continue
                    notes.append(
                        NoteEvent(
                            timestamp_ms=timestamp_ms,
                            duration_ms=duration_ms,
                            midi_note=note.realValue,
                            string=note.string,
                            fret=note.value,
                            measure=measure_idx,
                        )
                    )
    return notes


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

    Returns a list of dicts with keys: index, name, strings, instrument,
    is_percussion, is_guitar.
    """
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

    # Notes: id → (fret, gp7_string)  (gp7_string is 0-indexed, 0=low E)
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
                        if f != int(f):
                            continue  # fractional = drum/percussion, skip
                        string_val = int(f)
                    except ValueError:
                        pass
            if is_dead_or_muted:
                continue
            if fret is not None and string_val is not None:
                notes_map[nid] = (fret, string_val)

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
            is_drum = (midi_prog == 1024 or
                       t.findtext(".//InstrumentSet", "") == "drums")
            # Heuristic: drums have instrumentId=1024 in the search API;
            # in GPIF the program is the GM number. Drum kits use channel 10.
            track_list.append({
                "id": t.get("id", ""),
                "name": t.findtext("Name", "").strip(),
                "tuning": tuning_pitches,
                "midi_program": midi_prog,
                "num_strings": len(tuning_pitches),
                "is_guitar": (
                    len(tuning_pitches) == 6
                    and GUITAR_INSTRUMENT_MIN <= midi_prog <= GUITAR_INSTRUMENT_MAX
                ),
                "is_percussion": is_drum,
            })

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
    tuning = sel["tuning"]  # [low_E, A, D, G, B, high_E] (0-indexed)
    num_strings = sel["num_strings"] or 6

    # Build our tuning dict: {1: high_E_midi, ..., 6: low_E_midi}
    tuning_dict: dict[int, int] = {}
    for gp7_idx, midi_val in enumerate(tuning):
        our_string = num_strings - gp7_idx  # 0→6, 1→5, ..., 5→1
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
                        fret, gp7_string = notes_map[nid]
                        our_string = num_strings - gp7_string
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
        path: Path to GP3/GP4/GP5/GP7/GP8 file.
        track_index: Explicit track index to load. If None, auto-selects
                     the first guitar track (or track 0 as fallback).
    """
    if _is_gp7_file(path):
        return _load_gp7_file(path, track_index)

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
                        if note.type.value not in (1, 3):
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
