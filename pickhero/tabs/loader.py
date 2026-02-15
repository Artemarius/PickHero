"""Guitar Pro file loader.

Parses GP3/GP4/GP5 files via pyguitarpro, builds a TempoMap for tick-to-ms
conversion, extracts note events, and produces a Timeline.
"""

from __future__ import annotations

from pathlib import Path

import guitarpro

from pickhero.tabs.timeline import NoteEvent, SongMetadata, Timeline

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
    for measure in track.measures:
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
                        )
                    )
    return notes


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


def load_gp_file(path: str | Path, track_index: int | None = None) -> Timeline:
    """Load a GP file and return a Timeline.

    Args:
        path: Path to GP3/GP4/GP5 file.
        track_index: Explicit track index to load. If None, auto-selects
                     the first guitar track (or track 0 as fallback).
    """
    song = guitarpro.parse(str(path))
    tempo_map = _build_tempo_map(song)

    # Select track
    if track_index is not None:
        track = song.tracks[track_index]
    else:
        track = song.tracks[0]  # default fallback
        for t in song.tracks:
            if is_guitar_track(t):
                track = t
                break

    notes = _extract_notes(track, tempo_map)

    metadata = SongMetadata(
        title=song.title or "",
        artist=song.artist or "",
        album=song.album or "",
        track_name=track.name or "",
        tempo=song.tempo,
        tuning=_extract_tuning(track),
    )

    return Timeline(notes, metadata)
