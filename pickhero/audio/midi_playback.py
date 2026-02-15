"""Real-time MIDI playback for backing tracks.

Sends MIDI note-on/note-off events through pygame.midi.Output, driven by the
existing playback clock. No audio files or temp files needed.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field


# MIDI status bytes
NOTE_ON = 0x90
NOTE_OFF = 0x80
PROGRAM_CHANGE = 0xC0

# All-notes-off CC
ALL_NOTES_OFF_CC = 123


@dataclass(frozen=True, order=True)
class MidiEvent:
    """A single MIDI event, ordered by timestamp."""

    timestamp_ms: float
    channel: int       # 0-15
    event_type: int    # 0x90, 0x80, 0xC0
    data1: int         # note number or program number
    data2: int = 0     # velocity (ignored for program change)


class BackingTrack:
    """Sorted list of MidiEvents with cursor-based scheduling."""

    def __init__(self, events: list[MidiEvent] | None = None):
        self._events = sorted(events or [])
        self._timestamps = [e.timestamp_ms for e in self._events]
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._events)

    @property
    def events(self) -> list[MidiEvent]:
        return list(self._events)

    def get_events_until(self, time_ms: float) -> list[MidiEvent]:
        """Return events from cursor to time_ms (inclusive), advancing cursor."""
        if self._cursor >= len(self._events):
            return []
        end = bisect.bisect_right(self._timestamps, time_ms, lo=self._cursor)
        result = self._events[self._cursor:end]
        self._cursor = end
        return result

    def seek(self, time_ms: float) -> None:
        """Reset cursor to the first event at or after time_ms."""
        self._cursor = bisect.bisect_left(self._timestamps, time_ms)

    def get_program_changes_before(self, time_ms: float) -> list[MidiEvent]:
        """Return the last program change per channel before time_ms.

        Used to re-send instrument assignments after seeking.
        """
        end = bisect.bisect_left(self._timestamps, time_ms)
        latest: dict[int, MidiEvent] = {}
        for event in self._events[:end]:
            if event.event_type == PROGRAM_CHANGE:
                latest[event.channel] = event
        return list(latest.values())


class MidiPlayer:
    """Wraps pygame.midi.Output for backing track playback."""

    def __init__(self, backing_track: BackingTrack):
        self._track = backing_track
        self._output = None
        self._active_notes: set[tuple[int, int]] = set()  # (channel, note)
        self._muted = False
        self._opened = False

    def open(self) -> bool:
        """Initialize pygame.midi and open the default output device.

        Returns True on success, False if MIDI is unavailable.
        """
        try:
            import pygame.midi
            pygame.midi.init()
            device_id = pygame.midi.get_default_output_id()
            if device_id == -1:
                print("No MIDI output device found")
                return False
            self._output = pygame.midi.Output(device_id)
            self._opened = True
            return True
        except Exception as e:
            print(f"MIDI init failed: {e}")
            return False

    def play_click(self, velocity: int = 100) -> None:
        """Play a single metronome click on the MIDI percussion channel.

        Uses channel 9 (percussion), note 76 (hi wood block).
        """
        if self._output is None or self._muted:
            return
        try:
            self._output.write_short(NOTE_ON | 9, 76, velocity)
        except Exception:
            pass

    def update(self, playback_ms: float) -> None:
        """Fire all events up to playback_ms. Called each frame."""
        events = self._track.get_events_until(playback_ms)
        if not events or self._output is None:
            return
        for event in events:
            self._dispatch(event)

    def seek(self, time_ms: float) -> None:
        """Silence all notes, reset cursor, re-send program changes."""
        self._all_notes_off()
        self._track.seek(time_ms)
        # Re-send instrument assignments so the right sounds play
        if self._output is not None and not self._muted:
            for pc in self._track.get_program_changes_before(time_ms):
                self._output.write_short(
                    PROGRAM_CHANGE | pc.channel, pc.data1, 0
                )

    def pause(self) -> None:
        """Silence all active notes."""
        self._all_notes_off()

    def set_muted(self, muted: bool) -> None:
        """Toggle output. When muted, cursor still advances to avoid burst on unmute."""
        if muted and not self._muted:
            self._all_notes_off()
        self._muted = muted

    @property
    def is_muted(self) -> bool:
        return self._muted

    def close(self) -> None:
        """Silence notes and close MIDI output."""
        self._all_notes_off()
        if self._output is not None:
            try:
                self._output.close()
            except Exception:
                pass
            self._output = None
        if self._opened:
            try:
                import pygame.midi
                pygame.midi.quit()
            except Exception:
                pass
            self._opened = False

    def _dispatch(self, event: MidiEvent) -> None:
        """Send a single MIDI event to the output."""
        if event.event_type == NOTE_ON:
            self._active_notes.add((event.channel, event.data1))
            if not self._muted and self._output is not None:
                self._output.write_short(
                    NOTE_ON | event.channel, event.data1, event.data2
                )
        elif event.event_type == NOTE_OFF:
            self._active_notes.discard((event.channel, event.data1))
            if not self._muted and self._output is not None:
                self._output.write_short(
                    NOTE_OFF | event.channel, event.data1, event.data2
                )
        elif event.event_type == PROGRAM_CHANGE:
            if not self._muted and self._output is not None:
                self._output.write_short(
                    PROGRAM_CHANGE | event.channel, event.data1, 0
                )

    def _all_notes_off(self) -> None:
        """Send note-off for all tracked active notes, then CC123 on all channels."""
        if self._output is None:
            return
        # Explicit note-off for tracked notes
        for channel, note in list(self._active_notes):
            try:
                self._output.write_short(NOTE_OFF | channel, note, 0)
            except Exception:
                pass
        self._active_notes.clear()
        # CC 123 (All Notes Off) on all 16 channels as safety net
        for ch in range(16):
            try:
                self._output.write_short(0xB0 | ch, ALL_NOTES_OFF_CC, 0)
            except Exception:
                pass
