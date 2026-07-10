"""Stream↔song clock mapping for verifier-driven gameplay.

The audio stream advances in real time (stream time), while the chart playback
position advances at ``tempo_factor`` relative to the song (song time). Each
discontinuity (start, resume, seek, loop, tempo change, wait-mode freeze) is
recorded as a new clock segment so that windows can be fetched in stream time
and judgments reported back in song time.
"""

from __future__ import annotations

from dataclasses import dataclass


MAX_SEGMENTS = 32


@dataclass
class ClockSegment:
    """One continuous mapping between song time and stream time."""

    song_origin_ms: float
    stream_origin_ms: float
    tempo_factor: float


class StreamClock:
    """Segmented clock mapping between stream time and song time."""

    def __init__(self) -> None:
        self._segments: list[ClockSegment] = [ClockSegment(0.0, 0.0, 1.0)]

    def set_segment(
        self,
        song_origin_ms: float,
        stream_origin_ms: float,
        tempo_factor: float,
    ) -> None:
        """Create a new mapping segment. Drops oldest complete segment if over limit."""
        self._segments.append(
            ClockSegment(song_origin_ms, stream_origin_ms, tempo_factor)
        )
        # Bound growth: drop oldest COMPLETE segment (not the active one)
        if len(self._segments) > MAX_SEGMENTS:
            for i, seg in enumerate(self._segments[:-1]):  # never drop the active segment
                self._segments.pop(i)
                break

    def song_to_stream_ms(self, song_ms: float) -> float:
        """Convert song time to stream time.

        Searches backwards for the segment whose song range contains song_ms.
        The latest segment (no end bound) is the active fallback.
        """
        seg = self._segments[-1]  # default: active segment
        for i in range(len(self._segments) - 2, -1, -1):
            next_seg = self._segments[i + 1]
            if song_ms >= self._segments[i].song_origin_ms and song_ms < next_seg.song_origin_ms:
                seg = self._segments[i]
                break
        return seg.stream_origin_ms + (song_ms - seg.song_origin_ms) / seg.tempo_factor


    def stream_to_song_ms(self, stream_ms: float) -> float:
        """Convert stream time to song time using the active segment."""
        seg = self._segments[-1]
        for s in reversed(self._segments):
            if stream_ms >= s.stream_origin_ms:
                seg = s
                break
        return seg.song_origin_ms + (stream_ms - seg.stream_origin_ms) * seg.tempo_factor

    def reset(self) -> None:
        """Reset to a single identity segment."""
        self._segments = [ClockSegment(0.0, 0.0, 1.0)]
