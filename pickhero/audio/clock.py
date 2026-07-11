"""Latency-aware stream↔song clock mapping.

The capture stream is monotonic real time. Song time can pause, seek, loop and
run at a different tempo. A bounded segment history keeps both directions
consistent without deriving timestamps from detector-worker progress.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClockSegment:
    song_origin_ms: float
    stream_origin_ms: float
    tempo_factor: float
    window_factor: float = 1.0

    @property
    def frozen(self) -> bool:
        return self.tempo_factor == 0.0


class StreamClock:
    """Segmented capture clock with persistent input-latency compensation."""

    MAX_SEGMENTS = 256

    def __init__(self, latency_offset_ms: float = 0.0) -> None:
        self._latency_offset_ms = float(latency_offset_ms)
        self._segments: list[ClockSegment] = [ClockSegment(0.0, 0.0, 1.0)]

    @property
    def latency_offset_ms(self) -> float:
        return self._latency_offset_ms

    @latency_offset_ms.setter
    def latency_offset_ms(self, value: float) -> None:
        self._latency_offset_ms = float(value)

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    def set_segment(
        self,
        song_origin_ms: float,
        stream_origin_ms: float,
        tempo_factor: float,
        *,
        replace_latest: bool = False,
        window_factor: float | None = None,
    ) -> None:
        """Create or refresh the active mapping.

        ``tempo_factor=0`` represents wait-mode freeze. Stream→song then pins
        detections to the frozen chart position. Song→stream still needs a
        local slope to request an audio window, supplied by ``window_factor``
        (normally 1.0) and refreshed with ``replace_latest=True``.
        """
        tempo = float(tempo_factor)
        if tempo < 0.0:
            raise ValueError("tempo_factor must be non-negative")
        if window_factor is None:
            window_factor = tempo if tempo > 0.0 else 1.0
        if window_factor <= 0.0:
            raise ValueError("window_factor must be greater than zero")
        segment = ClockSegment(
            float(song_origin_ms),
            float(stream_origin_ms),
            tempo,
            float(window_factor),
        )
        if replace_latest:
            self._segments[-1] = segment
        else:
            latest = self._segments[-1]
            # Avoid duplicate anchors emitted by start/seek helper chains.
            if latest == segment:
                return
            self._segments.append(segment)
            if len(self._segments) > self.MAX_SEGMENTS:
                self._segments = self._segments[-self.MAX_SEGMENTS:]

    def refresh_frozen_anchor(self, song_ms: float, stream_ms: float) -> None:
        """Refresh one wait-mode segment instead of appending every frame."""
        latest = self._segments[-1]
        replace_latest = latest.frozen
        self.set_segment(
            song_ms,
            stream_ms,
            0.0,
            replace_latest=replace_latest,
            window_factor=1.0,
        )

    def song_to_stream_ms(self, song_ms: float) -> float:
        """Map chart time to the capture position containing that performance."""
        song_ms = float(song_ms)
        # Most recent segment whose song origin is at or before song_ms,
        # mirroring stream_to_song_ms. Falls back to the oldest segment.
        seg = self._segments[0]
        for candidate in reversed(self._segments):
            if song_ms >= candidate.song_origin_ms:
                seg = candidate
                break
        stream = seg.stream_origin_ms + (
            song_ms - seg.song_origin_ms
        ) / seg.window_factor
        return stream + self._latency_offset_ms

    def stream_to_song_ms(self, stream_ms: float) -> float:
        """Map a captured sample timestamp back to chart time."""
        uncompensated = float(stream_ms) - self._latency_offset_ms
        seg = self._segments[0]
        for candidate in reversed(self._segments):
            if uncompensated >= candidate.stream_origin_ms:
                seg = candidate
                break
        if seg.frozen:
            return seg.song_origin_ms
        return seg.song_origin_ms + (
            uncompensated - seg.stream_origin_ms
        ) * seg.tempo_factor

    def reset(self) -> None:
        """Reset mapping while preserving the calibrated latency offset."""
        self._segments = [ClockSegment(0.0, 0.0, 1.0)]


# Backwards-compatible module-level constant used by tests.
MAX_SEGMENTS = StreamClock.MAX_SEGMENTS
