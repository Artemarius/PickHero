"""Tests for pickhero.audio.clock.StreamClock."""

import pytest

from pickhero.audio.clock import MAX_SEGMENTS, StreamClock


class TestStreamClock:
    def test_identity_mapping(self):
        clock = StreamClock()
        assert clock.song_to_stream_ms(1000.0) == pytest.approx(1000.0)
        assert clock.stream_to_song_ms(1000.0) == pytest.approx(1000.0)

    def test_half_tempo_stretches_stream(self):
        clock = StreamClock()
        clock.set_segment(0.0, 0.0, 0.5)
        # Song time advances at half speed relative to stream time.
        assert clock.song_to_stream_ms(1000.0) == pytest.approx(2000.0)
        assert clock.stream_to_song_ms(2000.0) == pytest.approx(1000.0)

    def test_tempo_change_preserves_old_segment(self):
        clock = StreamClock()
        # First segment: 1.0 tempo, song 0..5000 == stream 0..5000
        clock.set_segment(5000.0, 5000.0, 0.5)
        # A note emitted before the change (stream 3000) maps back with old segment.
        assert clock.stream_to_song_ms(3000.0) == pytest.approx(3000.0)
        # A note emitted after the change: song 6000 -> stream 5000 + 2000 = 7000
        assert clock.song_to_stream_ms(6000.0) == pytest.approx(7000.0)

    def test_wait_mode_freeze_returns_recent_stream_time(self):
        clock = StreamClock()
        clock.set_segment(0.0, 0.0, 1.0)
        # Player reaches 5000ms and freezes; stream keeps advancing.
        clock.set_segment(5000.0, 7000.0, 1.0)
        # The frozen song position maps to the current stream origin.
        assert clock.song_to_stream_ms(5000.0) == pytest.approx(7000.0)
        # Unfrozen and slowed.
        clock.set_segment(5000.0, 7000.0, 0.5)
        assert clock.song_to_stream_ms(6000.0) == pytest.approx(9000.0)

    def test_reset_clears_segments(self):
        clock = StreamClock()
        clock.set_segment(5000.0, 5000.0, 0.5)
        clock.reset()
        assert clock.song_to_stream_ms(1000.0) == pytest.approx(1000.0)

    def test_song_to_stream_ms_searches_historical_segments(self):
        """song_to_stream_ms must find the right historical segment, not always the latest."""
        clock = StreamClock()
        # Segment 0 (initial): song 0..5000, tempo 1.0, stream 0..5000
        clock.set_segment(5000.0, 5000.0, 0.5)
        # Segment 1: song 5000..10000, tempo 0.5, stream 5000..
        clock.set_segment(10000.0, 15000.0, 1.0)
        # Segment 2 (active): song 10000+, tempo 1.0, stream 15000+

        # A song time in segment 0's range (0..5000) should use segment 0
        # With tempo 1.0: stream = 0 + (2500 - 0) / 1.0 = 2500
        assert clock.song_to_stream_ms(2500.0) == pytest.approx(2500.0)

        # A song time in segment 1's range (5000..10000) should use segment 1
        # With tempo 0.5: stream = 5000 + (7500 - 5000) / 0.5 = 5000 + 5000 = 10000
        assert clock.song_to_stream_ms(7500.0) == pytest.approx(10000.0)

        # A song time in segment 2's range (10000+) should use segment 2 (active)
        # With tempo 1.0: stream = 15000 + (12000 - 10000) / 1.0 = 17000
        assert clock.song_to_stream_ms(12000.0) == pytest.approx(17000.0)

    def test_song_to_stream_ms_uses_active_segment_as_fallback(self):
        """When song_ms is beyond all segment origins, the latest (active) segment is used."""
        clock = StreamClock()
        clock.set_segment(5000.0, 5000.0, 1.0)
        # song_ms=6000 is >= 5000 (segment 1 origin) and there's no segment after it
        # So it falls through to the active segment
        assert clock.song_to_stream_ms(6000.0) == pytest.approx(6000.0)

    def test_set_segment_bounds_to_max_segments(self):
        """set_segment drops oldest complete segment when count exceeds MAX_SEGMENTS."""
        clock = StreamClock()
        # Initial segment counts as 1. Add MAX_SEGMENTS more to reach MAX_SEGMENTS+1 total.
        for i in range(MAX_SEGMENTS):
            song = float(i + 1) * 1000.0
            clock.set_segment(song, song, 1.0)
        # Total segments should be MAX_SEGMENTS (dropped the oldest complete one)
        assert len(clock._segments) == MAX_SEGMENTS
        # The active (last) segment must never be dropped
        assert clock._segments[-1].song_origin_ms == float(MAX_SEGMENTS) * 1000.0
        # The initial segment (song_origin=0.0) should have been dropped
        assert clock._segments[0].song_origin_ms != 0.0

    def test_set_segment_never_drops_active_segment(self):
        """Even when over limit, the active (last) segment is always preserved."""
        clock = StreamClock()
        for i in range(MAX_SEGMENTS + 5):
            song = float(i + 1) * 1000.0
            clock.set_segment(song, song, 0.5)
        assert len(clock._segments) == MAX_SEGMENTS
        # Active segment is the last one we added
        expected_last_song = float(MAX_SEGMENTS + 5) * 1000.0
        assert clock._segments[-1].song_origin_ms == expected_last_song

    def test_wait_mode_freeze_tempo_zero(self):
        """Freeze segment with tempo_factor=0.0: stream advances, song frozen.

        stream_to_song_ms multiplies by tempo_factor, so any stream time after
        the freeze origin maps back to the frozen song position.
        """
        clock = StreamClock()
        # Song plays normally to 5000ms (stream also at 5000ms)
        clock.set_segment(5000.0, 5000.0, 0.0)  # freeze: tempo=0
        # Stream advances to 8000ms, but song stays frozen at 5000ms
        assert clock.stream_to_song_ms(8000.0) == pytest.approx(5000.0)
        assert clock.stream_to_song_ms(9999.0) == pytest.approx(5000.0)
