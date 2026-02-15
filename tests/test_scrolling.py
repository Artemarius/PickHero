"""Tests for scrolling display math.

Tests the pure computation functions in PlayingScreen without needing
a running PyGame display.
"""

import pytest

from pickhero.tabs.timeline import NoteEvent, SongMetadata, Timeline
from pickhero.ui.scrolling import (
    MIN_NOTE_WIDTH_PX,
    PlayingScreen,
    format_time,
)


def _make_timeline(tempo: int = 120, notes: list[NoteEvent] | None = None) -> Timeline:
    """Create a timeline with given tempo and notes."""
    meta = SongMetadata(title="Test", artist="Tester", tempo=tempo)
    return Timeline(notes or [], meta)


class TestFormatTime:
    def test_zero(self):
        assert format_time(0) == "0:00"

    def test_one_minute(self):
        assert format_time(60_000) == "1:00"

    def test_seconds_padded(self):
        assert format_time(5_000) == "0:05"

    def test_mixed(self):
        assert format_time(125_000) == "2:05"

    def test_negative_clamps_to_zero(self):
        assert format_time(-1000) == "0:00"

    def test_fractional_truncates(self):
        assert format_time(61_999) == "1:01"


class TestNoteX:
    """Test PlayingScreen.note_x static method."""

    def test_note_at_playback_time_sits_on_hit_zone(self):
        hit_zone_x = 256.0
        pixels_per_ms = 0.5
        playback_ms = 1000.0
        x = PlayingScreen.note_x(1000.0, playback_ms, hit_zone_x, pixels_per_ms)
        assert x == pytest.approx(hit_zone_x)

    def test_future_note_is_right_of_hit_zone(self):
        hit_zone_x = 256.0
        pixels_per_ms = 0.5
        playback_ms = 1000.0
        x = PlayingScreen.note_x(2000.0, playback_ms, hit_zone_x, pixels_per_ms)
        assert x > hit_zone_x
        assert x == pytest.approx(256.0 + 1000.0 * 0.5)

    def test_past_note_is_left_of_hit_zone(self):
        hit_zone_x = 256.0
        pixels_per_ms = 0.5
        playback_ms = 1000.0
        x = PlayingScreen.note_x(500.0, playback_ms, hit_zone_x, pixels_per_ms)
        assert x < hit_zone_x

    def test_proportional_to_time_difference(self):
        hit_zone_x = 200.0
        pixels_per_ms = 1.0
        playback_ms = 0.0
        x1 = PlayingScreen.note_x(100.0, playback_ms, hit_zone_x, pixels_per_ms)
        x2 = PlayingScreen.note_x(200.0, playback_ms, hit_zone_x, pixels_per_ms)
        assert x2 - x1 == pytest.approx(100.0)


class TestNoteWidth:
    """Test PlayingScreen.note_width static method."""

    def test_enforces_minimum(self):
        w = PlayingScreen.note_width(1.0, 0.5)  # 0.5 px, below minimum
        assert w == MIN_NOTE_WIDTH_PX

    def test_long_note_exceeds_minimum(self):
        w = PlayingScreen.note_width(500.0, 0.5)  # 250 px
        assert w == pytest.approx(250.0)

    def test_zero_duration(self):
        w = PlayingScreen.note_width(0.0, 1.0)
        assert w == MIN_NOTE_WIDTH_PX


class TestPlayingScreenLayout:
    """Test layout computation with a mock surface."""

    class _MockSurface:
        def __init__(self, w: int, h: int):
            self._size = (w, h)

        def get_size(self):
            return self._size

    def test_layout_dimensions(self):
        timeline = _make_timeline(tempo=120)
        screen = PlayingScreen(timeline, visible_beats=4, hit_zone_fraction=0.20)
        surface = self._MockSurface(1280, 720)
        layout = screen._layout(surface)

        assert layout.screen_w == 1280
        assert layout.screen_h == 720
        assert layout.hit_zone_x == pytest.approx(1280 * 0.20)

        # 120 BPM → 500ms per beat → 2000ms visible window
        expected_window = 4 * (60_000 / 120)
        assert screen._visible_window_ms == pytest.approx(expected_window)

        usable = 1280 - layout.hit_zone_x
        assert layout.usable_width == pytest.approx(usable)
        assert layout.pixels_per_ms == pytest.approx(usable / expected_window)

    def test_layout_adapts_to_different_size(self):
        timeline = _make_timeline(tempo=120)
        screen = PlayingScreen(timeline)
        layout_small = screen._layout(self._MockSurface(800, 600))
        layout_large = screen._layout(self._MockSurface(1920, 1080))

        assert layout_large.usable_width > layout_small.usable_width
        assert layout_large.lane_height > layout_small.lane_height
        assert layout_large.pixels_per_ms > layout_small.pixels_per_ms


class TestPlaybackClock:
    """Test play/pause/seek logic."""

    def test_starts_paused(self):
        screen = PlayingScreen(_make_timeline())
        assert not screen.is_playing()

    def test_toggle_play(self):
        screen = PlayingScreen(_make_timeline())
        screen.toggle_play()
        assert screen.is_playing()
        screen.toggle_play()
        assert not screen.is_playing()

    def test_seek_clamps_to_zero(self):
        screen = PlayingScreen(_make_timeline())
        screen.seek(-100)
        assert screen._playback_ms == 0.0

    def test_seek_clamps_to_duration(self):
        notes = [NoteEvent(1000, 500, 64, 1, 5)]
        timeline = _make_timeline(notes=notes)
        screen = PlayingScreen(timeline)
        screen.seek(999999)
        assert screen._playback_ms == timeline.duration_ms


class TestTempoFactor:
    """Test tempo factor (speed adjustment) logic."""

    def test_tempo_factor_default(self):
        screen = PlayingScreen(_make_timeline())
        assert screen._tempo_factor == 1.0

    def test_tempo_factor_clamps_low(self):
        screen = PlayingScreen(_make_timeline())
        screen.set_tempo_factor(0.3)
        assert screen._tempo_factor == 0.5

    def test_tempo_factor_clamps_high(self):
        screen = PlayingScreen(_make_timeline())
        screen.set_tempo_factor(1.5)
        assert screen._tempo_factor == 1.0

    def test_tempo_factor_rounds_to_nearest_005(self):
        screen = PlayingScreen(_make_timeline())
        screen.set_tempo_factor(0.73)
        assert screen._tempo_factor == pytest.approx(0.75)
        screen.set_tempo_factor(0.82)
        assert screen._tempo_factor == pytest.approx(0.80)
        screen.set_tempo_factor(0.68)
        assert screen._tempo_factor == pytest.approx(0.70)

    def test_tempo_factor_from_config(self):
        from pickhero.config import Config
        config = Config(tempo_factor=0.75)
        screen = PlayingScreen(_make_timeline(), config=config)
        assert screen._tempo_factor == pytest.approx(0.75)

    def test_tempo_factor_config_clamped_on_init(self):
        from pickhero.config import Config
        config = Config(tempo_factor=0.2)
        screen = PlayingScreen(_make_timeline(), config=config)
        assert screen._tempo_factor == 0.5
