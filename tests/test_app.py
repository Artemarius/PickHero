"""Tests for pickhero.ui.app error reporting."""

import os

# Headless SDL: tests must not require a real display or audio device.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from pathlib import Path

import pygame
import pytest

from pickhero.config import Config
from pickhero.ui.app import App

FIXTURES = Path(__file__).parent / "fixtures"

class TestErrorBanner:
    """Verify transient error banner state."""

    @pytest.fixture(autouse=True)
    def init_pygame(self):
        pygame.init()
        yield
        pygame.quit()

    def test_show_error_sets_message_and_expiry(self):
        app = App(Config())
        app._show_error("Test error", duration_ms=1000.0)
        assert app._error_message == "Test error"
        assert app._error_expiry_ms > pygame.time.get_ticks()

    def test_update_error_clears_expired_message(self):
        app = App(Config())
        app._error_message = "Old error"
        app._error_expiry_ms = pygame.time.get_ticks() - 1.0
        app._update_error()
        assert app._error_message == ""

    def test_update_error_keeps_active_message(self):
        app = App(Config())
        app._show_error("Active error", duration_ms=5000.0)
        app._update_error()
        assert app._error_message == "Active error"

    def test_draw_error_renders_when_active(self):
        app = App(Config())
        app._show_error("Render me", duration_ms=5000.0)
        surface = pygame.Surface((640, 480))
        app._draw_error(surface)
        # Banner is centered near the top; just ensure it doesn't raise.

    def test_draw_error_no_op_when_inactive(self):
        app = App(Config())
        surface = pygame.Surface((640, 480))
        app._draw_error(surface)  # Should not raise


class TestSongLoadTransition:
    """Regression for the dc6c023 bug: `_load_song` constructed PlayingScreen
    but never set `self._state = "playing"`, so the menu kept rendering and the
    player never reached the game screen."""

    @pytest.fixture(autouse=True)
    def init_pygame(self):
        pygame.init()
        yield
        pygame.quit()

    def test_load_song_sets_state_to_playing(self):
        guitarpro = pytest.importorskip("guitarpro")
        notes_gp5 = FIXTURES / "notes.gp5"
        assert notes_gp5.exists(), f"missing fixture: {notes_gp5}"
        app = App(Config())
        app._load_song(notes_gp5, -1)
        assert app._state == "playing", f"state={app._state!r}"
        assert app._playing_screen is not None

    def test_load_song_load_failure_stays_in_menu(self):
        app = App(Config())
        app._state = "menu"
        app._load_song(Path("nonexistent_song_12345.gp"), -1)
        assert app._state == "menu", f"state={app._state!r}"

    def test_load_song_does_not_double_set_state_on_seek_error(self, monkeypatch):
        pytest.importorskip("guitarpro")
        notes_gp5 = FIXTURES / "notes.gp5"
        assert notes_gp5.exists()
        app = App(Config())
        # Force PlayingScreen.seek to raise after the screen is constructed.
        # The state assignment precedes the seek call, so it must remain "playing".
        original_init = app._load_song  # noqa: F841 (sanity anchor)

        def boom(_seek_to):
            raise RuntimeError("synthetic seek failure")

        # Patch seek on the instance after construction is impossible (screen
        # doesn't exist yet), so patch the class method to raise on first call.
        from pickhero.ui.scrolling import PlayingScreen

        monkeypatch.setattr(PlayingScreen, "seek", boom)
        # The _load_song path constructs the screen then calls seek; even if seek
        # raises, _state must already be "playing".
        try:
            app._load_song(notes_gp5, -1)
        except RuntimeError:
            pass  # seek raised — acceptable for this test
        assert app._state == "playing", f"state={app._state!r}"
        assert app._playing_screen is not None
