"""Tests for pickhero.ui.app error reporting."""

import pygame
import pytest

from pickhero.config import Config
from pickhero.ui.app import App


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
