"""Pytest configuration shared across all test modules.

Ensures pygame is initialized for the entire test session with a dummy
display driver so headless CI works. Individual test modules that import
pygame no longer need to call pygame.init() or pygame.quit() themselves.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest


@pytest.fixture(scope="session", autouse=True)
def _pygame_session():
    """Initialize pygame once for the entire test session.

    Creates a minimal display surface so convert_alpha() works in font
    rendering tests.  Does NOT call pygame.quit() at session end —
    leaving pygame initialized avoids breaking test files that run later.
    """
    if not pygame.get_init():
        pygame.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1, 1))
    yield
