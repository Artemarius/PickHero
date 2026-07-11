"""Tests for pickhero.ui.overlays — overlay data model and font cache."""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

from pickhero.ui.overlays import CompletionState, _FontCache
class TestCompletionState:
    def test_required_fields(self):
        state = CompletionState(
            audio_enabled=True, matcher=None, is_new_best=False,
            recommendations=[], weakest_sections=[], timing_judge=False,
            timing_summary=None, timing_worst_measures=[],
        )
        assert state.audio_enabled is True
        assert state.matcher is None
        assert state.is_new_best is False
        assert state.recommendations == []
        assert state.weakest_sections == []
        assert state.timing_judge is False
        assert state.timing_summary is None
        assert state.timing_worst_measures == []

    def test_default_technique_heatmap(self):
        state = CompletionState(
            audio_enabled=True, matcher=None, is_new_best=False,
            recommendations=[], weakest_sections=[], timing_judge=False,
            timing_summary=None, timing_worst_measures=[],
        )
        assert state.technique_heatmap == {}

    def test_default_drill_recommendation(self):
        state = CompletionState(
            audio_enabled=True, matcher=None, is_new_best=False,
            recommendations=[], weakest_sections=[], timing_judge=False,
            timing_summary=None, timing_worst_measures=[],
        )
        assert state.drill_recommendation is None

    def test_with_values(self):
        state = CompletionState(
            audio_enabled=True, matcher=None, is_new_best=True,
            recommendations=["Practice bars 3-4"],
            weakest_sections=[(3, 4, 65.0)],
            timing_judge=True, timing_summary=None,
            timing_worst_measures=[(3, 15.0, 8.0)],
            technique_heatmap={"bend": {"accuracy": 0.7, "count": 5}},
            drill_recommendation="Loop bars 4-5 at 70%",
        )
        assert state.is_new_best is True
        assert len(state.recommendations) == 1
        assert state.weakest_sections[0] == (3, 4, 65.0)
        assert "bend" in state.technique_heatmap
        assert state.drill_recommendation is not None


class TestFontCache:
    def test_init(self):
        cache = _FontCache()
        assert cache is not None

    def test_render_returns_surface(self):
        cache = _FontCache()
        font = pygame.font.Font(None, 14)
        surf = cache.render(font, "Hello", True, (255, 255, 255))
        assert isinstance(surf, pygame.Surface)

    def test_render_caches(self):
        cache = _FontCache()
        font = pygame.font.Font(None, 14)
        surf1 = cache.render(font, "Test", True, (255, 255, 255))
        surf2 = cache.render(font, "Test", True, (255, 255, 255))
        assert surf1 is surf2

    def test_render_different_text(self):
        cache = _FontCache()
        font = pygame.font.Font(None, 14)
        surf1 = cache.render(font, "Hello", True, (255, 255, 255))
        surf2 = cache.render(font, "World", True, (255, 255, 255))
        assert surf1 is not surf2

    def test_render_different_sizes(self):
        cache = _FontCache()
        font1 = pygame.font.Font(None, 12)
        font2 = pygame.font.Font(None, 24)
        surf1 = cache.render(font1, "Test", True, (255, 255, 255))
        surf2 = cache.render(font2, "Test", True, (255, 255, 255))
        assert surf1.get_height() <= surf2.get_height()
