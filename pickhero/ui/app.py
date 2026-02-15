"""PickHero application — PyGame game loop and state machine.

Two states: MENU (song selection) and PLAYING (scrolling display).
"""

from __future__ import annotations

from pathlib import Path

import pygame

from pickhero.config import Config
from pickhero.tabs.loader import load_gp_file
from pickhero.ui.menu import MenuScreen
from pickhero.ui.scrolling import PlayingScreen


class App:
    """Main application with game loop."""

    def __init__(self, config: Config | None = None):
        self._config = config or Config.load()
        self._running = False
        self._state = "menu"
        self._menu: MenuScreen | None = None
        self._playing_screen: PlayingScreen | None = None

    def run(self) -> None:
        """Initialize PyGame, run main loop, clean up."""
        pygame.init()
        pygame.display.set_caption("PickHero")

        dc = self._config.display
        surface = pygame.display.set_mode(
            (dc.width, dc.height), pygame.RESIZABLE
        )
        clock = pygame.time.Clock()

        songs_dir = Path(self._config.songs_dir)
        self._menu = MenuScreen(songs_dir)
        self._state = "menu"
        self._running = True

        while self._running:
            self._process_events(surface)
            self._update()
            self._render(surface)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    def _process_events(self, surface: pygame.Surface) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return

            if event.type == pygame.VIDEORESIZE:
                surface = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE
                )

            if self._state == "menu":
                self._handle_menu_event(event)
            elif self._state == "playing":
                self._handle_playing_event(event)

    def _handle_menu_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._running = False
            return

        result = self._menu.handle_event(event)
        if result is not None:
            self._load_song(result)

    def _handle_playing_event(self, event: pygame.event.Event) -> None:
        result = self._playing_screen.handle_event(event)
        if result == "menu":
            self._playing_screen.stop_audio()
            self._playing_screen = None
            self._state = "menu"
            self._menu.scan_files()

    def _load_song(self, path: Path) -> None:
        """Load a GP file and switch to playing state."""
        try:
            timeline = load_gp_file(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return

        dc = self._config.display
        self._playing_screen = PlayingScreen(
            timeline,
            visible_beats=dc.visible_beats,
            hit_zone_fraction=dc.hit_zone_fraction,
            config=self._config,
        )
        self._state = "playing"

    def _update(self) -> None:
        if self._state == "playing" and self._playing_screen is not None:
            self._playing_screen.update()

    def _render(self, surface: pygame.Surface) -> None:
        if self._state == "menu" and self._menu is not None:
            self._menu.render(surface)
        elif self._state == "playing" and self._playing_screen is not None:
            self._playing_screen.render(surface)
