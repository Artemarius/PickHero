"""PickHero application — PyGame game loop and state machine.

Two states: MENU (song selection) and PLAYING (scrolling display).
"""

from __future__ import annotations

from pathlib import Path

import pygame

from pickhero.audio.input import validate_device_index
from pickhero.config import Config
from pickhero.progress import ProgressTracker
from pickhero.tabs.loader import extract_backing_track, load_gp_file
from pickhero.ui.colors import set_theme
from pickhero.ui.device_menu import DeviceMenuScreen
from pickhero.ui.menu import MenuScreen
from pickhero.ui.scrolling import PlayingScreen


class App:
    """Main application with game loop."""

    def __init__(self, config: Config | None = None):
        self._config = config or Config.load()
        self._progress = ProgressTracker()
        self._running = False
        self._state = "menu"
        self._menu: MenuScreen | None = None
        self._playing_screen: PlayingScreen | None = None
        self._device_menu: DeviceMenuScreen | None = None

    def run(self) -> None:
        """Initialize PyGame, run main loop, clean up."""
        # Apply saved theme
        set_theme(self._config.theme)

        # Validate saved audio device — fall back to default if unavailable
        if not validate_device_index(self._config.audio.device_index):
            print(
                f"Saved audio device #{self._config.audio.device_index} not available, "
                "falling back to system default."
            )
            self._config.audio.device_index = None
            self._config.save()

        pygame.init()
        pygame.display.set_caption("PickHero")

        dc = self._config.display
        surface = pygame.display.set_mode(
            (dc.width, dc.height), pygame.RESIZABLE
        )
        clock = pygame.time.Clock()

        songs_dir = Path(self._config.songs_dir)
        self._menu = MenuScreen(songs_dir, config=self._config, progress=self._progress)
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
            elif self._state == "device":
                self._handle_device_event(event)

    def _handle_menu_event(self, event: pygame.event.Event) -> None:
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_d
            and not self._menu.is_searching
        ):
            self._device_menu = DeviceMenuScreen(self._config)
            self._state = "device"
            return

        result = self._menu.handle_event(event)
        if result == "escape":
            self._running = False
        elif isinstance(result, Path):
            self._load_song(result)

    def _handle_playing_event(self, event: pygame.event.Event) -> None:
        result = self._playing_screen.handle_event(event)
        if result == "menu":
            self._playing_screen.stop_audio()
            self._playing_screen = None
            self._state = "menu"
            self._menu.scan_files()

    def _handle_device_event(self, event: pygame.event.Event) -> None:
        result = self._device_menu.handle_event(event)
        if result in ("back", "selected"):
            if result == "selected":
                self._menu.refresh_device_name()
            self._device_menu = None
            self._state = "menu"

    def _load_song(self, path: Path) -> None:
        """Load a GP file and switch to playing state."""
        try:
            timeline = load_gp_file(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return

        # Extract backing track (non-guitar tracks as MIDI)
        backing_track = None
        try:
            backing_track = extract_backing_track(
                path, exclude_track_indices={timeline.metadata.track_index},
            )
        except Exception as e:
            print(f"Backing track extraction failed: {e}")

        dc = self._config.display
        self._playing_screen = PlayingScreen(
            timeline,
            visible_beats=dc.visible_beats,
            hit_zone_fraction=dc.hit_zone_fraction,
            config=self._config,
            backing_track=backing_track,
            progress_tracker=self._progress,
            song_key=path.stem,
        )
        self._state = "playing"

        # Skip ahead so the first note is just entering the visible window
        if timeline.notes:
            first_note_ms = timeline.notes[0].timestamp_ms
            seek_to = max(0.0, first_note_ms - self._playing_screen._visible_window_ms)
            if seek_to > 0:
                self._playing_screen.seek(seek_to)

    def _update(self) -> None:
        if self._state == "playing" and self._playing_screen is not None:
            self._playing_screen.update()

    def _render(self, surface: pygame.Surface) -> None:
        if self._state == "menu" and self._menu is not None:
            self._menu.render(surface)
        elif self._state == "playing" and self._playing_screen is not None:
            self._playing_screen.render(surface)
        elif self._state == "device" and self._device_menu is not None:
            self._device_menu.render(surface)
