"""PickHero application — PyGame game loop and state machine.

Two states: MENU (song selection) and PLAYING (scrolling display).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pygame

from pickhero.audio.input import validate_device_index
from pickhero.config import Config
from pickhero.progress import ProgressTracker
from pickhero.tabs.loader import extract_backing_track, list_tracks, load_gp_file
from pickhero.ui.calibration_menu import CalibrationMenuScreen
from pickhero.ui.device_menu import DeviceMenuScreen
from pickhero.ui.download_menu import DownloadMenuScreen
from pickhero.ui.menu import MenuScreen
from pickhero.ui.scrolling import PlayingScreen

if TYPE_CHECKING:
    from pickhero.timing import TimingStats


class App:
    """Main application with game loop."""

    def __init__(self, config: Config | None = None):
        self._config = config or Config.load()
        self._progress = ProgressTracker()
        self._state = "menu"
        self._menu: MenuScreen | None = None
        self._playing_screen: PlayingScreen | None = None
        self._device_menu: DeviceMenuScreen | None = None
        self._download_menu: DownloadMenuScreen | None = None
        self._calibration_menu: CalibrationMenuScreen | None = None
        # Track switching state
        self._song_path: Path | None = None
        self._track_index: int = 0
        self._track_count: int = 0
        # Transient on-screen error message (replacing print-only errors)
        self._error_message: str = ""
        self._error_expiry_ms: float = 0.0
        # Cached TimingStats from the last playing session, for the calibration
        # nudge UI's early/late histogram. Refreshed when the playing screen tears down.
        self._last_timing_stats: "TimingStats | None" = None

    def run(self) -> None:
        """Initialize PyGame, run main loop, clean up."""
        # Validate saved audio device — fall back to default if unavailable
        if not validate_device_index(self._config.audio.device_index):
            self._show_error(
                f"Audio device #{self._config.audio.device_index} not available, "
                "using default."
            )
            self._config.audio.device_index = None
            self._config.save()

        pygame.init()
        pygame.key.set_repeat(300, 40)  # 300ms delay, then repeat every 40ms
        pygame.display.set_caption("PickHero")

        dc = self._config.display
        flags = pygame.RESIZABLE | pygame.SCALED
        surface = pygame.display.set_mode(
            (dc.width, dc.height), flags, vsync=1
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
            if self._state == "playing":
                clock.tick_busy_loop(60)
            else:
                clock.tick(60)

        pygame.quit()

    def _process_events(self, surface: pygame.Surface) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return

            if event.type == pygame.VIDEORESIZE:
                _ = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE | pygame.SCALED, vsync=1
                )

            if self._state == "menu":
                self._handle_menu_event(event)
            elif self._state == "playing":
                self._handle_playing_event(event)
            elif self._state == "device":
                self._handle_device_event(event)
            elif self._state == "download":
                self._handle_download_event(event)
            elif self._state == "calibration":
                self._handle_calibration_event(event)

    def _handle_menu_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and not self._menu.is_searching:
            if event.key == pygame.K_d:
                self._device_menu = DeviceMenuScreen(self._config)
                self._state = "device"
                return
            if event.key == pygame.K_s:
                songs_dir = Path(self._config.songs_dir)
                self._download_menu = DownloadMenuScreen(songs_dir)
                self._state = "download"
                return
            if event.key == pygame.K_g:
                self._calibration_menu = CalibrationMenuScreen(
                    self._config,
                    timing_stats_provider=lambda: self._last_timing_stats,
                )
                self._state = "calibration"
                return

        result = self._menu.handle_event(event)
        if result == "escape":
            self._running = False
        elif isinstance(result, Path):
            self._load_song(result, -1)

    def _handle_playing_event(self, event: pygame.event.Event) -> None:
        result = self._playing_screen.handle_event(event)
        if result == "menu":
            pygame.display.set_caption("PickHero")
            self._cache_timing_stats()
            self._playing_screen.stop_audio()
            self._playing_screen = None
            self._state = "menu"
            self._menu.scan_files()
        elif result == "next_track":
            # Cycle to next track
            if self._song_path is not None and self._track_count > 1:
                next_idx = (self._track_index + 1) % self._track_count
                self._cache_timing_stats()
                self._playing_screen.stop_audio()
                self._playing_screen = None
                self._load_song(self._song_path, next_idx)

    def _cache_timing_stats(self) -> None:
        """Snapshot the playing screen's TimingStats before tearing it down.

        Lets the calibration nudge UI show the early/late histogram from the
        most recent run even after returning to the menu.
        """
        if self._playing_screen is not None:
            try:
                self._last_timing_stats = self._playing_screen.get_timing_stats()
            except Exception:
                self._last_timing_stats = None

    def _handle_device_event(self, event: pygame.event.Event) -> None:
        result = self._device_menu.handle_event(event)
        if result in ("back", "selected"):
            if result == "selected":
                self._menu.refresh_device_name()
            self._device_menu = None
            self._state = "menu"

    def _handle_download_event(self, event: pygame.event.Event) -> None:
        result = self._download_menu.handle_event(event)
        if result in ("back", "downloaded"):
            if result == "downloaded":
                self._menu.scan_files()
            self._download_menu = None
            self._state = "menu"

    def _handle_calibration_event(self, event: pygame.event.Event) -> None:
        result = self._calibration_menu.handle_event(event)
        if result == "back":
            self._calibration_menu = None
            self._state = "menu"
        elif result == "complete":
            # Save calibration results to config
            from datetime import datetime
            results = self._calibration_menu.get_results()
            self._config.calibration = {"strings": {}, "calibrated_at": datetime.now().isoformat()}
            for string_num, cal in results.items():
                self._config.set_string_calibration(string_num, cal)
            self._config.save()
            self._calibration_menu = None
            self._state = "menu"

    def _load_song(self, path: Path, track_index: int = 0) -> None:
        """Load a GP file and switch to playing state.

        If track_index is -1, auto-selects the first guitar track.
        """
        try:
            timeline = load_gp_file(path, track_index if track_index >= 0 else None)
        except Exception as e:
            try:
                self._show_error(f"Error loading {path.name}: {e}")
            except UnicodeEncodeError:
                self._show_error(f"Error loading {path.name}: {type(e).__name__}")
            return

        self._song_path = path
        self._track_index = timeline.metadata.track_index

        # Determine total track count from the file
        try:
            self._track_count = len(list_tracks(path))
        except Exception as e:
            # Parse failure: don't pretend there's only one track silently.
            self._show_error(f"Could not enumerate tracks in {path.name}: {e}", duration_ms=3000.0)
            self._track_count = 1

        # Extract backing track (non-guitar tracks as MIDI)
        backing_track = None
        try:
            backing_track = extract_backing_track(
                path, exclude_track_indices={timeline.metadata.track_index},
            )
        except Exception as e:
            self._show_error(f"Backing track extraction failed: {e}", duration_ms=3000.0)
            # backing_track stays None — playback continues without accompaniment

        dc = self._config.display
        self._playing_screen = PlayingScreen(
            timeline,
            visible_beats=dc.visible_beats,
            hit_zone_fraction=dc.hit_zone_fraction,
            config=self._config,
            backing_track=backing_track,
            progress_tracker=self._progress,
            song_key=path.stem,
            on_error=self._show_error,
        )
        self._state = "playing"
        pygame.display.set_caption(
            f"PickHero — {timeline.metadata.artist} — {timeline.metadata.title}"
            .replace(" —  — ", " — ")
        )

        # Skip ahead so the first note is just entering the visible window
        if timeline.notes:
            first_note_ms = timeline.notes[0].timestamp_ms
            seek_to = max(0.0, first_note_ms - self._playing_screen._visible_window_ms)
            if seek_to > 0:
                self._playing_screen.seek(seek_to)

    def _show_error(self, message: str, duration_ms: float = 4000.0) -> None:
        """Display a transient error banner and also log to stderr."""
        import sys
        print(message, file=sys.stderr)
        self._error_message = message
        self._error_expiry_ms = pygame.time.get_ticks() + duration_ms

    def _update_error(self) -> None:
        """Clear expired error banner."""
        if self._error_message and pygame.time.get_ticks() > self._error_expiry_ms:
            self._error_message = ""

    def _draw_error(self, surface: pygame.Surface) -> None:
        """Render the active error banner at the top of the screen."""
        if not self._error_message:
            return
        font = pygame.font.SysFont("arial", 20)
        text = font.render(self._error_message, True, (255, 80, 80))
        pad = 8
        bg = pygame.Surface((text.get_width() + pad * 2, text.get_height() + pad * 2), pygame.SRCALPHA)
        bg.fill((30, 10, 10, 220))
        bg.blit(text, (pad, pad))
        x = (surface.get_width() - bg.get_width()) // 2
        surface.blit(bg, (x, 10))

    def _update(self) -> None:
        self._update_error()
        if self._state == "playing" and self._playing_screen is not None:
            self._playing_screen.update()
        elif self._state == "calibration" and self._calibration_menu is not None:
            self._calibration_menu.update()

    def _render(self, surface: pygame.Surface) -> None:
        if self._state == "menu" and self._menu is not None:
            self._menu.render(surface)
        elif self._state == "playing" and self._playing_screen is not None:
            self._playing_screen.render(surface)
        elif self._state == "device" and self._device_menu is not None:
            self._device_menu.render(surface)
        elif self._state == "download" and self._download_menu is not None:
            self._download_menu.render(surface)
        elif self._state == "calibration" and self._calibration_menu is not None:
            self._calibration_menu.render(surface)
        self._draw_error(surface)
