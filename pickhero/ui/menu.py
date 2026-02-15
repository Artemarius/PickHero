"""Song selection menu screen.

Scans a directory for GP3/GP4/GP5 files and provides keyboard/mouse navigation.
"""

from __future__ import annotations

from pathlib import Path

import pygame

from pickhero.audio.input import list_audio_devices
from pickhero.config import Config
from pickhero.ui.colors import (
    BG_COLOR,
    HUD_ACCENT_COLOR,
    HUD_TEXT_COLOR,
    MENU_BG_COLOR,
    MENU_ITEM_COLOR,
    MENU_SELECTED_BG,
    MENU_SELECTED_COLOR,
)

GP_EXTENSIONS = {".gp3", ".gp4", ".gp5"}

# How many items visible at once before scrolling
VISIBLE_ITEMS = 18


def _get_font(name: str, size: int) -> pygame.font.Font:
    """Try to load a system font with fallbacks."""
    for family in (name, "Courier New", "monospace"):
        font = pygame.font.SysFont(family, size)
        if font:
            return font
    return pygame.font.Font(None, size)


class MenuScreen:
    """File browser for selecting GP tab files."""

    def __init__(self, songs_dir: Path, config: Config | None = None):
        self._songs_dir = Path(songs_dir)
        self._config = config
        self._files: list[Path] = []
        self._selected = 0
        self._scroll_offset = 0
        self._last_click_time = 0
        self._device_name = self._resolve_device_name()
        self.scan_files()

    def refresh_device_name(self) -> None:
        """Re-resolve the current device name (call after device selection)."""
        self._device_name = self._resolve_device_name()

    def _resolve_device_name(self) -> str:
        """Return a display name for the current audio device."""
        if self._config is None:
            return "Default"
        idx = self._config.audio.device_index
        if idx is None:
            return "System Default"
        try:
            for dev in list_audio_devices():
                if dev["index"] == idx:
                    return dev["name"]
        except Exception:
            pass
        return f"Device #{idx}"

    def scan_files(self) -> None:
        """Scan songs directory for GP files."""
        self._songs_dir.mkdir(parents=True, exist_ok=True)
        self._files = sorted(
            p
            for p in self._songs_dir.iterdir()
            if p.is_file() and p.suffix.lower() in GP_EXTENSIONS
        )
        self._selected = 0
        self._scroll_offset = 0

    def handle_event(self, event: pygame.event.Event) -> Path | None:
        """Process input. Returns a Path when a file is confirmed, else None."""
        if not self._files:
            return None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self._selected = max(0, self._selected - 1)
                self._ensure_visible()
            elif event.key == pygame.K_DOWN:
                self._selected = min(len(self._files) - 1, self._selected + 1)
                self._ensure_visible()
            elif event.key == pygame.K_RETURN:
                return self._files[self._selected]
            elif event.key == pygame.K_PAGEUP:
                self._selected = max(0, self._selected - VISIBLE_ITEMS)
                self._ensure_visible()
            elif event.key == pygame.K_PAGEDOWN:
                self._selected = min(
                    len(self._files) - 1, self._selected + VISIBLE_ITEMS
                )
                self._ensure_visible()
            elif event.key == pygame.K_HOME:
                self._selected = 0
                self._ensure_visible()
            elif event.key == pygame.K_END:
                self._selected = len(self._files) - 1
                self._ensure_visible()

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            idx = self._hit_test(event.pos)
            if idx is not None:
                now = pygame.time.get_ticks()
                if idx == self._selected and now - self._last_click_time < 400:
                    return self._files[self._selected]
                self._selected = idx
                self._last_click_time = now

        return None

    def render(self, surface: pygame.Surface) -> None:
        """Draw the menu screen."""
        surface.fill(MENU_BG_COLOR)
        w, h = surface.get_size()

        title_font = _get_font("arial", 36)
        item_font = _get_font("consolas", 22)
        hint_font = _get_font("arial", 16)

        # Title
        title_surf = title_font.render("PickHero", True, HUD_ACCENT_COLOR)
        surface.blit(title_surf, (w // 2 - title_surf.get_width() // 2, 24))

        # Subtitle
        sub_surf = hint_font.render(
            "Select a song to play", True, HUD_TEXT_COLOR
        )
        surface.blit(sub_surf, (w // 2 - sub_surf.get_width() // 2, 68))

        list_top = 110
        item_h = 30
        list_left = 60
        list_width = w - 120

        if not self._files:
            empty_msg = (
                f"No songs found — add .gp3/.gp4/.gp5 files to {self._songs_dir}/"
            )
            msg_surf = item_font.render(empty_msg, True, MENU_ITEM_COLOR)
            surface.blit(msg_surf, (w // 2 - msg_surf.get_width() // 2, h // 2))
            return

        # File list
        visible_end = min(self._scroll_offset + VISIBLE_ITEMS, len(self._files))
        for i in range(self._scroll_offset, visible_end):
            y = list_top + (i - self._scroll_offset) * item_h
            if i == self._selected:
                pygame.draw.rect(
                    surface,
                    MENU_SELECTED_BG,
                    (list_left - 8, y, list_width, item_h),
                    border_radius=4,
                )
                color = MENU_SELECTED_COLOR
            else:
                color = MENU_ITEM_COLOR

            label = self._files[i].name
            text_surf = item_font.render(label, True, color)
            surface.blit(text_surf, (list_left, y + 4))

        # Scroll indicators
        if self._scroll_offset > 0:
            arrow = hint_font.render("▲ more", True, HUD_TEXT_COLOR)
            surface.blit(arrow, (w // 2 - arrow.get_width() // 2, list_top - 20))
        if visible_end < len(self._files):
            arrow = hint_font.render("▼ more", True, HUD_TEXT_COLOR)
            y_bottom = list_top + VISIBLE_ITEMS * item_h + 4
            surface.blit(arrow, (w // 2 - arrow.get_width() // 2, y_bottom))

        # Current audio device
        dev_text = f"Audio: {self._device_name}"
        dev_surf = hint_font.render(dev_text, True, HUD_TEXT_COLOR)
        surface.blit(dev_surf, (w // 2 - dev_surf.get_width() // 2, h - 56))

        # Controls hint
        hint = "UP/DOWN: navigate  |  ENTER: select  |  D: audio device  |  ESC: quit"
        hint_surf = hint_font.render(hint, True, HUD_TEXT_COLOR)
        surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, h - 36))

        # Store layout for hit testing
        self._list_top = list_top
        self._item_h = item_h
        self._list_left = list_left
        self._list_width = list_width

    def _ensure_visible(self) -> None:
        """Adjust scroll offset so selected item is visible."""
        if self._selected < self._scroll_offset:
            self._scroll_offset = self._selected
        elif self._selected >= self._scroll_offset + VISIBLE_ITEMS:
            self._scroll_offset = self._selected - VISIBLE_ITEMS + 1

    def _hit_test(self, pos: tuple[int, int]) -> int | None:
        """Return index of file at mouse position, or None."""
        x, y = pos
        top = getattr(self, "_list_top", 110)
        item_h = getattr(self, "_item_h", 30)
        left = getattr(self, "_list_left", 60)
        width = getattr(self, "_list_width", 1000)

        if x < left - 8 or x > left + width:
            return None
        rel_y = y - top
        if rel_y < 0:
            return None
        idx = self._scroll_offset + int(rel_y // item_h)
        if 0 <= idx < len(self._files):
            return idx
        return None
