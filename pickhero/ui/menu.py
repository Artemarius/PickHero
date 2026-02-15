"""Song selection menu screen.

Scans a directory for GP3/GP4/GP5 files and provides keyboard/mouse navigation.
"""

from __future__ import annotations

from pathlib import Path

import pygame

from pickhero.audio.input import list_audio_devices
from pickhero.config import Config
from pickhero.progress import ProgressTracker
from pickhero.ui.colors import cycle_theme, get_theme

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

    def __init__(self, songs_dir: Path, config: Config | None = None,
                 progress: ProgressTracker | None = None):
        self._songs_dir = Path(songs_dir)
        self._config = config
        self._progress = progress
        self._files: list[Path] = []
        self._search_text: str = ""
        self._filtered_files: list[Path] = []
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

    @property
    def _display_files(self) -> list[Path]:
        """Return the currently visible file list (filtered or full)."""
        return self._filtered_files

    @property
    def is_searching(self) -> bool:
        """True when the user has typed search text."""
        return bool(self._search_text)

    def _apply_filter(self) -> None:
        """Filter _files by search text and reset selection."""
        if self._search_text:
            query = self._search_text.lower()
            self._filtered_files = [
                p for p in self._files if query in p.stem.lower()
            ]
        else:
            self._filtered_files = list(self._files)
        self._selected = 0
        self._scroll_offset = 0

    def scan_files(self) -> None:
        """Scan songs directory for GP files."""
        self._songs_dir.mkdir(parents=True, exist_ok=True)
        self._files = sorted(
            p
            for p in self._songs_dir.iterdir()
            if p.is_file() and p.suffix.lower() in GP_EXTENSIONS
        )
        self._search_text = ""
        self._apply_filter()

    def handle_event(self, event: pygame.event.Event) -> Path | str | None:
        """Process input. Returns Path (file selected), "escape" (quit), or None."""
        files = self._display_files

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self._search_text:
                    self._search_text = ""
                    self._apply_filter()
                    return None
                return "escape"

            if event.key == pygame.K_BACKSPACE:
                if self._search_text:
                    self._search_text = self._search_text[:-1]
                    self._apply_filter()
                return None

            # Navigation keys
            if event.key == pygame.K_UP:
                if files:
                    self._selected = max(0, self._selected - 1)
                    self._ensure_visible()
                return None
            if event.key == pygame.K_DOWN:
                if files:
                    self._selected = min(len(files) - 1, self._selected + 1)
                    self._ensure_visible()
                return None
            if event.key == pygame.K_RETURN:
                if files:
                    return files[self._selected]
                return None
            if event.key == pygame.K_PAGEUP:
                if files:
                    self._selected = max(0, self._selected - VISIBLE_ITEMS)
                    self._ensure_visible()
                return None
            if event.key == pygame.K_PAGEDOWN:
                if files:
                    self._selected = min(
                        len(files) - 1, self._selected + VISIBLE_ITEMS
                    )
                    self._ensure_visible()
                return None
            if event.key == pygame.K_HOME:
                if files:
                    self._selected = 0
                    self._ensure_visible()
                return None
            if event.key == pygame.K_END:
                if files:
                    self._selected = len(files) - 1
                    self._ensure_visible()
                return None

            # T key: toggle theme (only when not searching to avoid conflict)
            if event.key == pygame.K_t and not self._search_text:
                name = cycle_theme()
                if self._config:
                    self._config.theme = name
                    self._config.save()
                return None

            # Printable character → append to search
            ch = event.unicode
            if ch and ch.isprintable() and ch not in ("\r", "\n", "\t"):
                self._search_text += ch
                self._apply_filter()
                return None

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            idx = self._hit_test(event.pos)
            if idx is not None and files:
                now = pygame.time.get_ticks()
                if idx == self._selected and now - self._last_click_time < 400:
                    return files[self._selected]
                self._selected = idx
                self._last_click_time = now

        return None

    def render(self, surface: pygame.Surface) -> None:
        """Draw the menu screen."""
        t = get_theme()
        surface.fill(t.menu_bg)
        w, h = surface.get_size()
        files = self._display_files

        title_font = _get_font("arial", 36)
        item_font = _get_font("consolas", 22)
        hint_font = _get_font("arial", 16)

        # Title
        title_surf = title_font.render("PickHero", True, t.hud_accent)
        surface.blit(title_surf, (w // 2 - title_surf.get_width() // 2, 24))

        # Subtitle
        sub_surf = hint_font.render(
            "Select a song to play", True, t.hud_text
        )
        surface.blit(sub_surf, (w // 2 - sub_surf.get_width() // 2, 68))

        list_top = 110
        item_h = 30
        list_left = 60
        list_width = w - 120

        # Search bar
        if self._search_text:
            search_label = f"Search: {self._search_text}_"
            search_surf = item_font.render(search_label, True, t.hud_accent)
            surface.blit(search_surf, (list_left, 90))
            count_label = f"({len(files)} of {len(self._files)} songs)"
            count_surf = hint_font.render(count_label, True, t.hud_text)
            surface.blit(count_surf, (list_left + search_surf.get_width() + 12, 95))
            list_top = 120

        # Empty states
        if not files:
            if not self._files:
                empty_msg = (
                    f"No songs found — add .gp3/.gp4/.gp5 files to {self._songs_dir}/"
                )
            else:
                empty_msg = f'No songs match "{self._search_text}"'
            msg_surf = item_font.render(empty_msg, True, t.menu_item)
            surface.blit(msg_surf, (w // 2 - msg_surf.get_width() // 2, h // 2))
        else:
            # File list
            visible_end = min(self._scroll_offset + VISIBLE_ITEMS, len(files))
            for i in range(self._scroll_offset, visible_end):
                y = list_top + (i - self._scroll_offset) * item_h
                if i == self._selected:
                    pygame.draw.rect(
                        surface,
                        t.menu_selected_bg,
                        (list_left - 8, y, list_width, item_h),
                        border_radius=4,
                    )
                    color = t.menu_selected
                else:
                    color = t.menu_item

                label = files[i].name
                text_surf = item_font.render(label, True, color)
                surface.blit(text_surf, (list_left, y + 4))

                # Show best accuracy if available
                if self._progress is not None:
                    record = self._progress.get_best(files[i].stem)
                    if record is not None and record.attempts > 0:
                        pct = f"{record.best_accuracy:.0f}%"
                        pct_surf = item_font.render(pct, True, t.hud_accent)
                        pct_x = list_left + list_width - pct_surf.get_width() - 8
                        surface.blit(pct_surf, (pct_x, y + 4))

            # Scroll indicators
            if self._scroll_offset > 0:
                arrow = hint_font.render("▲ more", True, t.hud_text)
                surface.blit(arrow, (w // 2 - arrow.get_width() // 2, list_top - 20))
            if visible_end < len(files):
                arrow = hint_font.render("▼ more", True, t.hud_text)
                y_bottom = list_top + VISIBLE_ITEMS * item_h + 4
                surface.blit(arrow, (w // 2 - arrow.get_width() // 2, y_bottom))

        # Current audio device
        dev_text = f"Audio: {self._device_name}"
        dev_surf = hint_font.render(dev_text, True, t.hud_text)
        surface.blit(dev_surf, (w // 2 - dev_surf.get_width() // 2, h - 72))

        # Scoring hint
        score_hint = "Press A during playback to enable scoring"
        score_surf = hint_font.render(score_hint, True, t.hud_text)
        surface.blit(score_surf, (w // 2 - score_surf.get_width() // 2, h - 56))

        # Controls hint
        if self._search_text:
            hint = "BACKSPACE: edit  |  ESC: clear search  |  ENTER: select  |  UP/DOWN: navigate"
        else:
            hint = "Type to search  |  UP/DOWN: navigate  |  ENTER: select  |  D: audio device  |  T: theme  |  ESC: quit"
        hint_surf = hint_font.render(hint, True, t.hud_text)
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
        if 0 <= idx < len(self._display_files):
            return idx
        return None
