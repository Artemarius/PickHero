"""Audio input device selector screen.

Lists available audio input devices and lets the user pick one.
Selected device is persisted to config immediately on confirm.
"""

from __future__ import annotations

import pygame

from pickhero.audio.input import list_audio_devices
from pickhero.config import Config
from pickhero.ui.colors import get_theme

# Max items visible before scrolling
VISIBLE_ITEMS = 14


def _get_font(name: str, size: int) -> pygame.font.Font:
    for family in (name, "Courier New", "monospace"):
        font = pygame.font.SysFont(family, size)
        if font:
            return font
    return pygame.font.Font(None, size)


class DeviceMenuScreen:
    """Device selector screen — mirrors MenuScreen list-selection pattern."""

    def __init__(self, config: Config):
        self._config = config
        self._devices: list[dict] = []
        self._selected = 0
        self._scroll_offset = 0
        self._refresh_devices()

    def _refresh_devices(self) -> None:
        """Re-scan available audio input devices."""
        self._devices = list_audio_devices()
        # Set selection to match current config
        self._selected = 0  # "System Default" is index 0
        current = self._config.audio.device_index
        if current is not None:
            for i, dev in enumerate(self._devices):
                if dev["index"] == current:
                    self._selected = i + 1  # +1 because of "System Default" entry
                    break
        self._scroll_offset = 0
        self._ensure_visible()

    @property
    def _item_count(self) -> int:
        return 1 + len(self._devices)  # "System Default" + real devices

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Process input. Returns "back" on ESC, "selected" on confirm, else None."""
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            return "back"

        if event.key == pygame.K_UP:
            self._selected = max(0, self._selected - 1)
            self._ensure_visible()
        elif event.key == pygame.K_DOWN:
            self._selected = min(self._item_count - 1, self._selected + 1)
            self._ensure_visible()
        elif event.key == pygame.K_PAGEUP:
            self._selected = max(0, self._selected - VISIBLE_ITEMS)
            self._ensure_visible()
        elif event.key == pygame.K_PAGEDOWN:
            self._selected = min(self._item_count - 1, self._selected + VISIBLE_ITEMS)
            self._ensure_visible()
        elif event.key == pygame.K_HOME:
            self._selected = 0
            self._ensure_visible()
        elif event.key == pygame.K_END:
            self._selected = self._item_count - 1
            self._ensure_visible()
        elif event.key == pygame.K_r:
            self._refresh_devices()
        elif event.key == pygame.K_RETURN:
            self._apply_selection()
            return "selected"

        return None

    def _apply_selection(self) -> None:
        """Persist the selected device to config."""
        if self._selected == 0:
            self._config.audio.device_index = None
        else:
            dev = self._devices[self._selected - 1]
            self._config.audio.device_index = dev["index"]
        self._config.save()

    def render(self, surface: pygame.Surface) -> None:
        """Draw the device selector screen."""
        t = get_theme()
        surface.fill(t.menu_bg)
        w, h = surface.get_size()

        title_font = _get_font("arial", 36)
        item_font = _get_font("consolas", 22)
        hint_font = _get_font("arial", 16)

        # Title
        title_surf = title_font.render("Audio Device", True, t.hud_accent)
        surface.blit(title_surf, (w // 2 - title_surf.get_width() // 2, 24))

        sub_surf = hint_font.render(
            "Select an audio input device", True, t.hud_text
        )
        surface.blit(sub_surf, (w // 2 - sub_surf.get_width() // 2, 68))

        list_top = 110
        item_h = 30
        list_left = 60
        list_width = w - 120

        current_device_index = self._config.audio.device_index

        # Build display items: (label, is_active)
        items: list[tuple[str, bool]] = []
        items.append(("System Default", current_device_index is None))
        for dev in self._devices:
            label = f"[{dev['index']}] {dev['name']}  ({dev['sample_rate']:.0f} Hz)"
            is_active = dev["index"] == current_device_index
            items.append((label, is_active))

        # Render visible slice
        visible_end = min(self._scroll_offset + VISIBLE_ITEMS, len(items))
        for i in range(self._scroll_offset, visible_end):
            y = list_top + (i - self._scroll_offset) * item_h
            label, is_active = items[i]

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

            # Active device indicator
            if is_active:
                check = item_font.render("●", True, t.menu_check)
                surface.blit(check, (list_left - 4, y + 4))

            text_surf = item_font.render(label, True, color)
            surface.blit(text_surf, (list_left + 20, y + 4))

        # Scroll indicators
        if self._scroll_offset > 0:
            arrow = hint_font.render("▲ more", True, t.hud_text)
            surface.blit(arrow, (w // 2 - arrow.get_width() // 2, list_top - 20))
        if visible_end < len(items):
            arrow = hint_font.render("▼ more", True, t.hud_text)
            y_bottom = list_top + VISIBLE_ITEMS * item_h + 4
            surface.blit(arrow, (w // 2 - arrow.get_width() // 2, y_bottom))

        # Hints
        hint = "UP/DOWN: navigate  |  ENTER: select  |  R: refresh  |  ESC: back"
        hint_surf = hint_font.render(hint, True, t.hud_text)
        surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, h - 36))

    def _ensure_visible(self) -> None:
        if self._selected < self._scroll_offset:
            self._scroll_offset = self._selected
        elif self._selected >= self._scroll_offset + VISIBLE_ITEMS:
            self._scroll_offset = self._selected - VISIBLE_ITEMS + 1
