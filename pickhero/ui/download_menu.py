"""Songsterr search and download screen.

Lets the user search for tabs online and download them to the songs directory.
Falls back to opening the browser if direct download fails.
"""

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path

import pygame

from pickhero.tabs.downloader import (
    SongsterrResult,
    download_gp5,
    get_songsterr_url,
    sanitize_filename,
    search,
)
from pickhero.ui.colors import get_theme

VISIBLE_ITEMS = 14


def _get_font(name: str, size: int) -> pygame.font.Font:
    for family in (name, "Courier New", "monospace"):
        font = pygame.font.SysFont(family, size)
        if font:
            return font
    return pygame.font.Font(None, size)


class DownloadMenuScreen:
    """Songsterr search/download screen — 3 states: input, results, status."""

    def __init__(self, songs_dir: Path):
        self._songs_dir = songs_dir
        self._state = "input"  # "input" | "results" | "status"
        self._query = ""
        self._results: list[SongsterrResult] = []
        self._selected = 0
        self._scroll_offset = 0
        self._status_msg = ""
        self._searching = False
        self._downloading = False
        self._download_result: str | None = None  # "downloaded" or "failed"

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Process input. Returns "back" (ESC from input) or "downloaded" (success)."""
        if self._state == "input":
            return self._handle_input_event(event)
        elif self._state == "results":
            return self._handle_results_event(event)
        elif self._state == "status":
            return self._handle_status_event(event)
        return None

    def _handle_input_event(self, event: pygame.event.Event) -> str | None:
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            return "back"

        if event.key == pygame.K_RETURN and self._query.strip() and not self._searching:
            self._searching = True
            self._status_msg = "Searching..."
            self._state = "status"
            thread = threading.Thread(target=self._do_search, daemon=True)
            thread.start()
            return None

        if event.key == pygame.K_BACKSPACE:
            self._query = self._query[:-1]
            return None

        ch = event.unicode
        if ch and ch.isprintable() and ch not in ("\r", "\n", "\t"):
            self._query += ch

        return None

    def _handle_results_event(self, event: pygame.event.Event) -> str | None:
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            self._state = "input"
            return None

        if event.key == pygame.K_UP:
            self._selected = max(0, self._selected - 1)
            self._ensure_visible()
        elif event.key == pygame.K_DOWN:
            self._selected = min(len(self._results) - 1, self._selected + 1)
            self._ensure_visible()
        elif event.key == pygame.K_PAGEUP:
            self._selected = max(0, self._selected - VISIBLE_ITEMS)
            self._ensure_visible()
        elif event.key == pygame.K_PAGEDOWN:
            self._selected = min(len(self._results) - 1, self._selected + VISIBLE_ITEMS)
            self._ensure_visible()
        elif event.key == pygame.K_HOME:
            self._selected = 0
            self._ensure_visible()
        elif event.key == pygame.K_END:
            self._selected = len(self._results) - 1
            self._ensure_visible()
        elif event.key == pygame.K_RETURN and self._results:
            self._start_download(self._results[self._selected])

        return None

    def _handle_status_event(self, event: pygame.event.Event) -> str | None:
        # Check if background work finished
        if self._download_result == "downloaded":
            self._download_result = None
            return "downloaded"

        if self._download_result == "failed":
            self._download_result = None
            # Already opened browser as fallback; go back to results
            self._state = "results"
            return None

        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            if self._searching or self._downloading:
                return None  # can't cancel network ops
            self._state = "input"
            return None

        return None

    def _do_search(self) -> None:
        """Run search in background thread."""
        results = search(self._query.strip())
        self._results = results
        self._selected = 0
        self._scroll_offset = 0
        self._searching = False
        if results:
            self._state = "results"
        else:
            self._status_msg = "No results found. Press ESC to try again."

    def _start_download(self, result: SongsterrResult) -> None:
        """Begin download in background thread."""
        self._downloading = True
        self._status_msg = f"Downloading {result.artist} - {result.title}..."
        self._state = "status"
        thread = threading.Thread(
            target=self._do_download, args=(result,), daemon=True
        )
        thread.start()

    def _do_download(self, result: SongsterrResult) -> None:
        """Run download in background thread."""
        name = sanitize_filename(f"{result.artist} - {result.title}")
        output_path = self._songs_dir / f"{name}.gp5"

        ok = download_gp5(result.song_id, output_path)
        self._downloading = False
        if ok:
            self._status_msg = f"Downloaded: {name}.gp5"
            self._download_result = "downloaded"
        else:
            self._status_msg = "Download failed — opening Songsterr in browser..."
            webbrowser.open(get_songsterr_url(result.song_id))
            self._download_result = "failed"

    def render(self, surface: pygame.Surface) -> None:
        t = get_theme()
        surface.fill(t.menu_bg)
        w, h = surface.get_size()

        title_font = _get_font("arial", 36)
        item_font = _get_font("consolas", 22)
        hint_font = _get_font("arial", 16)

        # Title
        title_surf = title_font.render("Search Songsterr", True, t.hud_accent)
        surface.blit(title_surf, (w // 2 - title_surf.get_width() // 2, 24))

        if self._state == "input":
            self._render_input(surface, w, h, item_font, hint_font, t)
        elif self._state == "results":
            self._render_results(surface, w, h, item_font, hint_font, t)
        elif self._state == "status":
            self._render_status(surface, w, h, item_font, hint_font, t)

    def _render_input(self, surface, w, h, item_font, hint_font, t) -> None:
        sub_surf = hint_font.render(
            "Type a song or artist name", True, t.hud_text
        )
        surface.blit(sub_surf, (w // 2 - sub_surf.get_width() // 2, 68))

        # Search box
        box_left = 60
        box_top = 140
        box_width = w - 120
        box_height = 36

        pygame.draw.rect(
            surface, t.menu_selected_bg,
            (box_left, box_top, box_width, box_height),
            border_radius=4,
        )
        prompt = f"> {self._query}_"
        prompt_surf = item_font.render(prompt, True, t.menu_selected)
        surface.blit(prompt_surf, (box_left + 8, box_top + 6))

        hint = "ENTER: search  |  ESC: back"
        hint_surf = hint_font.render(hint, True, t.hud_text)
        surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, h - 36))

    def _render_results(self, surface, w, h, item_font, hint_font, t) -> None:
        sub_surf = hint_font.render(
            f"{len(self._results)} results for \"{self._query}\"", True, t.hud_text
        )
        surface.blit(sub_surf, (w // 2 - sub_surf.get_width() // 2, 68))

        list_top = 110
        item_h = 30
        list_left = 60
        list_width = w - 120

        visible_end = min(self._scroll_offset + VISIBLE_ITEMS, len(self._results))
        for i in range(self._scroll_offset, visible_end):
            y = list_top + (i - self._scroll_offset) * item_h
            result = self._results[i]

            if i == self._selected:
                pygame.draw.rect(
                    surface, t.menu_selected_bg,
                    (list_left - 8, y, list_width, item_h),
                    border_radius=4,
                )
                color = t.menu_selected
            else:
                color = t.menu_item

            label = f"{result.artist} - {result.title}"
            text_surf = item_font.render(label, True, color)
            surface.blit(text_surf, (list_left, y + 4))

        # Scroll indicators
        if self._scroll_offset > 0:
            arrow = hint_font.render("▲ more", True, t.hud_text)
            surface.blit(arrow, (w // 2 - arrow.get_width() // 2, list_top - 20))
        if visible_end < len(self._results):
            arrow = hint_font.render("▼ more", True, t.hud_text)
            y_bottom = list_top + VISIBLE_ITEMS * item_h + 4
            surface.blit(arrow, (w // 2 - arrow.get_width() // 2, y_bottom))

        hint = "UP/DOWN: navigate  |  ENTER: download  |  ESC: back to search"
        hint_surf = hint_font.render(hint, True, t.hud_text)
        surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, h - 36))

    def _render_status(self, surface, w, h, item_font, hint_font, t) -> None:
        msg_surf = item_font.render(self._status_msg, True, t.hud_text)
        surface.blit(msg_surf, (w // 2 - msg_surf.get_width() // 2, h // 2))

        if not self._searching and not self._downloading:
            hint = "ESC: back"
            hint_surf = hint_font.render(hint, True, t.hud_text)
            surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, h - 36))

    def _ensure_visible(self) -> None:
        if self._selected < self._scroll_offset:
            self._scroll_offset = self._selected
        elif self._selected >= self._scroll_offset + VISIBLE_ITEMS:
            self._scroll_offset = self._selected - VISIBLE_ITEMS + 1
