"""Tests for pickhero.ui.download_menu — Songsterr search/download screen logic."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.init()

from pickhero.ui.download_menu import DownloadMenuScreen, VISIBLE_ITEMS
from pickhero.tabs.downloader import SongsterrResult


def make_event(key, unicode=""):
    return pygame.event.Event(pygame.KEYDOWN, key=key, unicode=unicode)


def make_result(artist="Artist", title="Song", song_id="123"):
    return SongsterrResult(artist=artist, title=title, song_id=song_id)


@pytest.fixture
def songs_dir(tmp_path):
    return tmp_path / "songs"


class TestDownloadMenuScreenInit:
    def test_default_state_is_input(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        assert menu._state == "input"

    def test_empty_query(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        assert menu._query == ""

    def test_empty_results(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        assert menu._results == []

    def test_selected_starts_at_zero(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        assert menu._selected == 0

    def test_not_searching(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        assert not menu._searching

    def test_not_downloading(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        assert not menu._downloading


class TestHandleInputEvent:
    def test_escape_returns_back(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        result = menu.handle_event(make_event(pygame.K_ESCAPE))
        assert result == "back"

    def test_non_keydown_returns_none(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        event = pygame.event.Event(pygame.MOUSEMOTION, key=0)
        result = menu.handle_event(event)
        assert result is None

    def test_backspace_removes_last_char(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu._query = "test"
        menu.handle_event(make_event(pygame.K_BACKSPACE))
        assert menu._query == "tes"

    def test_backspace_empty_query(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu.handle_event(make_event(pygame.K_BACKSPACE))
        assert menu._query == ""

    def test_typing_adds_char(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu.handle_event(make_event(pygame.K_a, unicode="a"))
        assert menu._query == "a"

    def test_typing_multiple_chars(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        for ch in "hello":
            menu.handle_event(make_event(pygame.K_UNKNOWN, unicode=ch))
        assert menu._query == "hello"

    def test_non_printable_ignored(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu.handle_event(make_event(pygame.K_UNKNOWN, unicode="\t"))
        assert menu._query == ""

    @patch("pickhero.ui.download_menu.search")
    def test_enter_with_empty_query_does_nothing(self, mock_search, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu.handle_event(make_event(pygame.K_RETURN))
        assert menu._state == "input"
        mock_search.assert_not_called()

    @patch("pickhero.ui.download_menu.search")
    def test_enter_starts_search(self, mock_search, songs_dir):
        mock_search.return_value = []
        menu = DownloadMenuScreen(songs_dir)
        menu._query = "test"
        menu.handle_event(make_event(pygame.K_RETURN))
        assert menu._searching is True or menu._state == "status"


class TestHandleResultsEvent:
    def test_escape_returns_to_input(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu._state = "results"
        menu._results = [make_result()]
        result = menu.handle_event(make_event(pygame.K_ESCAPE))
        assert menu._state == "input"

    def test_up_decrements_selected(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu._state = "results"
        menu._results = [make_result("A"), make_result("B"), make_result("C")]
        menu._selected = 2
        menu.handle_event(make_event(pygame.K_UP))
        assert menu._selected == 1

    def test_down_increments_selected(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu._state = "results"
        menu._results = [make_result("A"), make_result("B"), make_result("C")]
        menu.handle_event(make_event(pygame.K_DOWN))
        assert menu._selected == 1

    def test_down_at_max_stays(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu._state = "results"
        menu._results = [make_result("A"), make_result("B")]
        menu._selected = 1
        menu.handle_event(make_event(pygame.K_DOWN))
        assert menu._selected == 1


class TestHandleStatusEvent:
    def test_escape_from_status(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu._state = "status"
        menu._downloading = True
        result = menu.handle_event(make_event(pygame.K_ESCAPE))
        assert result is None or result == "back"

    def test_returns_downloaded_when_complete(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu._state = "status"
        menu._downloading = False
        menu._download_result = "downloaded"
        result = menu.handle_event(make_event(pygame.K_RETURN))
        assert result == "downloaded"


class TestEnsureVisible:
    def test_scroll_up(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu._scroll_offset = 5
        menu._selected = 2
        menu._ensure_visible()
        assert menu._scroll_offset == 2

    def test_no_change_when_visible(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu._scroll_offset = 0
        menu._selected = 1
        menu._ensure_visible()
        assert menu._scroll_offset == 0

    def test_scroll_down(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        menu._scroll_offset = 0
        menu._selected = VISIBLE_ITEMS + 2
        menu._ensure_visible()
        assert menu._scroll_offset == menu._selected - VISIBLE_ITEMS + 1


class TestDoSearch:
    @patch("pickhero.ui.download_menu.search")
    def test_search_sets_results(self, mock_search, songs_dir):
        results = [make_result("A"), make_result("B")]
        mock_search.return_value = results
        menu = DownloadMenuScreen(songs_dir)
        menu._query = "test"
        menu._do_search()
        assert len(menu._results) == 2
        assert menu._state == "results"
        assert menu._selected == 0

    @patch("pickhero.ui.download_menu.search")
    def test_search_no_results(self, mock_search, songs_dir):
        mock_search.return_value = []
        menu = DownloadMenuScreen(songs_dir)
        menu._query = "test"
        menu._state = "status"
        menu._do_search()
        assert menu._searching is False
        assert "No results" in menu._status_msg


class TestStartDownload:
    def test_sets_downloading(self, songs_dir):
        menu = DownloadMenuScreen(songs_dir)
        result = make_result()
        menu._start_download(result)
        assert menu._downloading is True
        assert menu._state == "status"


