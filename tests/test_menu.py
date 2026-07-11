"""Tests for pickhero.ui.menu — song selection menu logic.

Uses SDL dummy driver for headless testing.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.init()

from pickhero.config import Config
from pickhero.progress import ProgressTracker
from pickhero.ui.menu import MenuScreen, SORT_MODES, GP_EXTENSIONS


@pytest.fixture
def songs_dir(tmp_path):
    d = tmp_path / "songs"
    d.mkdir()
    (d / "song1.gp5").write_bytes(b"")
    (d / "song2.gp4").write_bytes(b"")
    (d / "song3.gp3").write_bytes(b"")
    (d / "readme.txt").write_text("not a song")
    (d / "sub").mkdir()
    (d / "sub" / "nested.gp5").write_bytes(b"")
    return d


@pytest.fixture
def progress(tmp_path):
    from pickhero import progress as progress_mod
    progress_mod.PROGRESS_FILE = tmp_path / "progress.json"
    return ProgressTracker()


class TestSortModes:
    def test_sort_modes(self):
        assert SORT_MODES == ["name_asc", "name_za", "accuracy", "last_played"]


class TestGPExtensions:
    def test_includes_gp3_to_gp5(self):
        assert ".gp3" in GP_EXTENSIONS
        assert ".gp4" in GP_EXTENSIONS
        assert ".gp5" in GP_EXTENSIONS

    def test_includes_gp7_gp8(self):
        assert ".gp7" in GP_EXTENSIONS
        assert ".gp8" in GP_EXTENSIONS

    def test_excludes_non_gp(self):
        assert ".txt" not in GP_EXTENSIONS
        assert ".wav" not in GP_EXTENSIONS


class TestMenuScreenInit:
    def test_scans_files(self, songs_dir):
        menu = MenuScreen(songs_dir)
        assert len(menu._files) == 4
        names = [p.name for p in menu._files]
        assert "song1.gp5" in names
        assert "song2.gp4" in names
        assert "song3.gp3" in names
        assert "nested.gp5" in names

    def test_excludes_non_gp_files(self, songs_dir):
        menu = MenuScreen(songs_dir)
        names = [p.name for p in menu._files]
        assert "readme.txt" not in names

    def test_default_sort_mode(self, songs_dir):
        menu = MenuScreen(songs_dir)
        assert menu._sort_mode == "name_asc"

    def test_filtered_files_populated(self, songs_dir):
        menu = MenuScreen(songs_dir)
        assert len(menu._filtered_files) == len(menu._files)

    def test_selected_starts_at_zero(self, songs_dir):
        menu = MenuScreen(songs_dir)
        assert menu._selected == 0

    def test_search_not_active_by_default(self, songs_dir):
        menu = MenuScreen(songs_dir)
        assert not menu.is_searching

    def test_creates_dir_if_missing(self, tmp_path):
        d = tmp_path / "nonexistent"
        MenuScreen(d)
        assert d.exists()

    def test_with_config(self, songs_dir, tmp_path):
        from pickhero import progress as progress_mod
        progress_mod.PROGRESS_FILE = tmp_path / "progress.json"
        config = Config(songs_dir=str(songs_dir))
        menu = MenuScreen(songs_dir, config=config)
        assert menu._config is config


class TestApplyFilter:
    def test_no_filter_shows_all(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._search_text = ""
        menu._apply_filter()
        assert len(menu._filtered_files) == len(menu._files)

    def test_filter_by_name(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._search_text = "song1"
        menu._apply_filter()
        assert len(menu._filtered_files) == 1
        assert menu._filtered_files[0].name == "song1.gp5"

    def test_filter_case_insensitive(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._search_text = "SONG2"
        menu._apply_filter()
        assert len(menu._filtered_files) == 1
        assert menu._filtered_files[0].name == "song2.gp4"

    def test_filter_no_match(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._search_text = "nonexistent"
        menu._apply_filter()
        assert len(menu._filtered_files) == 0

    def test_filter_resets_selection(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._selected = 3
        menu._search_text = "song1"
        menu._apply_filter()
        assert menu._selected == 0

    def test_filter_resets_scroll(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._scroll_offset = 5
        menu._search_text = "song1"
        menu._apply_filter()
        assert menu._scroll_offset == 0


class TestSortFiles:
    def test_sort_name_asc(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._sort_mode = "name_asc"
        menu._sort_files()
        names = [p.name for p in menu._filtered_files]
        assert names == sorted(names, key=str.lower)

    def test_sort_name_za(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._sort_mode = "name_za"
        menu._sort_files()
        names = [p.name for p in menu._filtered_files]
        assert names == sorted(names, key=str.lower, reverse=True)

    def test_sort_accuracy_played_present(self, songs_dir, progress):
        menu = MenuScreen(songs_dir, progress=progress)
        progress.record_result("song1", {"accuracy_percent": 90.0, "hits": 9, "total": 10})
        menu._sort_mode = "accuracy"
        menu._sort_files()
        stems = [p.stem for p in menu._filtered_files]
        assert "song1" in stems

    def test_sort_last_played_played_present(self, songs_dir, progress):
        menu = MenuScreen(songs_dir, progress=progress)
        progress.record_result("song1", {"accuracy_percent": 90.0, "hits": 9, "total": 10})
        menu._sort_mode = "last_played"
        menu._sort_files()
        stems = [p.stem for p in menu._filtered_files]
        assert "song1" in stems


class TestCycleSort:
    def test_cycles_through_modes(self, songs_dir):
        menu = MenuScreen(songs_dir)
        assert menu._sort_mode == "name_asc"
        menu._cycle_sort()
        assert menu._sort_mode == "name_za"
        menu._cycle_sort()
        assert menu._sort_mode == "accuracy"
        menu._cycle_sort()
        assert menu._sort_mode == "last_played"
        menu._cycle_sort()
        assert menu._sort_mode == "name_asc"

    def test_resets_selection(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._selected = 2
        menu._cycle_sort()
        assert menu._selected == 0


class TestEnsureVisible:
    def test_scroll_up(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._scroll_offset = 5
        menu._selected = 2
        menu._ensure_visible()
        assert menu._scroll_offset == 2

    def test_scroll_down(self, songs_dir):
        from pickhero.ui.menu import VISIBLE_ITEMS
        menu = MenuScreen(songs_dir)
        menu._scroll_offset = 0
        menu._selected = VISIBLE_ITEMS + 2
        menu._ensure_visible()
        assert menu._scroll_offset == menu._selected - VISIBLE_ITEMS + 1

    def test_no_change_when_visible(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._scroll_offset = 0
        menu._selected = 1
        menu._ensure_visible()
        assert menu._scroll_offset == 0


class TestHitTest:
    def test_hit_within_list(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._list_top = 110
        menu._item_h = 30
        menu._list_left = 60
        menu._list_width = 1000
        menu._scroll_offset = 0
        idx = menu._hit_test((100, 110))
        assert idx == 0

    def test_hit_second_item(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._list_top = 110
        menu._item_h = 30
        idx = menu._hit_test((100, 140))
        assert idx == 1

    def test_hit_outside_left(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._list_left = 60
        idx = menu._hit_test((10, 110))
        assert idx is None

    def test_hit_above_list(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._list_top = 110
        idx = menu._hit_test((100, 50))
        assert idx is None

    def test_hit_with_scroll_offset(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._list_top = 110
        menu._item_h = 30
        menu._scroll_offset = 3
        idx = menu._hit_test((100, 110))
        assert idx == 3

    def test_hit_past_end(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._list_top = 110
        menu._item_h = 30
        menu._scroll_offset = 0
        idx = menu._hit_test((100, 10000))
        assert idx is None


class TestScanFiles:
    def test_finds_new_files(self, songs_dir):
        menu = MenuScreen(songs_dir)
        initial = len(menu._files)
        (songs_dir / "new_song.gp5").write_bytes(b"")
        menu.scan_files()
        assert len(menu._files) == initial + 1

    def test_clears_search(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._search_active = True
        menu._search_text = "test"
        menu.scan_files()
        assert menu._search_text == ""
        assert not menu._search_active


class TestDisplayFiles:
    def test_reflects_filtered(self, songs_dir):
        menu = MenuScreen(songs_dir)
        menu._search_text = "song1"
        menu._apply_filter()
        assert len(menu._display_files) == 1


class TestResolveDeviceName:
    @patch("pickhero.ui.menu.list_audio_devices")
    def test_returns_device_name(self, mock_list, songs_dir):
        mock_list.return_value = [{"index": 0, "name": "Test Device", "channels": 1, "sample_rate": 44100}]
        menu = MenuScreen(songs_dir)
        name = menu._resolve_device_name()
        assert isinstance(name, str)

    @patch("pickhero.ui.menu.list_audio_devices")
    def test_no_devices_returns_default(self, mock_list, songs_dir):
        mock_list.return_value = []
        menu = MenuScreen(songs_dir)
        name = menu._resolve_device_name()
        assert isinstance(name, str)


