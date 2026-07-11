"""Tests for pickhero.ui.device_menu — device selection screen logic."""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.init()

from pickhero.config import Config, LATENCY_PRESETS
from pickhero.ui.device_menu import DeviceMenuScreen, VISIBLE_ITEMS


def make_config():
    config = Config()
    config.audio.device_index = None
    config.audio.device_name = ""
    config.audio.input_channel = 0
    config.audio.latency_mode = "low"
    return config


def make_event(key, event_type=None):
    if event_type is None:
        event_type = pygame.KEYDOWN
    return pygame.event.Event(event_type, key=key, unicode="")


@pytest.fixture
def mock_devices():
    return [
        {"index": 0, "name": "Dev A", "channels": 2, "sample_rate": 44100, "hostapi": "ALSA"},
        {"index": 1, "name": "Dev B", "channels": 1, "sample_rate": 48000, "hostapi": "ALSA"},
    ]


class TestDeviceMenuScreenInit:
    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_init_with_devices(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        assert menu._selected == 0  # system default
        assert len(menu._devices) == 2

    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_init_no_devices(self, mock_list):
        mock_list.return_value = []
        config = make_config()
        menu = DeviceMenuScreen(config)
        assert len(menu._devices) == 0

    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_init_matches_current_device(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        config.audio.device_index = 1
        menu = DeviceMenuScreen(config)
        assert menu._selected == 2  # index 1 device is at position 2 (after system default)


class TestItemCount:
    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_item_count_includes_system_default(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        assert menu._item_count == 3  # 1 system default + 2 devices

    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_item_count_empty(self, mock_list):
        mock_list.return_value = []
        config = make_config()
        menu = DeviceMenuScreen(config)
        assert menu._item_count == 1  # just system default


class TestHandleEvent:
    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_escape_returns_back(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        result = menu.handle_event(make_event(pygame.K_ESCAPE))
        assert result == "back"

    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_non_keydown_returns_none(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        result = menu.handle_event(make_event(0, pygame.MOUSEMOTION))
        assert result is None

    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_up_decrements_selected(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        menu._selected = 2
        menu.handle_event(make_event(pygame.K_UP))
        assert menu._selected == 1

    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_up_at_zero_stays(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        menu._selected = 0
        menu.handle_event(make_event(pygame.K_UP))
        assert menu._selected == 0

    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_down_increments_selected(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        menu._selected = 0
        menu.handle_event(make_event(pygame.K_DOWN))
        assert menu._selected == 1

    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_down_at_max_stays(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        menu._selected = menu._item_count - 1
        menu.handle_event(make_event(pygame.K_DOWN))
        assert menu._selected == menu._item_count - 1

    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_home(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        menu._selected = 2
        menu.handle_event(make_event(pygame.K_HOME))
        assert menu._selected == 0

    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_end(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        menu.handle_event(make_event(pygame.K_END))
        assert menu._selected == menu._item_count - 1

    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_pageup(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        menu._selected = 2
        menu.handle_event(make_event(pygame.K_PAGEUP))
        assert menu._selected == max(0, 2 - VISIBLE_ITEMS)

    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_pagedown(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        menu.handle_event(make_event(pygame.K_PAGEDOWN))
        assert menu._selected <= menu._item_count - 1

    @patch("pickhero.ui.device_menu.list_audio_devices")
    @patch("pickhero.ui.device_menu.Config.save")
    def test_return_applies_selection(self, mock_save, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        result = menu.handle_event(make_event(pygame.K_RETURN))
        assert result == "selected"


class TestCycleLatency:
    @patch("pickhero.ui.device_menu.list_audio_devices")
    @patch.object(Config, "save")
    def test_cycles_low_to_medium(self, mock_save, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        config.audio.latency_mode = "low"
        menu = DeviceMenuScreen(config)
        menu._cycle_latency()
        assert config.audio.latency_mode == "medium"

    @patch("pickhero.ui.device_menu.list_audio_devices")
    @patch.object(Config, "save")
    def test_cycles_high_to_low(self, mock_save, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        config.audio.latency_mode = "high"
        menu = DeviceMenuScreen(config)
        menu._cycle_latency()
        assert config.audio.latency_mode == "low"

    @patch("pickhero.ui.device_menu.list_audio_devices")
    @patch.object(Config, "save")
    def test_updates_buf_hop(self, mock_save, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        config.audio.latency_mode = "low"
        menu = DeviceMenuScreen(config)
        menu._cycle_latency()
        buf, hop, _ = LATENCY_PRESETS["medium"]
        assert config.audio.buf_size == buf
        assert config.audio.hop_size == hop


class TestEnsureVisible:
    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_scroll_up(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        menu._scroll_offset = 5
        menu._selected = 2
        menu._ensure_visible()
        assert menu._scroll_offset == 2

    @patch("pickhero.ui.device_menu.list_audio_devices")
    def test_no_change_when_visible(self, mock_list, mock_devices):
        mock_list.return_value = mock_devices
        config = make_config()
        menu = DeviceMenuScreen(config)
        menu._scroll_offset = 0
        menu._selected = 1
        menu._ensure_visible()
        assert menu._scroll_offset == 0


