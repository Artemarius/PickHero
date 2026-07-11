"""Tests for pickhero.audio.device_matrix — tested device database."""

import pytest

from pickhero.audio.device_matrix import (
    DEVICE_MATRIX,
    get_recommended_settings,
    validate_device_config,
)


class TestDeviceMatrixEntries:
    """DEVICE_MATRIX entries have correct structure and fields."""

    def test_is_list(self):
        """DEVICE_MATRIX is a list of dicts."""
        assert isinstance(DEVICE_MATRIX, list)
        assert len(DEVICE_MATRIX) > 0

    def test_each_entry_has_required_fields(self):
        """Each device matrix entry has all required fields."""
        required = {
            "device_name", "platforms", "backends", "sample_rates",
            "buffer_sizes", "input_latency_ms", "recommended", "notes",
        }
        for entry in DEVICE_MATRIX:
            missing = required - set(entry.keys())
            assert not missing, (
                f"Entry '{entry.get('device_name', '?')}' missing: {missing}"
            )

    def test_device_name_is_non_empty_string(self):
        """Every entry has a non-empty device_name."""
        for entry in DEVICE_MATRIX:
            assert isinstance(entry["device_name"], str)
            assert len(entry["device_name"]) > 0

    def test_platforms_is_list_of_strings(self):
        """Platforms field is a list of OS strings."""
        for entry in DEVICE_MATRIX:
            assert isinstance(entry["platforms"], list)
            for platform in entry["platforms"]:
                assert isinstance(platform, str)

    def test_sample_rates_contain_44100_and_48000(self):
        """All entries support at least 44100 and 48000."""
        for entry in DEVICE_MATRIX:
            assert 44100 in entry["sample_rates"]
            assert 48000 in entry["sample_rates"]

    def test_buffer_sizes_are_positive_integers(self):
        """Buffer sizes are positive ints."""
        for entry in DEVICE_MATRIX:
            for buf in entry["buffer_sizes"]:
                assert isinstance(buf, int)
                assert buf > 0

    def test_recommended_is_valid_preset(self):
        """recommended field is 'low', 'medium', or 'high'."""
        valid = {"low", "medium", "high"}
        for entry in DEVICE_MATRIX:
            assert entry["recommended"] in valid, (
                f"Invalid preset '{entry['recommended']}' for '{entry['device_name']}'"
            )


class TestGetRecommendedSettings:
    """get_recommended_settings returns platform-appropriate config."""

    def test_returns_valid_dict(self):
        """Returns a dict with all expected keys."""
        result = get_recommended_settings("Focusrite Scarlett Solo (3rd gen)")
        assert isinstance(result, dict)
        assert "sample_rate" in result
        assert "buffer_size" in result
        assert "hop_size" in result
        assert "latency_mode" in result
        assert "asio_enabled" in result

    def test_known_device_linux_default_portable(self):
        """Known device on linux returns matching config."""
        result = get_recommended_settings(
            "Focusrite Scarlett Solo (3rd gen)",
            platform="linux",
        )
        assert result["latency_mode"] == "medium"
        assert result["buffer_size"] == 2048
        assert result["hop_size"] == 512
        assert result["asio_enabled"] is False

    def test_unknown_device_returns_fallback_defaults(self):
        """Unknown device returns safe portable defaults."""
        result = get_recommended_settings(
            "Completely Unknown Device XYZ",
            platform="linux",
        )
        assert result["sample_rate"] == 44100
        assert result["buffer_size"] == 2048
        assert result["hop_size"] == 512
        assert result["latency_mode"] == "medium"
        assert result["asio_enabled"] is False

    def test_high_accuracy_profile_uses_48k(self):
        """High-accuracy profile uses 48000 sample rate."""
        result = get_recommended_settings(
            "Focusrite Scarlett Solo (3rd gen)",
            profile="high_accuracy",
        )
        assert result["sample_rate"] == 48000

    def test_windows_device_sets_asio_when_available(self):
        """Windows device with ASIO backend has asio_enabled=True."""
        result = get_recommended_settings(
            "Focusrite Scarlett Solo (3rd gen)",
            platform="win32",
        )
        assert result["asio_enabled"] is True

    def test_non_windows_device_asio_disabled(self):
        """Non-Windows platform always has asio_enabled=False."""
        result = get_recommended_settings(
            "Focusrite Scarlett Solo (3rd gen)",
            platform="linux",
        )
        assert result["asio_enabled"] is False


class TestValidateDeviceConfig:
    """validate_device_config returns warnings for unsupported combos."""

    def test_known_device_valid_config_returns_empty(self):
        """Known device with valid settings returns no warnings."""
        warnings = validate_device_config(
            "Focusrite Scarlett Solo (3rd gen)",
            platform="linux",
            sample_rate=48000,
            buffer_size=512,
        )
        assert warnings == []

    def test_unknown_device_returns_empty(self):
        """Unknown device returns no warnings (cannot validate)."""
        warnings = validate_device_config(
            "Unknown Device",
            platform="linux",
            sample_rate=12345,
            buffer_size=99,
        )
        assert warnings == []

    def test_unsupported_sample_rate_warns(self):
        """Unsupported sample rate produces a warning."""
        warnings = validate_device_config(
            "Focusrite Scarlett Solo (3rd gen)",
            platform="linux",
            sample_rate=96000,
            buffer_size=512,
        )
        assert len(warnings) >= 1
        assert "sample rate" in warnings[0].lower()

    def test_unsupported_buffer_size_warns(self):
        """Unsupported buffer size produces a warning."""
        warnings = validate_device_config(
            "Focusrite Scarlett Solo (3rd gen)",
            platform="linux",
            sample_rate=48000,
            buffer_size=128,
        )
        assert len(warnings) >= 1
        assert "buffer" in warnings[0].lower()

    def test_very_small_buffer_warns(self):
        """Buffer below 256 produces a warning regardless of device."""
        warnings = validate_device_config(
            "Focusrite Scarlett Solo (3rd gen)",
            platform="linux",
            sample_rate=48000,
            buffer_size=64,
        )
        buffer_warnings = [w for w in warnings if "buffer" in w.lower()]
        assert len(buffer_warnings) >= 1
