"""Tests for pickhero.main — entry-point helpers and main()."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

from pickhero.main import _build_parser, _resolve_songs_dir


class FakeConfig:
    """Minimal stand-in for pickhero.config.Config."""
    def __init__(self, songs_dir: str = "songs"):
        self.songs_dir = songs_dir
        self.audio = MagicMock()
        self.display = MagicMock()


class TestResolveSongsDir:
    """_resolve_songs_dir resolves a relative songs_dir against cwd
    or the frozen executable's parent directory."""

    def test_non_frozen_relative(self) -> None:
        """Non-frozen app: relative path resolved against cwd."""
        config = FakeConfig(songs_dir="songs")
        _resolve_songs_dir(config)
        expected = str(Path.cwd() / "songs")
        assert config.songs_dir == expected

    def test_non_frozen_relative_subdir(self) -> None:
        """Non-frozen app: nested relative path also resolved."""
        config = FakeConfig(songs_dir="data/songs")
        _resolve_songs_dir(config)
        expected = str(Path.cwd() / "data/songs")
        assert config.songs_dir == expected

    @patch("pickhero.main.sys")
    def test_frozen_relative(self, mock_sys: MagicMock) -> None:
        """Frozen app: relative path resolved against exe parent."""
        mock_sys.frozen = True
        mock_sys.executable = "/opt/pickhero/pickhero.exe"
        config = FakeConfig(songs_dir="songs")
        _resolve_songs_dir(config)
        expected = "/opt/pickhero/songs"
        assert config.songs_dir == expected

    @patch("pickhero.main.sys")
    def test_frozen_relative_subdir(self, mock_sys: MagicMock) -> None:
        """Frozen app: nested relative path resolved against exe parent."""
        mock_sys.frozen = True
        mock_sys.executable = "/opt/pickhero/pickhero.exe"
        config = FakeConfig(songs_dir="data/songs")
        _resolve_songs_dir(config)
        expected = "/opt/pickhero/data/songs"
        assert config.songs_dir == expected

    def test_absolute_unchanged_non_frozen(self) -> None:
        """Absolute songs_dir is never modified (non-frozen)."""
        config = FakeConfig(songs_dir="/home/user/songs")
        _resolve_songs_dir(config)
        assert config.songs_dir == "/home/user/songs"

    @patch("pickhero.main.sys")
    def test_absolute_unchanged_frozen(self, mock_sys: MagicMock) -> None:
        """Absolute songs_dir is never modified (frozen)."""
        mock_sys.frozen = True
        mock_sys.executable = "/opt/pickhero/pickhero.exe"
        config = FakeConfig(songs_dir="/etc/pickhero/songs")
        _resolve_songs_dir(config)
        assert config.songs_dir == "/etc/pickhero/songs"

    def test_cwd_slash_unchanged(self) -> None:
        """Root-based absolute path stays untouched."""
        config = FakeConfig(songs_dir="/songs")
        _resolve_songs_dir(config)
        assert config.songs_dir == "/songs"


class TestBuildParser:
    """_build_parser returns a configured ArgumentParser."""

    def test_returns_argument_parser(self) -> None:
        parser = _build_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.prog == "pickhero"

    def test_has_version_flag(self) -> None:
        """Parser exposes --version as a store-true action."""
        parser = _build_parser()
        action = parser._option_string_actions.get("--version")
        assert action is not None
        # _StoreTrueAction: nargs=0, const=True, default=False
        assert action.nargs == 0
        assert action.const is True
        assert action.default is False

    def test_has_console_subcommand(self) -> None:
        """Parser has a 'console' subcommand."""
        parser = _build_parser()
        actions = parser._actions
        sub_actions = [a for a in actions if hasattr(a, "choices") and a.choices]
        assert len(sub_actions) == 1
        assert "console" in sub_actions[0].choices

    def test_version_flag_sets_version(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_console_command_sets_command(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["console"])
        assert args.command == "console"

    def test_empty_args_sets_command_none(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_version_not_set_by_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.version is False

    def test_allow_abbrev_false(self) -> None:
        """--ver should not expand to --version when allow_abbrev=False."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--ver"])

    @patch("sys.argv", ["pickhero", "--version"])
    @patch("pickhero.main.sys")
    @patch("builtins.print")
    def test_version_flag_prints_and_exits(
        self, mock_print: MagicMock, mock_sys: MagicMock
    ) -> None:
        """main() with --version prints PickHero version and exits."""
        mock_sys.exit.side_effect = SystemExit(0)

        from pickhero.main import main

        with pytest.raises(SystemExit):
            main()

        mock_print.assert_called_once_with("PickHero 1.2.0")
        mock_sys.exit.assert_called_once_with(0)

    @patch("sys.argv", ["pickhero", "--version"])
    @patch("pickhero.main.sys")
    @patch("builtins.print")
    def test_version_message_format(
        self, mock_print: MagicMock, mock_sys: MagicMock
    ) -> None:
        """The printed version string starts with 'PickHero'."""
        mock_sys.exit.side_effect = SystemExit(0)

        from pickhero.main import main

        with pytest.raises(SystemExit):
            main()

        call_arg = mock_print.call_args[0][0]
        assert call_arg.startswith("PickHero")

    @patch("sys.argv", ["pickhero", "--version"])
    @patch("pickhero.main.sys")
    @patch("builtins.print")
    def test_version_uses_exit_code_zero(
        self, mock_print: MagicMock, mock_sys: MagicMock
    ) -> None:
        """--version exits with code 0."""
        mock_sys.exit.side_effect = SystemExit(0)

        from pickhero.main import main

        with pytest.raises(SystemExit):
            main()

        mock_sys.exit.assert_called_once_with(0)
