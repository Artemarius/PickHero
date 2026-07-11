"""Tests for pickhero.__main__ — module entry point."""
from __future__ import annotations

import os


def test_main_importable():
    """__main__.py should import main from pickhero.main."""
    from pickhero.__main__ import main
    assert callable(main)


def test_main_is_same_function():
    """__main__.main should be the same function as pickhero.main.main."""
    from pickhero.__main__ import main as entry_main
    from pickhero.main import main as real_main
    assert entry_main is real_main
