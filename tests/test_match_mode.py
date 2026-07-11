"""Tests for pickhero.audio.match_mode."""

from __future__ import annotations

import pytest

from pickhero.audio.match_mode import MatchMode, _coerce_match_mode


class TestMatchModeEnum:
    """MatchMode enum has exactly 3 members with correct values."""

    def test_members(self) -> None:
        assert len(MatchMode) == 3

    def test_arcade(self) -> None:
        assert MatchMode.ARCADE.value == "arcade"

    def test_practice(self) -> None:
        assert MatchMode.PRACTICE.value == "practice"

    def test_judge(self) -> None:
        assert MatchMode.JUDGE.value == "judge"

    def test_enums_are_distinct(self) -> None:
        values = {m.value for m in MatchMode}
        assert len(values) == 3


class TestCoerceMatchMode:
    """_coerce_matchMode produces the correct MatchMode."""

    # --- direct MatchMode passthrough ----------------------------------------

    def test_accepts_arcade_enum(self) -> None:
        assert _coerce_match_mode(MatchMode.ARCADE) is MatchMode.ARCADE

    def test_accepts_practice_enum(self) -> None:
        assert _coerce_match_mode(MatchMode.PRACTICE) is MatchMode.PRACTICE

    def test_accepts_judge_enum(self) -> None:
        assert _coerce_match_mode(MatchMode.JUDGE) is MatchMode.JUDGE

    # --- lowercase strings ---------------------------------------------------

    def test_arcade_string(self) -> None:
        assert _coerce_match_mode("arcade") is MatchMode.ARCADE

    def test_practice_string(self) -> None:
        assert _coerce_match_mode("practice") is MatchMode.PRACTICE

    def test_judge_string(self) -> None:
        assert _coerce_match_mode("judge") is MatchMode.JUDGE

    # --- case-insensitivity --------------------------------------------------

    @pytest.mark.parametrize(
        "raw",
        [
            "ARCADE",
            "Arcade",
            "arCADE",
            "PRACTICE",
            "Practice",
            "JUDGE",
            "Judge",
            "JuDgE",
        ],
    )
    def test_case_insensitive(self, raw: str) -> None:
        result = _coerce_match_mode(raw)
        assert result is MatchMode(raw.lower())

    # --- whitespace stripping ------------------------------------------------

    @pytest.mark.parametrize("raw", ["  arcade  ", "\tarcade\n", "  judge  "])
    def test_strips_whitespace(self, raw: str) -> None:
        result = _coerce_match_mode(raw)
        assert result is MatchMode(raw.strip().lower())

    # --- unknown values raise ValueError -------------------------------------

    @pytest.mark.parametrize("bad", ["expert", "normal", "", "hard", "easy"])
    def test_unknown_string_raises(self, bad: str) -> None:
        with pytest.raises(ValueError, match="unknown MatchMode"):
            _coerce_match_mode(bad)

    # --- non-string / non-enum types raise -----------------------------------

    @pytest.mark.parametrize("bad", [42, 3.14, None, ["arcade"], {"mode": "judge"}])
    def test_invalid_type_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match="unknown MatchMode"):
            _coerce_match_mode(bad)
