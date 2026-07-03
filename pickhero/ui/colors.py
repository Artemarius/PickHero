"""Color constants for PickHero UI.

Rocksmith-style string palette. Supports dark/light themes via Theme dataclass.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """All UI colors for a theme."""

    # Background
    bg: tuple[int, int, int]

    # Lane backgrounds (alternating)
    lane_bg_even: tuple[int, int, int]
    lane_bg_odd: tuple[int, int, int]
    lane_line: tuple[int, int, int]

    # Hit zone
    hit_zone: tuple[int, int, int]

    # Note text and border
    note_text: tuple[int, int, int]
    note_border: tuple[int, int, int]

    # Menu
    menu_bg: tuple[int, int, int]
    menu_item: tuple[int, int, int]
    menu_selected: tuple[int, int, int]
    menu_selected_bg: tuple[int, int, int]
    menu_check: tuple[int, int, int]

    # HUD
    hud_text: tuple[int, int, int]
    hud_accent: tuple[int, int, int]

    # Feedback
    feedback_hit: tuple[int, int, int]
    feedback_close: tuple[int, int, int]
    feedback_miss: tuple[int, int, int]
    feedback_streak: tuple[int, int, int]

    # Loop markers
    loop_marker: tuple[int, int, int]
    loop_marker_disabled: tuple[int, int, int]
    loop_region: tuple[int, int, int, int]          # RGBA
    loop_region_disabled: tuple[int, int, int, int]  # RGBA

    # Signal meter
    signal_hot: tuple[int, int, int]
    signal_warm: tuple[int, int, int]
    signal_cold: tuple[int, int, int]

    # Tuner
    tuner_in_tune: tuple[int, int, int]
    tuner_close: tuple[int, int, int]
    tuner_off: tuple[int, int, int]

    # Timing Judge
    timing_early: tuple[int, int, int]
    timing_late: tuple[int, int, int]
    timing_on_time: tuple[int, int, int]


DARK_THEME = Theme(
    bg=(20, 20, 30),
    lane_bg_even=(30, 30, 42),
    lane_bg_odd=(25, 25, 36),
    lane_line=(60, 60, 80),
    hit_zone=(255, 255, 255),
    note_text=(255, 255, 255),
    note_border=(10, 10, 15),
    menu_bg=(20, 20, 30),
    menu_item=(180, 180, 200),
    menu_selected=(255, 255, 255),
    menu_selected_bg=(60, 60, 100),
    menu_check=(50, 220, 80),
    hud_text=(200, 200, 220),
    hud_accent=(100, 180, 255),
    feedback_hit=(50, 255, 50),
    feedback_close=(255, 220, 50),
    feedback_miss=(255, 50, 50),
    feedback_streak=(255, 180, 50),
    loop_marker=(0, 200, 255),
    loop_marker_disabled=(0, 80, 110),
    loop_region=(0, 200, 255, 25),
    loop_region_disabled=(0, 80, 110, 15),
    signal_hot=(50, 220, 80),
    signal_warm=(220, 200, 40),
    signal_cold=(70, 70, 90),
    tuner_in_tune=(50, 220, 80),
    tuner_close=(220, 200, 40),
    tuner_off=(220, 100, 40),
    timing_early=(100, 180, 255),
    timing_late=(255, 100, 100),
    timing_on_time=(50, 220, 80),
)

LIGHT_THEME = Theme(
    bg=(235, 235, 240),
    lane_bg_even=(225, 225, 232),
    lane_bg_odd=(218, 218, 228),
    lane_line=(180, 180, 195),
    hit_zone=(40, 40, 50),
    note_text=(255, 255, 255),
    note_border=(80, 80, 100),
    menu_bg=(235, 235, 240),
    menu_item=(80, 80, 100),
    menu_selected=(20, 20, 30),
    menu_selected_bg=(180, 200, 240),
    menu_check=(30, 160, 60),
    hud_text=(60, 60, 80),
    hud_accent=(30, 100, 200),
    feedback_hit=(30, 200, 30),
    feedback_close=(200, 170, 20),
    feedback_miss=(220, 40, 40),
    feedback_streak=(200, 140, 30),
    loop_marker=(0, 150, 200),
    loop_marker_disabled=(100, 140, 160),
    loop_region=(0, 150, 200, 30),
    loop_region_disabled=(100, 140, 160, 15),
    signal_hot=(30, 180, 60),
    signal_warm=(200, 170, 20),
    signal_cold=(160, 160, 175),
    tuner_in_tune=(30, 180, 60),
    tuner_close=(200, 170, 20),
    tuner_off=(200, 80, 30),
    timing_early=(40, 120, 200),
    timing_late=(200, 60, 60),
    timing_on_time=(40, 180, 60),
)

_THEMES = {"dark": DARK_THEME, "light": LIGHT_THEME}
_current_theme: Theme = DARK_THEME


def set_theme(name: str) -> None:
    """Set the active theme by name ('dark' or 'light')."""
    global _current_theme
    _current_theme = _THEMES.get(name, DARK_THEME)


def get_theme() -> Theme:
    """Return the active theme."""
    return _current_theme


def get_theme_name() -> str:
    """Return the name of the active theme."""
    if _current_theme is LIGHT_THEME:
        return "light"
    return "dark"


def cycle_theme() -> str:
    """Cycle to the next theme. Returns the new theme name."""
    if _current_theme is DARK_THEME:
        set_theme("light")
        return "light"
    set_theme("dark")
    return "dark"


# String colors (Rocksmith palette, keyed 1-6: 1=high E, 6=low E)
# These don't change with theme — they're gameplay identifiers.
STRING_COLORS: dict[int, tuple[int, int, int]] = {
    1: (200, 50, 50),    # red
    2: (220, 200, 40),   # yellow
    3: (50, 120, 220),   # blue
    4: (220, 140, 30),   # orange
    5: (50, 180, 70),    # green
    6: (140, 60, 200),   # purple
}


def dimmed(color: tuple[int, int, int], factor: float = 0.4) -> tuple[int, int, int]:
    """Darken a color by multiplying each channel by factor."""
    return (
        int(color[0] * factor),
        int(color[1] * factor),
        int(color[2] * factor),
    )
