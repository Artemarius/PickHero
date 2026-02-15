"""Color constants for PickHero UI.

Rocksmith-style string palette. All colors are RGB tuples.
"""

# Background
BG_COLOR = (20, 20, 30)

# Lane backgrounds (alternating)
LANE_BG_EVEN = (30, 30, 42)
LANE_BG_ODD = (25, 25, 36)
LANE_LINE_COLOR = (60, 60, 80)

# Hit zone
HIT_ZONE_COLOR = (255, 255, 255)

# String colors (Rocksmith palette, keyed 1-6: 1=high E, 6=low E)
STRING_COLORS: dict[int, tuple[int, int, int]] = {
    1: (200, 50, 50),    # red
    2: (220, 200, 40),   # yellow
    3: (50, 120, 220),   # blue
    4: (220, 140, 30),   # orange
    5: (50, 180, 70),    # green
    6: (140, 60, 200),   # purple
}

# Note text and border
NOTE_TEXT_COLOR = (255, 255, 255)
NOTE_BORDER_COLOR = (10, 10, 15)

# Menu
MENU_BG_COLOR = (20, 20, 30)
MENU_ITEM_COLOR = (180, 180, 200)
MENU_SELECTED_COLOR = (255, 255, 255)
MENU_SELECTED_BG = (60, 60, 100)
MENU_CHECK_COLOR = (50, 220, 80)  # green indicator for active device

# HUD
HUD_TEXT_COLOR = (200, 200, 220)
HUD_ACCENT_COLOR = (100, 180, 255)

# Feedback
FEEDBACK_HIT = (50, 255, 50)       # bright green
FEEDBACK_CLOSE = (255, 220, 50)    # yellow
FEEDBACK_MISS = (255, 50, 50)      # red
FEEDBACK_STREAK = (255, 180, 50)   # gold

# Loop markers (cyan — distinct from all string colors and HUD accent)
LOOP_MARKER_COLOR = (0, 200, 255)
LOOP_MARKER_DISABLED_COLOR = (0, 80, 110)
LOOP_REGION_COLOR = (0, 200, 255, 25)            # RGBA for semi-transparent overlay
LOOP_REGION_DISABLED_COLOR = (0, 80, 110, 15)


def dimmed(color: tuple[int, int, int], factor: float = 0.4) -> tuple[int, int, int]:
    """Darken a color by multiplying each channel by factor."""
    return (
        int(color[0] * factor),
        int(color[1] * factor),
        int(color[2] * factor),
    )
