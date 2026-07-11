"""Static overlay renderers for the playing screen.

These were extracted from ``scrolling.py`` to reduce the size of that module
and make the large, self-contained drawing routines easier to read in
isolation. They are stateless: callers pass all required values explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pygame

from pickhero.ui.colors import STRING_COLORS, get_theme

if TYPE_CHECKING:
    from pickhero.matcher import NoteMatcher
    from pickhero.timing import TimingStats


def _get_font(name: str, size: int) -> pygame.font.Font:
    """Try to load a system font with fallbacks."""
    for family in (name, "Courier New", "monospace"):
        font = pygame.font.SysFont(family, size)
        if font:
            return font
    return pygame.font.Font(None, size)


class _FontCache:
    """Cache rendered font surfaces to avoid repeated font.render() calls."""

    def __init__(self) -> None:
        self._cache: dict[tuple, pygame.Surface] = {}

    def render(
        self,
        font: pygame.font.Font,
        text: str,
        antialias: bool,
        color: tuple,
    ) -> pygame.Surface:
        key = (id(font), text, antialias, color)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        surf = font.render(text, antialias, color)
        surf = surf.convert_alpha()
        self._cache[key] = surf
        return surf


_font_cache = _FontCache()


def draw_help_overlay(surface: pygame.Surface, layout) -> None:
    """Draw a help overlay explaining the track, note colors, and controls."""
    t = get_theme()
    w, h = layout.screen_w, layout.screen_h

    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))

    title_font = _get_font("arial", 28)
    section_font = _get_font("arial", 20)
    body_font = _get_font("arial", 17)
    hint_font = _get_font("arial", 14)

    cx = w // 2
    y = 30

    title_surf = _font_cache.render(title_font, "Help", True, t.hud_accent)
    surface.blit(title_surf, (cx - title_surf.get_width() // 2, y))
    y += 40

    lx = cx - 260
    section = _font_cache.render(section_font, "Reading the Track", True, t.hud_accent)
    surface.blit(section, (lx, y))
    y += 26

    track_lines = [
        "Notes scroll right-to-left toward the hit zone (white vertical line).",
        "The number on each note is the fret to press (0 = open string).",
        "Play the right fret on the right string as the note crosses the line.",
    ]
    for line in track_lines:
        surf = _font_cache.render(body_font, line, True, t.hud_text)
        surface.blit(surf, (lx, y))
        y += 21
    y += 6

    section = _font_cache.render(section_font, "The 6 Rows = 6 Guitar Strings", True, t.hud_accent)
    surface.blit(section, (lx, y))
    y += 26

    for line in ["Each horizontal row is one guitar string, top to bottom:"]:
        surf = _font_cache.render(body_font, line, True, t.hud_text)
        surface.blit(surf, (lx, y))
        y += 21

    string_info = [
        (1, "Row 1 (top)     = high E  (thinnest)"),
        (2, "Row 2              = B"),
        (3, "Row 3              = G"),
        (4, "Row 4              = D"),
        (5, "Row 5              = A"),
        (6, "Row 6 (bottom) = low E  (thickest)"),
    ]
    for s, label in string_info:
        color = STRING_COLORS.get(s, (180, 180, 180))
        pygame.draw.rect(surface, color, (lx, y + 3, 14, 14), border_radius=2)
        surf = _font_cache.render(body_font, label, True, t.hud_text)
        surface.blit(surf, (lx + 20, y))
        y += 20
    y += 4

    surf = _font_cache.render(
        body_font, "A note's color tells you which string to play — it matches the row.", True, t.hud_text
    )
    surface.blit(surf, (lx, y))
    y += 20
    surf = _font_cache.render(body_font, "Dimmed notes have already passed the hit zone.", True, t.hud_text)
    surface.blit(surf, (lx, y))
    y += 24

    section = _font_cache.render(section_font, "Scoring (colors change after you play)", True, t.hud_accent)
    surface.blit(section, (lx, y))
    y += 26

    surf = _font_cache.render(
        body_font, "When audio is on, notes change color after they pass the hit zone:", True, t.hud_text
    )
    surface.blit(surf, (lx, y))
    y += 22

    feedback = [
        (t.feedback_hit, "Turns green", "you played the correct note"),
        (t.feedback_close, "Turns yellow", "close, off by 1 semitone"),
        (t.feedback_miss, "Turns red", "you missed it (not played in time)"),
    ]
    for color, label, desc in feedback:
        pygame.draw.rect(surface, color, (lx + 10, y + 3, 14, 14), border_radius=2)
        surf = _font_cache.render(body_font, f"{label} — {desc}", True, t.hud_text)
        surface.blit(surf, (lx + 30, y))
        y += 21
    y += 10

    section = _font_cache.render(section_font, "Controls", True, t.hud_accent)
    surface.blit(section, (lx, y))
    y += 24

    controls = [
        "SPACE: play/pause    LEFT/RIGHT: seek    HOME: restart",
        "A: toggle audio    PgDn/PgUp: tempo    X/C: noise gate",
        "B: backing track    T: theme    I/O: loop markers    P: toggle loop",
        "F: fret limit    F1-F6: toggle strings    V: chord mode    L: loop weakest",
        "W: wait mode (pause until correct note played)",
    ]
    for line in controls:
        surf = _font_cache.render(hint_font, line, True, t.hud_text)
        surface.blit(surf, (lx, y))
        y += 18

    y += 10
    close_surf = _font_cache.render(hint_font, "Press H to close", True, t.hud_accent)
    surface.blit(close_surf, (cx - close_surf.get_width() // 2, y))


@dataclass(frozen=True)
class CompletionState:
    """Inputs needed by ``draw_completion_overlay``."""

    audio_enabled: bool
    matcher: "NoteMatcher | None"
    is_new_best: bool
    recommendations: list[str]
    weakest_sections: list[tuple[int, int, float]]
    timing_judge: bool
    timing_summary: "TimingStats | None"
    timing_worst_measures: list[tuple[int, float, float]]
    technique_heatmap: dict[str, dict[str, float]] = field(default_factory=dict)
    """kind -> {"accuracy": float, "count": int}. Empty when no verdicts."""
    drill_recommendation: str | None = None

def draw_completion_overlay(
    surface: pygame.Surface,
    layout,
    state: CompletionState,
) -> None:
    """Draw the song completion results overlay."""
    t = get_theme()
    w, h = layout.screen_w, layout.screen_h

    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))

    header_font = _get_font("arial", 48)
    stat_font = _get_font("consolas", 28)
    hint_font = _get_font("arial", 18)

    center_y = h // 2 - 80

    header_surf = _font_cache.render(header_font, "Song Complete!", True, t.hud_accent)
    surface.blit(header_surf, (w // 2 - header_surf.get_width() // 2, center_y))

    if state.audio_enabled and state.matcher is not None:
        stats = state.matcher.get_statistics()
        accuracy_text = f"Accuracy: {stats['accuracy_percent']:.1f}%  ({stats['hits']}/{stats['total']})"
        acc_surf = _font_cache.render(stat_font, accuracy_text, True, t.hud_text)
        surface.blit(acc_surf, (w // 2 - acc_surf.get_width() // 2, center_y + 60))

        if stats.get("technique_total", 0) > 0:
            tech_text = (
                f"Technique: {stats['technique_accuracy_percent']:.0f}%  "
                f"({stats['technique_correct']}/{stats['technique_total']})"
            )
            tech_color = t.hud_accent if stats["technique_accuracy_percent"] >= 70 else t.feedback_close
            tech_surf = _font_cache.render(stat_font, tech_text, True, tech_color)
            surface.blit(tech_surf, (w // 2 - tech_surf.get_width() // 2, center_y + 90))

        if state.is_new_best:
            best_surf = _font_cache.render(stat_font, "New Best!", True, (255, 220, 50))
            surface.blit(best_surf, (w // 2 - best_surf.get_width() // 2, center_y + 115))

        if state.weakest_sections:
            section = state.weakest_sections[0]
            weak_text = f"Weakest: bars {section[0]+1}-{section[1]+1} ({section[2]:.0f}%) -- press L to loop"
            weak_surf = _font_cache.render(hint_font, weak_text, True, t.feedback_close)
            surface.blit(weak_surf, (w // 2 - weak_surf.get_width() // 2, center_y + 150))

        rec_y = center_y + 180
        for rec in state.recommendations:
            rec_surf = _font_cache.render(hint_font, rec, True, t.hud_accent)
            surface.blit(rec_surf, (w // 2 - rec_surf.get_width() // 2, rec_y))
            rec_y += 24

        hint_y = max(center_y + 190, rec_y + 10)
        hint_text = "SPACE to replay  |  L to loop weak section  |  ESC to menu"
        hint_surf = _font_cache.render(hint_font, hint_text, True, t.hud_text)
        surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, hint_y))
    else:
        hint_text = "SPACE to replay  |  ESC to menu"
        hint_surf = _font_cache.render(hint_font, hint_text, True, t.hud_text)
        surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, center_y + 70))

    if state.timing_judge and state.timing_summary is not None:
        draw_timing_summary(
            surface,
            layout,
            state.timing_summary,
            state.timing_worst_measures,
            state.matcher.get_timing_observations() if state.matcher else [],
        )

    # Technique Heatmap panel (Phase 1)
    if state.technique_heatmap:
        draw_technique_heatmap(surface, layout, state.technique_heatmap,
                                state.drill_recommendation)


def draw_timing_summary(
    surface: pygame.Surface,
    layout,
    stats: "TimingStats",
    worst_measures: list[tuple[int, float, float]],
    observations: list,
) -> None:
    """Draw the Timing Judge results panel within the completion overlay."""
    t = get_theme()
    w, h = layout.screen_w, layout.screen_h

    stat_font = _get_font("consolas", 22)
    small_font = _get_font("arial", 16)
    hist_font = _get_font("consolas", 14)

    panel_x = w - 460
    panel_y = h // 2 - 100

    title_surf = _font_cache.render(stat_font, "Timing Judge", True, t.hud_accent)
    surface.blit(title_surf, (panel_x, panel_y))
    panel_y += 30

    if stats.count == 0:
        no_data = _font_cache.render(
            small_font,
            "No timing data -- enable audio (A) and play a take",
            True,
            t.hud_text,
        )
        surface.blit(no_data, (panel_x, panel_y))
        return

    mean_sign = "+" if stats.mean_error_ms >= 0 else ""
    mean_text = f"\u03bc = {mean_sign}{stats.mean_error_ms:.1f} ms"
    mean_color = t.timing_late if stats.mean_error_ms > 5 else (
        t.timing_early if stats.mean_error_ms < -5 else t.timing_on_time
    )
    mean_surf = _font_cache.render(stat_font, mean_text, True, mean_color)
    surface.blit(mean_surf, (panel_x, panel_y))
    panel_y += 28

    std_text = f"\u03c3 = {stats.std_dev_ms:.1f} ms"
    std_surf = _font_cache.render(stat_font, std_text, True, t.hud_text)
    surface.blit(std_surf, (panel_x, panel_y))
    panel_y += 28

    counts_text = (
        f"Early: {stats.early_count}  On-time: {stats.on_time_count}  "
        f"Late: {stats.late_count}"
    )
    counts_surf = _font_cache.render(small_font, counts_text, True, t.hud_text)
    surface.blit(counts_surf, (panel_x, panel_y))
    panel_y += 20

    missed_extra = f"Missed: {stats.missed_count}  Extra: {stats.extra_count}"
    me_surf = _font_cache.render(small_font, missed_extra, True, t.hud_text)
    surface.blit(me_surf, (panel_x, panel_y))
    panel_y += 28

    hist_label = _font_cache.render(small_font, "Error Distribution", True, t.hud_accent)
    surface.blit(hist_label, (panel_x, panel_y))
    panel_y += 20

    hist_w = 400
    hist_h = 100
    hist_x = panel_x
    hist_y = panel_y

    pygame.draw.rect(surface, t.signal_cold, (hist_x, hist_y, hist_w, hist_h))

    max_bin = max(stats.histogram_bins) if stats.histogram_bins else 0
    bar_w = hist_w // len(stats.histogram_bins)
    center_bin = len(stats.histogram_bins) // 2

    for i, count in enumerate(stats.histogram_bins):
        if count == 0 or max_bin == 0:
            continue
        bar_h = int((count / max_bin) * (hist_h - 4))
        bx = hist_x + i * bar_w
        by = hist_y + hist_h - bar_h
        if i < center_bin:
            color = t.timing_early
        elif i > center_bin:
            color = t.timing_late
        else:
            color = t.timing_on_time
        pygame.draw.rect(surface, color, (bx + 1, by, bar_w - 2, bar_h))

    center_x = hist_x + center_bin * bar_w + bar_w // 2
    pygame.draw.line(
        surface,
        t.hud_text,
        (center_x, hist_y - 2),
        (center_x, hist_y + hist_h + 2),
        1,
    )

    left_label = _font_cache.render(hist_font, "-100ms", True, t.hud_text)
    right_label = _font_cache.render(hist_font, "+100ms", True, t.hud_text)
    zero_label = _font_cache.render(hist_font, "0", True, t.hud_text)
    surface.blit(left_label, (hist_x, hist_y + hist_h + 2))
    surface.blit(
        right_label,
        (hist_x + hist_w - right_label.get_width(), hist_y + hist_h + 2),
    )
    surface.blit(
        zero_label,
        (center_x - zero_label.get_width() // 2, hist_y + hist_h + 2),
    )
    panel_y += hist_h + 22

    if worst_measures:
        worst_label = _font_cache.render(small_font, "Worst Bars:", True, t.hud_accent)
        surface.blit(worst_label, (panel_x, panel_y))
        panel_y += 20
        for measure_idx, mean_err, std_dev in worst_measures:
            sign = "+" if mean_err >= 0 else ""
            line = f"  Bar {measure_idx + 1}: \u03bc={sign}{mean_err:.0f}ms  \u03c3={std_dev:.0f}ms"
            worst_color = t.timing_late if mean_err > 5 else (
                t.timing_early if mean_err < -5 else t.timing_on_time
            )
            worst_surf = _font_cache.render(small_font, line, True, worst_color)
            surface.blit(worst_surf, (panel_x, panel_y))
            panel_y += 18

    if stats.timing_slope_ms_per_measure != 0.0:
        trend_text = f"Trend: {stats.trend} ({stats.timing_slope_ms_per_measure:+.2f} ms/measure)"
        trend_color = t.timing_on_time if stats.trend in ("stable", "improving") else t.feedback_close
        trend_surf = _font_cache.render(small_font, trend_text, True, trend_color)
        surface.blit(trend_surf, (panel_x, panel_y))
        panel_y += 18

    art_counts: dict[str, int] = {}
    for obs in observations:
        if obs.articulation:
            art_counts[obs.articulation] = art_counts.get(obs.articulation, 0) + 1
    if art_counts:
        art_label = _font_cache.render(small_font, "Articulations:", True, t.hud_accent)
        surface.blit(art_label, (panel_x, panel_y))
        panel_y += 18
        for art_name, count in sorted(art_counts.items()):
            display = art_name.replace("_", " ").title()
            line = f"  {display}: {count}"
            art_surf = _font_cache.render(small_font, line, True, t.hud_text)
            surface.blit(art_surf, (panel_x, panel_y))
            panel_y += 16

def draw_technique_heatmap(
    surface: pygame.Surface,
    layout,
    heatmap: dict[str, dict[str, float]],
    drill_recommendation: str | None,
) -> None:
    """Draw the Technique Heatmap panel within the completion overlay.

    One row per technique kind present in the run, sorted by accuracy ascending
    (weakest first). Below the table, render the drill recommendation line.
    """
    t = get_theme()
    w, h = layout.screen_w, layout.screen_h
    panel_font = _get_font("consolas", 16)
    label_font = _get_font("arial", 14)

    # Sort by accuracy ascending (weakest first)
    rows = sorted(
        heatmap.items(),
        key=lambda kv: kv[1].get("accuracy", 100.0),
    )
    panel_h = 40 + len(rows) * 20 + (30 if drill_recommendation else 0)
    panel_w = 360
    panel_x = w // 2 - panel_w // 2
    panel_y = h - panel_h - 40

    # Background panel
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 180))
    surface.blit(panel, (panel_x, panel_y))

    title_surf = _font_cache.render(panel_font, "Technique Heatmap", True, t.hud_accent)
    surface.blit(title_surf, (panel_x + 12, panel_y + 8))

    row_y = panel_y + 32
    for kind, data in rows:
        acc = data.get("accuracy", 0.0)
        count = int(data.get("count", 0))
        display = kind.replace("_", " ").title()
        line = f"{display:<10} {acc:>3.0f}%  ({count} notes)"
        color = t.hud_text if acc >= 70 else (t.feedback_close if acc < 50 else t.hud_accent)
        line_surf = _font_cache.render(label_font, line, True, color)
        surface.blit(line_surf, (panel_x + 12, row_y))
        # Bar
        bar_x = panel_x + 200
        bar_w = int((panel_w - 220) * (acc / 100.0))
        if bar_w > 0:
            pygame.draw.rect(surface, color, (bar_x, row_y + 6, bar_w, 6))
        row_y += 20

    if drill_recommendation:
        drill_surf = _font_cache.render(label_font, drill_recommendation, True, t.hud_accent)
        surface.blit(drill_surf, (panel_x + 12, row_y + 4))


def draw_why_missed(
    surface: pygame.Surface,
    verdicts: list,
    font,
    x: int,
    y: int,
) -> None:
    """Render the last 3 failed-technique verdict explanations on the HUD.

    Called from the HUD region in scrolling.py — replaces the old single
    articulation label with a rolling display of verdict explanations.
    """
    t = get_theme()
    # Show up to 3 most recent failed/weak/missed verdicts
    failed = [v for v in verdicts if v.grade in ("missed", "weak")]
    for i, v in enumerate(failed[-3:]):
        text = v.explanation
        color = t.feedback_close if v.grade == "missed" else t.hud_accent
        surf = _font_cache.render(font, text, True, color)
        surface.blit(surf, (x, y + i * 18))


__all__ = [
    "CompletionState",
    "draw_completion_overlay",
    "draw_help_overlay",
    "draw_timing_summary",
    "draw_technique_heatmap",
    "draw_why_missed",
]
