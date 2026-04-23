"""
medaccur Forest Plot Renderer — NEJM Publication Quality

Reconstructs a subgroup forest plot from structured input (subgroups with HR
and 95% CI). Output is the platform's own visualisation — NOT a crop of the
original figure. Liability model matches Kaplan-Meier curves: data from paper,
visualisation is ours ("Reconstructed from [ref]"), confidence tier 2.

Input contract:
    render_forest_nejm(
        subgroups=[
          {"name": "All patients", "n": "286 vs 145",
           "hr": 0.64, "ci_low": 0.50, "ci_high": 0.82, "is_overall": True},
          {"category": "Sex", "is_header": True},
          {"name": "Female", "n": "114 vs 58",
           "hr": 0.68, "ci_low": 0.46, "ci_high": 1.02},
          ...
        ],
        title="VIALE-A: Subgroup Analysis of Overall Survival",
        subtitle="Hazard Ratio for Death",
        favours_left="Ven+Aza better",
        favours_right="Placebo better",
        reference_line=1.0,
        source="DiNardo et al., NEJM 2020",
    ) -> bytes  (PNG)

Invalid numeric rows are silently dropped (no fabrication).
"""

from __future__ import annotations

import io
import math
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─── Style constants ────────────────────────────────────────────────────
_FONT_FAMILY = "serif"
_FONT_FALLBACK = ["DejaVu Serif", "Times New Roman", "Liberation Serif"]

_COLOR_MARKER = "#1E1B4B"       # deep indigo — matches medaccur header
_COLOR_DIAMOND = "#4F46E5"      # brighter for the overall diamond
_COLOR_CI = "#1E1B4B"
_COLOR_REFLINE = "#111827"      # near-black dashed null line
_COLOR_HEADER = "#0F172A"
_COLOR_TEXT = "#1F2937"
_COLOR_MUTED = "#6B7280"
_COLOR_ZEBRA = "#F6F7F9"

_HR_AXIS_MIN = 0.1
_HR_AXIS_MAX = 10.0

# Row heights
_ROW_H = 0.32            # inches per row — compact NEJM-style
_HEADER_MARGIN = 0.95    # inches for title + column headers at top
_FOOTER_MARGIN = 1.55    # inches for axis + favours + reconstructed-from
                         # (raised from 1.25 on 2026-04-24 to prevent overlap
                         # between "Hazard Ratio" axis title and favours labels
                         # after bumping their font sizes to 11pt)


# ─── Helpers ────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _parse_n_int(n_str: str) -> int:
    """Pull the leading integer out of an N string. '286 vs 145' -> 286."""
    m = re.search(r"(\d+)", n_str or "")
    return int(m.group(1)) if m else 0


def _normalize_subgroups(subgroups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Uniform row list. Invalid numeric rows are dropped."""
    rows: list[dict[str, Any]] = []
    for sg in subgroups or []:
        if not isinstance(sg, dict):
            continue
        if sg.get("is_header"):
            label = sg.get("category") or sg.get("name") or ""
            if label:
                rows.append({"type": "header", "label": str(label)})
            continue
        try:
            hr = float(sg.get("hr"))
            ci_low = float(sg.get("ci_low"))
            ci_high = float(sg.get("ci_high"))
        except (TypeError, ValueError):
            continue
        if hr <= 0 or ci_low <= 0 or ci_high <= 0:
            continue
        rows.append({
            "type": "overall" if sg.get("is_overall") else "item",
            "label": str(sg.get("name", "")),
            "n": str(sg.get("n", "")),
            "hr": hr,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "hr_text": sg.get("hr_text") or f"{hr:.2f} ({ci_low:.2f}–{ci_high:.2f})",
        })
    return rows


# ─── Main render function ───────────────────────────────────────────────

def render_forest_nejm(
    subgroups: list[dict[str, Any]],
    title: str = "",
    subtitle: str = "",
    favours_left: str = "experimental better",
    favours_right: str = "control better",
    reference_line: float = 1.0,
    source: str = "",
    dpi: int = 300,
) -> bytes:
    """Render a NEJM-style subgroup forest plot and return PNG bytes."""

    rows = _normalize_subgroups(subgroups)
    if not rows:
        raise ValueError("render_forest_nejm: no valid subgroup rows provided")
    # Must have at least one row with numeric data (not only headers)
    if not any(r["type"] in ("item", "overall") for r in rows):
        raise ValueError("render_forest_nejm: all provided rows are headers — no data to plot")

    # ── Typography ─────────────────────────────────────────────────────
    plt.rcParams["font.family"] = _FONT_FAMILY
    plt.rcParams["font.serif"] = _FONT_FALLBACK

    n_rows = len(rows)

    # Dynamic figure height: rows × row-height + fixed margins
    fig_height = max(5.5, _HEADER_MARGIN + n_rows * _ROW_H + _FOOTER_MARGIN)
    # Width tuned 2026-04-24 to 10.0 (from 12.0 original, then 9.0 intermediate).
    # The PPTX slide embeds this figure at a fixed box width that may stretch
    # horizontally if the plot aspect ratio doesn't match; 10.0 × dynamic-height
    # is closer to the slide-box aspect than 12.0 was. Combined with the font-size
    # bumps below, this should keep text readable when embedded in a 16:9 slide.
    fig_width = 10.0

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor("white")

    # Layout: left 55 % table (Subgroup + N + HR text), right 45 % plot.
    # Zero gutter between — the zebra stripes will span both.
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[0.55, 0.45],
        left=0.04, right=0.97,
        top=1 - _HEADER_MARGIN / fig_height,
        bottom=_FOOTER_MARGIN / fig_height,
        wspace=0.0,
    )
    ax_table = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    # ── Title / subtitle ───────────────────────────────────────────────
    # Positioned in figure coords at fixed offsets from the top.
    if title:
        fig.text(
            0.04, 1.0 - 0.22 / fig_height, title,
            fontsize=15, fontweight="bold", color=_COLOR_HEADER,
            ha="left", va="top",
        )
    if subtitle:
        fig.text(
            0.04, 1.0 - 0.52 / fig_height, subtitle,
            fontsize=11, color=_COLOR_MUTED,
            ha="left", va="top", fontstyle="italic",
        )

    # ── Shared y-coordinate system ─────────────────────────────────────
    y_positions = list(range(n_rows))

    # ╔═══ LEFT: TABLE SIDE ═══╗
    # Column split: 0.0 – 0.66 subgroup label, 0.66 – 0.82 N, 0.82 – 1.0 HR (95% CI)
    ax_table.set_xlim(0.0, 1.0)
    ax_table.set_ylim(-1.2, n_rows - 0.5)
    ax_table.invert_yaxis()
    ax_table.axis("off")

    # Column headers
    ax_table.text(0.01, -0.85, "Subgroup",
                  fontsize=11, fontweight="bold", color=_COLOR_HEADER,
                  ha="left", va="center")
    ax_table.text(0.60, -0.85, "N",
                  fontsize=11, fontweight="bold", color=_COLOR_HEADER,
                  ha="center", va="center")
    ax_table.text(0.99, -0.85, "HR (95% CI)",
                  fontsize=11, fontweight="bold", color=_COLOR_HEADER,
                  ha="right", va="center")
    # separator line
    ax_table.plot([0.0, 1.0], [-0.5, -0.5],
                  color=_COLOR_HEADER, lw=0.8, clip_on=False)

    # ╔═══ RIGHT: PLOT SIDE ═══╗
    ax_plot.set_xlim(_HR_AXIS_MIN, _HR_AXIS_MAX)
    ax_plot.set_xscale("log")
    ax_plot.set_ylim(-1.2, n_rows - 0.5)
    ax_plot.invert_yaxis()

    # Column header above plot area
    ax_plot.text(1.0, -0.85, "",  # placeholder so axis doesn't collapse
                 transform=ax_plot.get_yaxis_transform(), fontsize=0)

    # ── Zebra stripes (span BOTH axes for visual cohesion) ──────────────
    # Stripe every other non-header body row.
    for y, row in zip(y_positions, rows):
        if row["type"] == "header":
            continue
        if y % 2 == 1:
            # Figure-coord patch spanning from table left to plot right
            gs_pos_table = ax_table.get_position()
            gs_pos_plot = ax_plot.get_position()
            # Convert y (data) to figure coords
            # Easier: draw in each axis's own coordinate system
            ax_table.add_patch(mpatches.Rectangle(
                (0.0, y - 0.5), 1.0, 1.0,
                facecolor=_COLOR_ZEBRA, edgecolor="none", zorder=0,
            ))
            ax_plot.axhspan(y - 0.5, y + 0.5,
                            facecolor=_COLOR_ZEBRA, edgecolor="none", zorder=0)

    # ── Reference line (HR = 1.0) ───────────────────────────────────────
    ax_plot.axvline(
        x=reference_line, color=_COLOR_REFLINE, lw=0.9, ls=(0, (4, 2)),
        zorder=1, alpha=0.85,
    )

    # ── Rows ────────────────────────────────────────────────────────────
    for y, row in zip(y_positions, rows):
        if row["type"] == "header":
            ax_table.text(
                0.01, y, row["label"],
                fontsize=11, fontweight="bold", color=_COLOR_HEADER,
                ha="left", va="center",
            )
            continue

        indent = "" if row["type"] == "overall" else "   "
        label_weight = "bold" if row["type"] == "overall" else "normal"
        label_color = _COLOR_HEADER if row["type"] == "overall" else _COLOR_TEXT
        label_size = 10.5

        ax_table.text(0.01, y, indent + row["label"],
                      fontsize=label_size, fontweight=label_weight,
                      color=label_color, ha="left", va="center")
        ax_table.text(0.60, y, row["n"],
                      fontsize=10, color=label_color,
                      ha="center", va="center")
        ax_table.text(0.99, y, row["hr_text"],
                      fontsize=10, fontweight=label_weight,
                      color=label_color,
                      ha="right", va="center")

        # Plot: CI whisker
        lo = _clamp(row["ci_low"], _HR_AXIS_MIN, _HR_AXIS_MAX)
        hi = _clamp(row["ci_high"], _HR_AXIS_MIN, _HR_AXIS_MAX)
        ax_plot.plot([lo, hi], [y, y],
                     color=_COLOR_CI, lw=1.0, solid_capstyle="butt", zorder=3)
        # endcaps
        cap_h = 0.16
        ax_plot.plot([lo, lo], [y - cap_h, y + cap_h],
                     color=_COLOR_CI, lw=1.0, zorder=3)
        ax_plot.plot([hi, hi], [y - cap_h, y + cap_h],
                     color=_COLOR_CI, lw=1.0, zorder=3)

        # Central marker — diamond for overall, size-scaled square for items
        hr = row["hr"]
        if row["type"] == "overall":
            ax_plot.scatter([hr], [y],
                            marker="D", s=120,
                            facecolor=_COLOR_DIAMOND, edgecolor=_COLOR_CI, lw=0.8,
                            zorder=4)
        else:
            n_int = _parse_n_int(row["n"])
            # Scale marker size with sqrt(N) for visual weight; clamp to sensible range
            base = 30
            scaled = base + min(math.sqrt(max(n_int, 1)) * 4.5, 80.0)
            ax_plot.scatter([hr], [y],
                            marker="s", s=scaled,
                            facecolor=_COLOR_MARKER, edgecolor=_COLOR_CI, lw=0.5,
                            zorder=4)

    # ── X-axis styling ─────────────────────────────────────────────────
    ticks = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    ax_plot.set_xticks(ticks)
    ax_plot.set_xticklabels([f"{t:g}" for t in ticks],
                            fontsize=8.5, color=_COLOR_TEXT)
    ax_plot.xaxis.set_minor_locator(plt.NullLocator())
    ax_plot.tick_params(axis="x", length=3, width=0.6, pad=2)
    ax_plot.tick_params(axis="y", left=False, labelleft=False)

    # Only the bottom axis visible
    for spine in ("top", "right", "left"):
        ax_plot.spines[spine].set_visible(False)
    ax_plot.spines["bottom"].set_color(_COLOR_HEADER)
    ax_plot.spines["bottom"].set_linewidth(0.8)

    ax_plot.set_xlabel(
        "Hazard Ratio (95% CI, log scale)",
        fontsize=11, color=_COLOR_HEADER, labelpad=5, fontweight="bold",
    )

    # ── "Favours" arrows + labels under the axis ──────────────────────
    # Arrows sit in a narrow band just under the plot axis spine.
    ax_plot_pos = ax_plot.get_position()
    arrow_y = ax_plot_pos.y0 - 0.042
    text_y = arrow_y - 0.022
    plot_mid_x = (ax_plot_pos.x0 + ax_plot_pos.x1) / 2
    plot_left_mid = (ax_plot_pos.x0 + plot_mid_x) / 2
    plot_right_mid = (plot_mid_x + ax_plot_pos.x1) / 2

    # Arrows pointing outward from null-line
    fig.add_artist(mpatches.FancyArrowPatch(
        (plot_mid_x - 0.01, arrow_y), (ax_plot_pos.x0 + 0.01, arrow_y),
        arrowstyle="->", color=_COLOR_HEADER, lw=0.9,
        mutation_scale=10, transform=fig.transFigure,
    ))
    fig.add_artist(mpatches.FancyArrowPatch(
        (plot_mid_x + 0.01, arrow_y), (ax_plot_pos.x1 - 0.01, arrow_y),
        arrowstyle="->", color=_COLOR_HEADER, lw=0.9,
        mutation_scale=10, transform=fig.transFigure,
    ))
    fig.text(plot_left_mid, text_y, favours_left,
             fontsize=11, fontweight="bold", color=_COLOR_HEADER,
             ha="center", va="top")
    fig.text(plot_right_mid, text_y, favours_right,
             fontsize=11, fontweight="bold", color=_COLOR_HEADER,
             ha="center", va="top")

    # ── "Reconstructed from …" liability footer — bottom edge, its own line ─
    footer_y = 0.015
    if source:
        fig.text(
            0.97, footer_y,
            f"Reconstructed from {source}",
            fontsize=7, color=_COLOR_MUTED,
            ha="right", va="bottom", fontstyle="italic",
        )
    fig.text(
        0.04, footer_y, "medaccur",
        fontsize=7, color=_COLOR_MUTED,
        ha="left", va="bottom",
    )

    # ── Export ─────────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, facecolor="white",
        bbox_inches="tight", pad_inches=0.2,
    )
    plt.close(fig)
    return buf.getvalue()


if __name__ == "__main__":
    # Full VIALE-A subgroup analysis (Figure 3 of DiNardo et al., NEJM 2020)
    sg = [
        {"name": "All patients", "n": "286 vs 145",
         "hr": 0.64, "ci_low": 0.50, "ci_high": 0.82, "is_overall": True},
        {"category": "Sex", "is_header": True},
        {"name": "Female", "n": "114 vs 58",  "hr": 0.68, "ci_low": 0.46, "ci_high": 1.02},
        {"name": "Male",   "n": "172 vs 87",  "hr": 0.62, "ci_low": 0.46, "ci_high": 0.85},
        {"category": "Age", "is_header": True},
        {"name": "<75 yr",  "n": "112 vs 58", "hr": 0.89, "ci_low": 0.59, "ci_high": 1.33},
        {"name": "≥75 yr",  "n": "174 vs 87", "hr": 0.54, "ci_low": 0.39, "ci_high": 0.73},
        {"category": "Geographic region", "is_header": True},
        {"name": "United States", "n": "50 vs 24",   "hr": 0.47, "ci_low": 0.26, "ci_high": 0.83},
        {"name": "Europe",        "n": "116 vs 59",  "hr": 0.67, "ci_low": 0.46, "ci_high": 0.97},
        {"name": "China",         "n": "24 vs 13",   "hr": 1.05, "ci_low": 0.35, "ci_high": 3.13},
        {"name": "Japan",         "n": "24 vs 13",   "hr": 0.52, "ci_low": 0.20, "ci_high": 1.33},
        {"name": "Rest of world", "n": "72 vs 36",   "hr": 0.73, "ci_low": 0.45, "ci_high": 1.17},
        {"category": "Baseline ECOG score", "is_header": True},
        {"name": "Grade <2", "n": "157 vs 81", "hr": 0.61, "ci_low": 0.44, "ci_high": 0.84},
        {"name": "Grade ≥2", "n": "129 vs 64", "hr": 0.70, "ci_low": 0.48, "ci_high": 1.03},
        {"category": "Type of AML", "is_header": True},
        {"name": "De novo",   "n": "214 vs 110", "hr": 0.67, "ci_low": 0.51, "ci_high": 0.90},
        {"name": "Secondary", "n": "72 vs 35",   "hr": 0.56, "ci_low": 0.35, "ci_high": 0.91},
        {"category": "Cytogenetic risk", "is_header": True},
        {"name": "Intermediate", "n": "182 vs 89", "hr": 0.57, "ci_low": 0.41, "ci_high": 0.79},
        {"name": "Poor",         "n": "104 vs 56", "hr": 0.78, "ci_low": 0.54, "ci_high": 1.12},
        {"category": "Molecular marker", "is_header": True},
        {"name": "IDH1 or IDH2", "n": "61 vs 28", "hr": 0.34, "ci_low": 0.20, "ci_high": 0.60},
        {"name": "TP53",         "n": "38 vs 14", "hr": 0.76, "ci_low": 0.40, "ci_high": 1.45},
        {"name": "NPM1",         "n": "27 vs 17", "hr": 0.73, "ci_low": 0.36, "ci_high": 1.51},
    ]
    png = render_forest_nejm(
        sg,
        title="VIALE-A: Subgroup Analysis of Overall Survival",
        subtitle="Hazard Ratio for Death (Ven+Aza vs Placebo+Aza)",
        favours_left="experimental better",
        favours_right="control better",
        source="DiNardo et al., NEJM 2020",
    )
    with open("/tmp/forest_smoke_v2.png", "wb") as f:
        f.write(png)
    print(f"Wrote {len(png)//1024} KB to /tmp/forest_smoke_v2.png")
