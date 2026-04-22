"""
NEJM-style Kaplan-Meier renderer for Vision-extracted KM data.

Input contract: the dict produced by extract_km_vision_gemini.extract_km_vision().
  - curve_arms: list of {name, color, points:[{t,s}], censoring_times:[]}
  - curve_x_axis_unit, curve_x_axis_min, curve_x_axis_max
  - curve_y_axis_min, curve_y_axis_max (always 0..100 after normalisation)
  - arms (Pass 1 metadata): list of {name, n, color, median_value, median_unit}
  - nar_time_points, nar_time_unit, nar_arms
  - hr_value, hr_ci_low, hr_ci_high, p_value, p_value_operator
  - title, endpoint, x_label, y_label
  - median_validation: {verdict, arm_results, summary}
  - extraction_method_metadata / _nar / _curve

Output: PNG bytes ready to embed in a slide.

Design principles:
  - NEJM / Lancet figure conventions (minimal gridlines, clean typography,
    no chart-junk, consistent colour palette)
  - Faithful rendering of the extracted coordinates — we connect the
    dots as a step function, we do NOT smooth or resample
  - Censoring marks rendered exactly where the vision pass placed them
  - Validation footer bakes Tier 2 liability directly into the image:
    "Reconstructed from <paper>; median validated: extracted X vs
    published Y — match/warn". The renderer never lies about the quality
    of its own output.
  - Defensive: any missing field gets a sensible default rather than a
    crash. If curve_arms is empty, returns None so the caller can decide
    whether to fall back to a metadata-only slide.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# NEJM-inspired palette. Colourblind-safe, print-friendly.
# If the Vision pass returned a color word we try to match it, falling
# back to this palette in order.
# ─────────────────────────────────────────────────────────────────
PALETTE = [
    "#2166AC",  # deep blue — almost always arm 1
    "#B2182B",  # crimson   — almost always arm 2
    "#1B7837",  # forest green
    "#762A83",  # purple
    "#E08214",  # orange
]

# Map simple colour words from Vision output to hex. Each maps to a
# colour that's both visually correct AND consistent with NEJM's usual
# choices (blue=experimental, red=control is the common convention).
COLOR_WORD_TO_HEX = {
    "blue":   "#2166AC",
    "navy":   "#2166AC",
    "red":    "#B2182B",
    "crimson": "#B2182B",
    "green":  "#1B7837",
    "purple": "#762A83",
    "violet": "#762A83",
    "orange": "#E08214",
    "black":  "#1A1A1A",
    "gray":   "#555555",
    "grey":   "#555555",
    "brown":  "#8C510A",
    "teal":   "#01665E",
    "pink":   "#C994C7",
}

FONT = "DejaVu Sans"  # widely available, close enough to Helvetica/Arial


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def render_km_from_vision(vision_data: dict, source_name: Optional[str] = None) -> Optional[bytes]:
    """Render a NEJM-style KM curve from the Vision extraction output.

    Args:
        vision_data: dict matching extract_km_vision_gemini output schema
        source_name: optional paper/study name for the validation footer
                     (e.g. "DiNardo et al., NEJM 2020"). If None, uses
                     title from the vision data.

    Returns:
        PNG bytes, or None if there's no curve data to render (caller
        can fall back to metadata-only slide).
    """
    curve_arms = vision_data.get("curve_arms") or []
    if not curve_arms:
        logger.warning(
            f"render_km_from_vision: no curve_arms in vision_data "
            f"(source='{source_name}') — nothing to render"
        )
        return None

    # ── Extract the pieces we need, with defensive defaults ──
    x_min     = float(vision_data.get("curve_x_axis_min") or 0.0)
    x_max     = float(vision_data.get("curve_x_axis_max") or 36.0)
    x_unit    = vision_data.get("curve_x_axis_unit") or "months"
    x_label   = vision_data.get("x_label") or f"Time ({x_unit})"
    y_label   = vision_data.get("y_label") or "Survival (%)"
    title     = vision_data.get("title") or "Kaplan-Meier Curve"

    nar_time_points = vision_data.get("nar_time_points") or []
    nar_arms        = vision_data.get("nar_arms") or []
    has_nar_table   = bool(nar_time_points) and bool(nar_arms)

    metadata_arms = vision_data.get("arms") or []
    n_arms = len(curve_arms)

    # ── Figure layout ──
    # Risk table takes 0.8 inch base + 0.25 per arm; disabled if no NaR
    nar_height_in = (0.8 + 0.25 * n_arms) if has_nar_table else 0
    fig_height = 5.5 + nar_height_in
    fig_width  = 9.0

    if has_nar_table:
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=200)
        gs = GridSpec(
            2, 1,
            height_ratios=[5.5, nar_height_in],
            hspace=0.12,
            figure=fig,
        )
        ax     = fig.add_subplot(gs[0])
        ax_nar = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
        ax_nar = None

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Axis styling — NEJM convention ──
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")

    # ── Resolve colours for each arm ──
    arm_colors = _resolve_arm_colors(curve_arms, metadata_arms)

    # ── Plot each arm ──
    for i, arm in enumerate(curve_arms):
        color = arm_colors[i]
        pts = arm.get("points") or []
        if len(pts) < 2:
            continue

        # The vision pass already validated monotonicity and sorted by t,
        # but we defensively re-sort in case caller supplied raw data.
        pts_sorted = sorted(pts, key=lambda p: p["t"])
        times = [p["t"] for p in pts_sorted]
        survs = [p["s"] for p in pts_sorted]

        # Build step-function path (left-continuous, i.e. horizontal then
        # vertical drop — matches KM convention where survival changes at
        # event times)
        step_x, step_y = _build_step_path(times, survs)

        # Match arm name to metadata arm for legend label with N
        display_label = _build_legend_label(arm, metadata_arms)

        ax.plot(
            step_x, step_y,
            color=color,
            linewidth=1.8,
            label=display_label,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            zorder=3,
        )

        # Censoring marks — small vertical ticks at extracted censoring times
        cens_times = arm.get("censoring_times") or []
        if cens_times:
            _draw_censoring_ticks(ax, cens_times, times, survs, color)

    # ── Axes config ──
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 105)  # tiny headroom above 100%

    # X-axis tick spacing — pick a round interval appropriate for the range
    x_step = _pick_x_step(x_max - x_min)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.tick_params(
        axis="both", which="major",
        labelsize=10, length=5, width=1.0,
        direction="out", color="#000000", labelcolor="#000000",
    )
    ax.tick_params(axis="y", which="minor", length=3, color="#888888")
    ax.tick_params(axis="x", which="minor", length=0)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))

    # Subtle horizontal grid at 50% for median reference
    ax.axhline(50, color="#CCCCCC", linestyle=":", linewidth=0.7, zorder=1)

    # Axis labels
    ax.set_xlabel(x_label, fontsize=11, fontfamily=FONT, color="#000000", labelpad=8)
    ax.set_ylabel(y_label, fontsize=11, fontfamily=FONT, color="#000000", labelpad=8)

    # Title
    ax.set_title(
        title,
        fontsize=13, fontfamily=FONT, fontweight="bold",
        color="#000000", pad=12, loc="left",
    )

    # ── Legend (NEJM: inside upper right, no frame) ──
    legend = ax.legend(
        loc="upper right",
        fontsize=10,
        frameon=False,
        handlelength=2.2,
        labelspacing=0.5,
    )
    for text in legend.get_texts():
        text.set_fontfamily(FONT)
        text.set_color("#000000")

    # ── HR annotation box (NEJM: lower left or mid-plot, plain box) ──
    hr_text = _build_hr_text(vision_data)
    if hr_text:
        ax.text(
            0.98, 0.05, hr_text,
            transform=ax.transAxes,
            fontsize=9, fontfamily=FONT, color="#000000",
            ha="right", va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#666666",
                linewidth=0.8,
                alpha=0.95,
            ),
            zorder=5,
        )

    # ── Number-at-Risk table ──
    if ax_nar is not None:
        _draw_nar_table(ax_nar, nar_time_points, nar_arms, arm_colors, curve_arms, x_min, x_max)

    # ── Validation footer ──
    # Liability-critical: this footer makes the provenance explicit.
    footer = _build_validation_footer(vision_data, source_name)
    if footer:
        fig.text(
            0.015, 0.012, footer,
            fontsize=7, fontfamily=FONT, color="#666666",
            ha="left", va="bottom", style="italic",
        )

    # ── Export ──
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.10)
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=200,
        bbox_inches="tight", facecolor="white", edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────
# Helpers — all private, straightforward
# ─────────────────────────────────────────────────────────────────

def _resolve_arm_colors(curve_arms: list, metadata_arms: list) -> list[str]:
    """Pick a hex colour for each curve arm.

    Priority:
      1. Vision-supplied color word (mapped via COLOR_WORD_TO_HEX)
      2. Fall back to PALETTE by index
    We don't use metadata_arms for colour — the Vision pass sees the
    actual legend swatch and is more reliable.
    """
    out = []
    for i, arm in enumerate(curve_arms):
        color_word = (arm.get("color") or "").lower().strip()
        if color_word and color_word in COLOR_WORD_TO_HEX:
            out.append(COLOR_WORD_TO_HEX[color_word])
        else:
            out.append(PALETTE[i % len(PALETTE)])
    return out


def _build_step_path(times: list, survs: list) -> tuple[list, list]:
    """Build a step-function path from (t, s) pairs.

    Convention: horizontal segment at prev_s until next time t, then
    vertical drop to new s. This matches the "left-continuous" KM step
    function where survival holds constant between events and drops
    instantly at event time.
    """
    if not times:
        return [], []

    step_x = [times[0]]
    step_y = [survs[0]]
    for j in range(1, len(times)):
        # Horizontal to next time point (stay at previous survival)
        step_x.append(times[j])
        step_y.append(survs[j - 1])
        # Vertical drop (only if survival actually changed)
        if survs[j] != survs[j - 1]:
            step_x.append(times[j])
            step_y.append(survs[j])
    return step_x, step_y


def _draw_censoring_ticks(ax, cens_times: list, curve_times: list, curve_survs: list, color: str) -> None:
    """Draw small vertical ticks on the curve at each censoring time.

    For each censoring time, we look up where the curve is at that time
    and draw a tick there.
    """
    for ct in cens_times:
        # Find survival at censor time: step-function interpolation
        s_at_ct = _survival_at_time(curve_times, curve_survs, ct)
        if s_at_ct is None:
            continue
        ax.plot(
            [ct], [s_at_ct],
            marker="|",
            markersize=7,
            markeredgewidth=1.3,
            color=color,
            zorder=4,
        )


def _survival_at_time(times: list, survs: list, t: float) -> Optional[float]:
    """Step-function lookup: survival at time t."""
    if not times or t < times[0]:
        return None
    for i in range(len(times) - 1):
        if times[i] <= t < times[i + 1]:
            return survs[i]
    if t >= times[-1]:
        return survs[-1]
    return None


def _build_legend_label(curve_arm: dict, metadata_arms: list) -> str:
    """Build legend label: '<name> (n=N)' if we can match against metadata."""
    name = curve_arm.get("name") or "Arm"
    # Match metadata arm by name substring
    n_val = None
    for md_arm in metadata_arms:
        md_name = (md_arm.get("name") or "").lower()
        cv_name = name.lower()
        if md_name and cv_name and (md_name == cv_name or md_name in cv_name or cv_name in md_name):
            n_val = md_arm.get("n")
            break
    if n_val:
        return f"{name} (n={n_val})"
    return name


def _build_hr_text(vision_data: dict) -> Optional[str]:
    """Build the HR annotation text from the extracted HR/CI/p fields.

    Returns None if no HR was extracted (box will be skipped entirely).
    """
    hr = vision_data.get("hr_value")
    if hr is None:
        return None
    lines = []
    lo = vision_data.get("hr_ci_low")
    hi = vision_data.get("hr_ci_high")
    if lo is not None and hi is not None:
        lines.append(f"HR {hr:.2f} (95% CI {lo:.2f}–{hi:.2f})")
    else:
        lines.append(f"HR {hr:.2f}")

    p = vision_data.get("p_value")
    op = vision_data.get("p_value_operator") or ""
    if p is not None:
        # Format <0.001 as a common convention; other ops keep as printed
        if op == "<" and p <= 0.001:
            lines.append("P<0.001")
        elif op:
            lines.append(f"P{op}{p:g}")
        else:
            lines.append(f"P={p:g}")

    return "\n".join(lines)


def _draw_nar_table(
    ax_nar,
    nar_time_points: list,
    nar_arms: list,
    arm_colors: list,
    curve_arms: list,
    x_min: float,
    x_max: float,
) -> None:
    """Draw the Number-at-Risk table below the main plot.

    Layout: one row per arm, columns at each nar_time_point (aligned to
    the x-axis). Arm label on the left, colour-matched to the curve.
    """
    n_rows = len(nar_arms)
    ax_nar.set_facecolor("white")
    ax_nar.set_xlim(x_min, x_max)
    ax_nar.set_ylim(0, n_rows + 0.7)

    # Hide spines and ticks — this is a transparent data table
    for spine in ax_nar.spines.values():
        spine.set_visible(False)
    ax_nar.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Header
    ax_nar.text(
        x_min - (x_max - x_min) * 0.02,
        n_rows + 0.4,
        "No. at Risk",
        fontsize=9, fontfamily=FONT, fontweight="bold", fontstyle="italic",
        color="#333333",
        ha="right", va="center",
    )

    # Match NaR arm names to curve arm colours so rows are colour-consistent
    # with the plot above
    for i, nar_arm in enumerate(nar_arms):
        y_pos = n_rows - i - 0.3
        arm_name = nar_arm.get("name") or f"Arm {i+1}"

        # Find matching curve arm for colour (by name substring)
        row_color = "#000000"
        for ci, ca in enumerate(curve_arms):
            cv_name = (ca.get("name") or "").lower()
            nr_name = arm_name.lower()
            if cv_name and nr_name and (cv_name == nr_name or cv_name in nr_name or nr_name in cv_name):
                row_color = arm_colors[ci]
                break

        # Arm label (truncated if very long)
        display_name = arm_name if len(arm_name) <= 28 else arm_name[:26] + "…"
        ax_nar.text(
            x_min - (x_max - x_min) * 0.02,
            y_pos,
            display_name,
            fontsize=9, fontfamily=FONT, fontweight="normal",
            color=row_color,
            ha="right", va="center",
        )

        # Counts at each time point
        counts = nar_arm.get("counts") or []
        for j, t in enumerate(nar_time_points):
            if j >= len(counts):
                break
            ax_nar.text(
                float(t), y_pos,
                str(counts[j]),
                fontsize=9, fontfamily=FONT, color="#000000",
                ha="center", va="center",
            )


def _pick_x_step(range_width: float) -> float:
    """Pick a round tick spacing for the x-axis.

    Heuristic: aim for 5-8 major ticks along the range.
    """
    if range_width <= 12:
        return 2
    elif range_width <= 24:
        return 3
    elif range_width <= 36:
        return 6
    elif range_width <= 60:
        return 12
    elif range_width <= 120:
        return 24
    else:
        return round(range_width / 6)


def _build_validation_footer(vision_data: dict, source_name: Optional[str]) -> str:
    """Build the provenance + validation footer string.

    Examples:
      "Reconstructed from DiNardo et al., NEJM 2020; median validated:
       extracted 14.7 vs published 14.7 mo — match."

      "Reconstructed from Motzer et al., NEJM 2021; median extracted
       22.8 vs published 23.9 mo (4.6% deviation)."

      "Reconstructed via AI vision extraction; no published median anchor."
    """
    parts = []

    # Provenance
    src = source_name or vision_data.get("title") or ""
    if src:
        parts.append(f"Reconstructed from {src}")
    else:
        parts.append("Reconstructed via AI vision extraction")

    # Median validation summary
    mv = vision_data.get("median_validation") or {}
    arm_results = mv.get("arm_results") or []
    verdict = mv.get("verdict")

    validated_arms = [
        ar for ar in arm_results
        if ar.get("implicit_median") is not None and ar.get("published_median") is not None
    ]

    if validated_arms:
        # Report the worst arm's match (most informative)
        worst = max(validated_arms, key=lambda ar: ar.get("relative_error") or 0)
        extracted = worst["implicit_median"]
        published = worst["published_median"]
        rel_err = worst.get("relative_error") or 0

        if verdict == "match":
            parts.append(
                f"median validated: extracted {extracted:.1f} vs "
                f"published {published:.1f} mo ({rel_err*100:.1f}% deviation — match)"
            )
        elif verdict == "needs_reextract":
            retried = vision_data.get("curve_reextract_attempted")
            suffix = " after retry" if retried else ""
            parts.append(
                f"median validated{suffix}: extracted {extracted:.1f} vs "
                f"published {published:.1f} mo ({rel_err*100:.1f}% deviation — warn)"
            )
    elif arm_results:
        # Arms present but none validated against a published anchor
        parts.append("no published median anchors available to validate against")

    # Extraction provenance (small — users mostly care it's AI-extracted)
    method = vision_data.get("extraction_method_curve") or ""
    if method:
        parts.append(f"vision: {method}")

    return "  •  ".join(parts)
