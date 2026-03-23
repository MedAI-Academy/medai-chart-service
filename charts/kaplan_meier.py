"""
Kaplan-Meier Renderer — NEJM / Lancet Publication Quality

Design principles:
  - Clean serif/sans-serif typography matching top-tier journals
  - Muted, colorblind-safe palette (NEJM style)
  - Step function with optional 95% CI shading
  - Censoring marks as small vertical ticks
  - Median survival dashed lines
  - Number-at-risk table below x-axis
  - HR annotation box
  - High DPI for PPTX embedding (300 DPI default)
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from matplotlib import patheffects

# ── NEJM-inspired color palette (colorblind-safe) ──────────────
PALETTE = [
    "#2166AC",  # deep blue
    "#D6604D",  # warm red
    "#4DAC26",  # forest green
    "#B2ABD2",  # muted purple
]

# ── Global style settings ──────────────────────────────────────
FONT_FAMILY = "DejaVu Sans"  # clean sans-serif, universally available
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10.5
RISK_SIZE = 9


def kaplan_meier_estimator(times, events):
    """
    Compute KM survival curve with Greenwood 95% CI.
    Returns: (unique_times, survival, ci_lower, ci_upper, at_risk_at_time)
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    # Sort by time
    order = np.argsort(times)
    times = times[order]
    events = events[order]

    unique_times = np.unique(times)
    n = len(times)

    surv = 1.0
    var_sum = 0.0

    km_times = [0.0]
    km_surv = [1.0]
    km_lower = [1.0]
    km_upper = [1.0]

    for t in unique_times:
        mask = times == t
        at_risk = np.sum(times >= t)
        d = np.sum(events[mask])

        if at_risk > 0 and d > 0:
            surv *= (1 - d / at_risk)
            if at_risk > d:
                var_sum += d / (at_risk * (at_risk - d))

        se = surv * np.sqrt(var_sum) if var_sum > 0 else 0
        lower = max(0, surv - 1.96 * se)
        upper = min(1, surv + 1.96 * se)

        km_times.append(t)
        km_surv.append(surv)
        km_lower.append(lower)
        km_upper.append(upper)

    return (
        np.array(km_times),
        np.array(km_surv),
        np.array(km_lower),
        np.array(km_upper),
    )


def get_at_risk_counts(times, events, eval_times):
    """Number at risk at specific evaluation times."""
    times = np.asarray(times, dtype=float)
    counts = []
    for t in eval_times:
        counts.append(int(np.sum(times >= t)))
    return counts


def get_median_survival(km_times, km_surv):
    """Find time where survival crosses 0.5."""
    for i in range(1, len(km_surv)):
        if km_surv[i] <= 0.5:
            return km_times[i]
    return None


def render_kaplan_meier(req) -> io.BytesIO:
    """Render publication-quality KM curve, return PNG buffer."""

    # ── Figure setup ────────────────────────────────────────────
    has_risk_table = req.show_at_risk and len(req.arms) <= 4
    n_arms = len(req.arms)

    # Extra bottom space for at-risk table
    bottom_margin = 0.22 if has_risk_table else 0.12
    fig_height = req.height + (0.4 * n_arms if has_risk_table else 0)

    fig, ax = plt.subplots(
        figsize=(req.width, fig_height),
        dpi=req.dpi,
    )
    fig.subplots_adjust(
        left=0.14, right=0.92, top=0.90,
        bottom=0.22 + (0.05 * n_arms if has_risk_table else 0)
    )

    # ── Style ───────────────────────────────────────────────────
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    # ── Determine x-axis range ──────────────────────────────────
    all_times = []
    for arm in req.arms:
        all_times.extend(arm.times)
    max_time = max(all_times) if all_times else 12
    # Round up to nice interval
    if max_time <= 12:
        x_max = 12
        x_step = 3
    elif max_time <= 24:
        x_max = 24
        x_step = 6
    elif max_time <= 36:
        x_max = 36
        x_step = 6
    elif max_time <= 60:
        x_max = 60
        x_step = 12
    else:
        x_max = int(np.ceil(max_time / 12) * 12)
        x_step = 12

    eval_times = np.arange(0, x_max + 1, x_step)

    # ── Plot each arm ───────────────────────────────────────────
    arm_data = []
    for i, arm in enumerate(req.arms):
        color = arm.color or PALETTE[i % len(PALETTE)]
        km_t, km_s, km_lo, km_hi = kaplan_meier_estimator(arm.times, arm.events)
        median = get_median_survival(km_t, km_s)
        at_risk = get_at_risk_counts(arm.times, arm.events, eval_times)

        arm_data.append({
            "label": arm.label,
            "color": color,
            "km_t": km_t,
            "km_s": km_s,
            "km_lo": km_lo,
            "km_hi": km_hi,
            "median": median,
            "at_risk": at_risk,
        })

        # Step curve (survival × 100 for percentage)
        ax.step(
            km_t, km_s * 100,
            where="post",
            color=color,
            linewidth=2.2,
            label=f"{arm.label}" + (f"  (median: {median:.1f} mo)" if median else ""),
            zorder=3,
        )

        # 95% CI shading
        if req.show_ci:
            ax.fill_between(
                km_t, km_lo * 100, km_hi * 100,
                step="post",
                alpha=0.12,
                color=color,
                zorder=2,
            )

        # Censoring marks
        if req.show_censoring:
            censor_mask = np.array(arm.events) == 0
            censor_times = np.array(arm.times)[censor_mask]
            for ct in censor_times:
                # Find survival at censor time
                idx = np.searchsorted(km_t, ct, side="right") - 1
                idx = max(0, min(idx, len(km_s) - 1))
                s_val = km_s[idx] * 100
                ax.plot(
                    ct, s_val, "|",
                    color=color, markersize=8, markeredgewidth=1.5,
                    zorder=4,
                )

        # Median survival lines
        if req.show_median and median is not None:
            ax.plot(
                [median, median], [0, 50],
                linestyle="--", color=color, linewidth=1.0, alpha=0.6, zorder=2,
            )
            ax.plot(
                [0, median], [50, 50],
                linestyle="--", color=color, linewidth=1.0, alpha=0.6, zorder=2,
            )

    # ── Axes formatting ─────────────────────────────────────────
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE, length=5,
                   color="#666666", labelcolor="#333333")
    ax.tick_params(axis="y", which="minor", length=3, color="#999999")

    # Subtle grid
    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.4,
                  color="#E0E0E0", zorder=0)
    ax.yaxis.grid(True, which="minor", linestyle=":", linewidth=0.3,
                  color="#EEEEEE", zorder=0)
    ax.xaxis.grid(False)

    # Labels
    ax.set_xlabel(
        req.xlabel or "Time (months)",
        fontsize=LABEL_SIZE, fontfamily=FONT_FAMILY,
        color="#333333", labelpad=8,
    )
    ax.set_ylabel(
        req.ylabel or "Overall Survival (%)",
        fontsize=LABEL_SIZE, fontfamily=FONT_FAMILY,
        color="#333333", labelpad=8,
    )

    # Title
    if req.title:
        ax.set_title(
            req.title,
            fontsize=TITLE_SIZE, fontfamily=FONT_FAMILY,
            fontweight="bold", color="#1a1a1a",
            pad=16,
        )

    # ── Legend ───────────────────────────────────────────────────
    legend = ax.legend(
        loc="upper right",
        fontsize=LEGEND_SIZE,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        framealpha=0.95,
        borderpad=0.8,
        handlelength=2.5,
    )
    legend.get_frame().set_linewidth(0.8)

    # ── HR annotation box ───────────────────────────────────────
    if req.hr_text:
        ax.text(
            0.98, 0.35, req.hr_text,
            transform=ax.transAxes,
            fontsize=10, fontfamily=FONT_FAMILY,
            color="#333333",
            ha="right", va="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="#AAAAAA",
                linewidth=0.8,
                alpha=0.95,
            ),
        )

    # ── Number at Risk table ────────────────────────────────────
    if has_risk_table:
        row_height = 0.06
        header_y = -0.20
        first_row_y = header_y - 0.04  # gap between header and first row

        # Separator line at y=0
        ax.axhline(y=0, color="#CCCCCC", linewidth=0.6, zorder=1)

        # "No. at Risk" header
        ax.text(
            -0.02, header_y,
            "No. at Risk",
            transform=ax.transAxes,
            fontsize=RISK_SIZE - 0.5, fontfamily=FONT_FAMILY,
            fontweight="bold", fontstyle="italic", color="#777777",
            ha="right", va="top",
        )

        for i, ad in enumerate(arm_data):
            y_pos = first_row_y - (i * row_height)

            # Arm label (truncate if too long)
            label = ad["label"]
            if len(label) > 30:
                label = label[:28] + "…"
            ax.text(
                -0.02, y_pos,
                label,
                transform=ax.transAxes,
                fontsize=RISK_SIZE - 0.5, fontfamily=FONT_FAMILY,
                fontweight="semibold", color=ad["color"],
                ha="right", va="top",
            )

            # Counts at each eval time — aligned to x-axis ticks
            for j, t in enumerate(eval_times):
                x_data_frac = (t - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
                count = ad["at_risk"][j]
                ax.text(
                    x_data_frac, y_pos,
                    str(count),
                    transform=ax.transAxes,
                    fontsize=RISK_SIZE, fontfamily=FONT_FAMILY,
                    color="#444444",
                    ha="center", va="top",
                )

    # ── Export ───────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=req.dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf
