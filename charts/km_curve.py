"""
KM Curve Renderer — Renders Kaplan-Meier curves from EXTRACTED data points.

Unlike kaplan_meier.py (which computes KM from raw time/event data),
this module renders curves from pre-extracted km_points: [[time, survival%], ...]
as read by Claude Vision from publication PDFs.

Output: Publication-quality PNG matching NEJM/Lancet style.
"""

import io
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logger = logging.getLogger(__name__)

# ── NEJM-inspired palette ──
COLORS = ['#2166AC', '#D6604D', '#4DAC26', '#B2ABD2']
FONT = 'DejaVu Sans'


def render_km_curve(km_data):
    """
    Render KM curve PNG from extracted data.
    
    Args:
        km_data: dict with:
            arms: [{name, km_points: [[t,s],[t,s],...], median}]
            x_max: max x-axis value
            x_label, y_label: axis labels
            hr: {value, ci_low, ci_high, p}
            at_risk_times: [0, 6, 12, ...]
            at_risk_data: [{arm, counts: [n0, n6, ...]}]
            show_nar: bool
            show_censoring: bool
    
    Returns:
        PNG bytes
    """
    arms = km_data.get('arms', [])
    if not arms:
        return None

    n_arms = len(arms)
    x_max = km_data.get('x_max', 42)
    show_nar = km_data.get('show_nar', True)
    at_risk_data = km_data.get('at_risk_data', [])
    at_risk_times = km_data.get('at_risk_times', [])
    has_risk_table = show_nar and at_risk_data and at_risk_times

    # ── Figure setup ──
    fig_h = 7 + (0.4 * n_arms if has_risk_table else 0)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=300)
    fig.subplots_adjust(
        left=0.14, right=0.92, top=0.90,
        bottom=0.22 + (0.05 * n_arms if has_risk_table else 0)
    )

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')

    # ── X-axis range ──
    if x_max <= 12:
        x_step = 3
    elif x_max <= 24:
        x_step = 6
    elif x_max <= 48:
        x_step = 6
    else:
        x_step = 12

    # ── Plot each arm ──
    arm_info = []
    for i, arm in enumerate(arms):
        color = COLORS[i % len(COLORS)]
        name = arm.get('name', f'Arm {i+1}')
        points = arm.get('km_points', [])
        median = arm.get('median')

        if not points:
            continue

        # Convert points to arrays
        times = np.array([p[0] for p in points], dtype=float)
        survs = np.array([p[1] for p in points], dtype=float)

        # Normalize: if values > 1, they're percentages
        if np.max(survs) > 1.5:
            survs_pct = survs
        else:
            survs_pct = survs * 100

        # Label with median
        label = name
        if median:
            label += f'  (median: {median} mo)'

        # Step curve
        ax.step(times, survs_pct, where='post', color=color,
                linewidth=2.2, label=label, zorder=3)

        # 95% CI shading (light)
        # We don't have CI from extraction, skip

        # Median lines
        if median and float(median) > 0:
            med_val = float(median)
            ax.plot([med_val, med_val], [0, 50],
                    linestyle='--', color=color, linewidth=1.0, alpha=0.6, zorder=2)
            ax.plot([0, med_val], [50, 50],
                    linestyle='--', color=color, linewidth=1.0, alpha=0.6, zorder=2)

        arm_info.append({'name': name, 'color': color})

    # ── Axes ──
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.tick_params(axis='both', which='major', labelsize=10, length=5,
                   color='#666666', labelcolor='#333333')
    ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.4,
                  color='#E0E0E0', zorder=0)
    ax.yaxis.grid(True, which='minor', linestyle=':', linewidth=0.3,
                  color='#EEEEEE', zorder=0)

    ax.set_xlabel(km_data.get('x_label', 'Time (months)'),
                  fontsize=12, fontfamily=FONT, color='#333333', labelpad=8)
    ax.set_ylabel(km_data.get('y_label', 'Survival (%)'),
                  fontsize=12, fontfamily=FONT, color='#333333', labelpad=8)

    # ── Legend ──
    legend = ax.legend(loc='upper right', fontsize=10.5, frameon=True,
                       fancybox=False, edgecolor='#CCCCCC', framealpha=0.95,
                       borderpad=0.8, handlelength=2.5)
    legend.get_frame().set_linewidth(0.8)

    # ── HR box ──
    hr = km_data.get('hr')
    if hr and hr.get('value'):
        hr_text = f"HR {hr['value']}"
        if hr.get('ci_low') and hr.get('ci_high'):
            hr_text += f" (95% CI {hr['ci_low']}–{hr['ci_high']})"
        if hr.get('p'):
            hr_text += f"\nP = {hr['p']}"
        ax.text(0.98, 0.35, hr_text, transform=ax.transAxes,
                fontsize=10, fontfamily=FONT, color='#333333',
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='#AAAAAA', linewidth=0.8, alpha=0.95))

    # ── Number at risk table ──
    if has_risk_table:
        row_height = 0.06
        header_y = -0.20

        ax.axhline(y=0, color='#CCCCCC', linewidth=0.6, zorder=1)
        ax.text(-0.02, header_y, 'No. at Risk', transform=ax.transAxes,
                fontsize=8.5, fontfamily=FONT, fontweight='bold',
                fontstyle='italic', color='#777777', ha='right', va='top')

        for i, ard in enumerate(at_risk_data):
            y_pos = header_y - 0.04 - (i * row_height)
            arm_name = ard.get('arm', f'Arm {i+1}')
            color = COLORS[i % len(COLORS)]

            ax.text(-0.02, y_pos, arm_name[:30], transform=ax.transAxes,
                    fontsize=8.5, fontfamily=FONT, fontweight='semibold',
                    color=color, ha='right', va='top')

            counts = ard.get('counts', [])
            for j, t in enumerate(at_risk_times):
                if j >= len(counts):
                    break
                x_frac = (t - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
                ax.text(x_frac, y_pos, str(counts[j]), transform=ax.transAxes,
                        fontsize=9, fontfamily=FONT, color='#444444',
                        ha='center', va='top')

    # ── Export ──
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.read()
