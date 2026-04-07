"""
KM Curve Renderer v2 — NEJM Publication-Quality
Renders Kaplan-Meier curves from extracted km_points data.
Matches NEJM/Lancet figure style as closely as possible.
"""

import io
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logger = logging.getLogger(__name__)

# ── NEJM-style colors (blue/red like original publications) ──
COLORS = ['#1B4F9B', '#C13A2A', '#2E8B57', '#8B6914']
FONT = 'DejaVu Sans'


def render_km_curve(km_data):
    """
    Render publication-quality KM curve PNG from extracted data.
    Matches NEJM Figure 2 style as closely as possible.
    """
    arms = km_data.get('arms', [])
    if not arms:
        return None

    n_arms = len(arms)
    x_max = float(km_data.get('x_max', 24))
    show_nar = km_data.get('show_nar', True)
    at_risk_data = km_data.get('at_risk_data', [])
    at_risk_times = km_data.get('at_risk_times', [])
    has_risk_table = show_nar and at_risk_data and at_risk_times

    # ── X-axis step calculation ──
    if x_max <= 12:
        x_step = 2
    elif x_max <= 24:
        x_step = 2
    elif x_max <= 36:
        x_step = 4
    else:
        x_step = 6

    # ── Figure layout ──
    # Risk table height
    nar_height = 0.8 + (0.25 * n_arms) if has_risk_table else 0
    fig_h = 5.5 + nar_height
    
    if has_risk_table:
        fig, (ax, ax_nar) = plt.subplots(
            2, 1, figsize=(8.5, fig_h), dpi=300,
            gridspec_kw={'height_ratios': [5.5, nar_height], 'hspace': 0.08}
        )
    else:
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
        ax_nar = None

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ── NEJM-style axis formatting ──
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')

    # ── Plot each arm ──
    for i, arm in enumerate(arms):
        color = COLORS[i % len(COLORS)]
        name = arm.get('name', f'Arm {i+1}')
        points = arm.get('km_points', [])
        median = arm.get('median')

        if not points:
            continue

        # Sort by time
        points = sorted(points, key=lambda p: (p[0], -p[1]))
        
        # Convert to numpy
        times = np.array([p[0] for p in points], dtype=float)
        survs = np.array([p[1] for p in points], dtype=float)

        # Normalize: if values ≤ 1.0, they're fractions → convert to %
        if np.max(survs) <= 1.5:
            survs = survs * 100

        # Ensure monotonically decreasing
        for j in range(1, len(survs)):
            if survs[j] > survs[j-1]:
                survs[j] = survs[j-1]

        # Build step-function path (proper KM style)
        # Each point represents an event or censoring
        step_x = [times[0]]
        step_y = [survs[0]]
        for j in range(1, len(times)):
            # Horizontal segment to this time point
            step_x.append(times[j])
            step_y.append(survs[j-1])  # stay at previous survival
            # Vertical drop (if survival changed)
            if survs[j] != survs[j-1]:
                step_x.append(times[j])
                step_y.append(survs[j])

        # Plot the step function
        ax.plot(step_x, step_y, color=color, linewidth=1.8,
                label=name, solid_capstyle='butt', zorder=3)

        # Add censoring marks (small vertical ticks) at regular intervals
        # where survival doesn't change (patients censored, not events)
        show_censoring = km_data.get('show_censoring', False)
        if show_censoring:
            for j in range(1, len(times)):
                if abs(survs[j] - survs[j-1]) < 0.5:  # no event = censoring
                    ax.plot(times[j], survs[j], '|', color=color,
                            markersize=4, markeredgewidth=0.8, zorder=4)

        # Median dashed lines
        if median and str(median).replace('.','').isdigit():
            med_val = float(median)
            if 0 < med_val < x_max:
                ax.plot([0, med_val], [50, 50], '--', color='#888888',
                        linewidth=0.6, zorder=1)
                ax.plot([med_val, med_val], [0, 50], '--', color='#888888',
                        linewidth=0.6, zorder=1)

    # ── Axes ──
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    ax.tick_params(axis='both', which='major', labelsize=10, length=5, width=1.0,
                   direction='out', color='#000000', labelcolor='#000000')
    ax.tick_params(axis='both', which='minor', length=0)

    # Y-axis: percentage format
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # Subtle grid (NEJM style — very faint)
    ax.yaxis.grid(False)  # NEJM typically has no grid

    ax.set_xlabel(km_data.get('x_label', 'Months since Randomization'),
                  fontsize=11, fontfamily=FONT, color='#000000', fontweight='normal', labelpad=8)
    ax.set_ylabel(km_data.get('y_label', 'Percent of Patients without Event'),
                  fontsize=11, fontfamily=FONT, color='#000000', fontweight='normal', labelpad=8)

    # ── Legend (NEJM style: inside plot, upper right) ──
    legend = ax.legend(loc='upper right', fontsize=10, frameon=False,
                       handlelength=2.0, labelspacing=0.4)
    for text in legend.get_texts():
        text.set_fontfamily(FONT)
        text.set_color('#000000')

    # ── HR annotation box (NEJM style: inside plot) ──
    hr = km_data.get('hr')
    if hr and hr.get('value'):
        lines = []
        lines.append(f"Hazard ratio for major organ deterioration,")
        lines.append(f"hematologic progression, or death, {hr['value']}")
        if hr.get('ci_low') and hr.get('ci_high'):
            lines.append(f"(95% CI, {hr['ci_low']} to {hr['ci_high']})")
        if hr.get('p'):
            lines.append(f"P={hr['p']}")
        hr_text = '\n'.join(lines)
        
        # Position: lower-center like NEJM
        ax.text(0.35, 0.18, hr_text, transform=ax.transAxes,
                fontsize=8.5, fontfamily=FONT, color='#000000',
                ha='center', va='top', linespacing=1.3,
                bbox=dict(boxstyle='square,pad=0.4', facecolor='white',
                          edgecolor='none', alpha=0.9))

    # ── Number at Risk table (NEJM style: below x-axis) ──
    if has_risk_table and ax_nar is not None:
        ax_nar.set_facecolor('white')
        ax_nar.set_xlim(0, x_max)
        ax_nar.set_ylim(0, n_arms + 0.5)
        
        # Remove all spines and ticks
        for spine in ax_nar.spines.values():
            spine.set_visible(False)
        ax_nar.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Title
        ax_nar.text(-0.02, n_arms + 0.3, 'No. at Risk', transform=ax_nar.get_yaxis_transform(),
                    fontsize=9, fontfamily=FONT, fontweight='bold', color='#000000',
                    ha='right', va='top')

        for i, ard in enumerate(at_risk_data):
            y_pos = n_arms - i - 0.2
            arm_name = ard.get('arm', f'Arm {i+1}')
            color = COLORS[i % len(COLORS)]

            # Arm label
            ax_nar.text(-0.02, y_pos, arm_name, transform=ax_nar.get_yaxis_transform(),
                        fontsize=9, fontfamily=FONT, fontweight='normal',
                        color=color, ha='right', va='center')

            # Patient counts at each time point
            counts = ard.get('counts', [])
            for j, t in enumerate(at_risk_times):
                if j >= len(counts):
                    break
                ax_nar.text(float(t), y_pos, str(counts[j]),
                            fontsize=8.5, fontfamily=FONT, color='#000000',
                            ha='center', va='center')

    # ── Export ──
    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.02)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.read()
