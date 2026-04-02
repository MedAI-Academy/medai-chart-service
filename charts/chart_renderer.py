"""
Chart Renderer Dispatcher — Routes layout ID to correct matplotlib chart.

Each chart function receives content dict and returns PNG bytes.
"""

import io
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ── medaccur color palette ──
PURPLE = '#7C6FFF'
TEAL = '#22D3A5'
ROSE = '#FF5F7E'
GOLD = '#F5C842'
SLATE = '#64748B'
DARK = '#1E293B'
MUTED = '#94A3B8'
COLORS = [PURPLE, TEAL, ROSE, GOLD, '#3B82F6', '#F97316']

# ── Common figure settings ──
DPI = 200
FONT = 'DejaVu Sans'


def setup_style():
    """Common style for all charts."""
    plt.rcParams.update({
        'font.family': FONT,
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelcolor': DARK,
        'xtick.color': SLATE,
        'ytick.color': SLATE,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def fig_to_png(fig, dpi=DPI):
    """Convert matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════
# BAR CHART — Pivotal Studies (ORR / Response Rates)
# ═══════════════════════════════════════════════════════
def render_bar_chart(content):
    """Render grouped bar chart for pivotal study results."""
    setup_style()

    # Extract trial data from content
    trials = content.get('trials', [])
    if not trials:
        return None

    fig, ax = plt.subplots(figsize=(11, 3.5))

    # Collect metrics per trial
    labels = []
    drug_vals = []
    ctrl_vals = []
    for tr in trials[:6]:  # max 6 trials
        name = tr.get('name', tr.get('trial', ''))
        labels.append(name[:25])
        # Try ORR first, then mPFS, then generic
        drug_v = _parse_num(tr.get('orr_drug', tr.get('orr', tr.get('efficacy', {}).get('orr_drug', ''))))
        ctrl_v = _parse_num(tr.get('orr_control', tr.get('efficacy', {}).get('orr_control', '')))
        drug_vals.append(drug_v)
        ctrl_vals.append(ctrl_v)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, drug_vals, width, color=PURPLE, label='Drug', zorder=3)
    if any(v > 0 for v in ctrl_vals):
        bars2 = ax.bar(x + width/2, ctrl_vals, width, color=MUTED, label='Control', zorder=3)

    # Value labels on bars
    for bar, val in zip(bars1, drug_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color=PURPLE)

    ax.set_ylabel('Response Rate (%)', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
    ax.set_ylim(0, max(drug_vals + ctrl_vals + [50]) * 1.15)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    return fig_to_png(fig)


# ═══════════════════════════════════════════════════════
# FOREST PLOT — Subgroup Analysis / HR Comparison
# ═══════════════════════════════════════════════════════
def render_forest_plot(content):
    """Render forest plot with diamonds and CI lines."""
    setup_style()

    # Extract subgroups or comparisons
    items = content.get('subgroups', content.get('analyses', content.get('rows', [])))
    if not items:
        return None

    fig, ax = plt.subplots(figsize=(11, max(3, len(items) * 0.5 + 1)))

    y_positions = []
    for i, item in enumerate(reversed(items[:15])):  # max 15 rows
        y = i
        y_positions.append(y)

        label = item.get('subgroup', item.get('name', item.get('label', f'Subgroup {i+1}')))
        hr = _parse_num(item.get('hr', item.get('hazard_ratio', 1.0)))
        ci_low = _parse_num(item.get('ci_low', item.get('ci_lower', hr * 0.7)))
        ci_high = _parse_num(item.get('ci_high', item.get('ci_upper', hr * 1.3)))
        n = item.get('n', '')

        # CI line
        ax.plot([ci_low, ci_high], [y, y], color=DARK, linewidth=1.5, zorder=2)
        # Diamond
        ax.scatter(hr, y, s=80, color=PURPLE, marker='D', zorder=3, edgecolors='white', linewidth=0.5)

        # Label on left
        label_text = f'{label[:35]}'
        if n:
            label_text += f'  (n={n})'
        ax.text(-0.02, y, label_text, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=8, color=DARK)

        # HR value on right
        ax.text(1.02, y, f'{hr:.2f} [{ci_low:.2f}–{ci_high:.2f}]',
                transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=7, color=SLATE, family='monospace')

    # Reference line at HR=1
    ax.axvline(x=1.0, color=ROSE, linestyle='--', linewidth=1, alpha=0.7, zorder=1)

    # Labels
    ax.set_xlabel('Hazard Ratio', fontsize=10, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(0, max(2.5, max([_parse_num(it.get('ci_high', 2)) for it in items]) * 1.1))

    # Favor labels
    ax.text(0.3, -0.08, '← Favors Treatment', transform=ax.transAxes,
            fontsize=8, color=TEAL, ha='center')
    ax.text(0.7, -0.08, 'Favors Control →', transform=ax.transAxes,
            fontsize=8, color=ROSE, ha='center')

    fig.subplots_adjust(left=0.35, right=0.78)
    return fig_to_png(fig)


# ═══════════════════════════════════════════════════════
# SWIMMER PLOT — Duration of Response per Patient
# ═══════════════════════════════════════════════════════
def render_swimmer_plot(content):
    """Render swimmer plot showing treatment duration per patient."""
    setup_style()

    patients = content.get('patients', content.get('swimmers', content.get('rows', [])))
    if not patients:
        return None

    fig, ax = plt.subplots(figsize=(11, max(3, len(patients) * 0.35 + 1)))

    for i, pt in enumerate(patients[:20]):  # max 20 patients
        start = _parse_num(pt.get('start', 0))
        duration = _parse_num(pt.get('duration', pt.get('months', pt.get('time', 6))))
        response = pt.get('response', pt.get('status', 'PR'))
        ongoing = pt.get('ongoing', False)

        color = PURPLE if response in ('CR', 'Complete') else TEAL if response in ('PR', 'Partial') else MUTED
        ax.barh(i, duration, left=start, height=0.6, color=color, edgecolor='white', linewidth=0.5, zorder=2)

        # Arrow for ongoing
        if ongoing:
            ax.annotate('', xy=(start + duration + 0.3, i), xytext=(start + duration, i),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        # Label
        label = pt.get('id', pt.get('patient', f'Pt {i+1}'))
        ax.text(-0.5, i, str(label)[:12], ha='right', va='center', fontsize=7, color=SLATE)

    ax.set_xlabel('Time (months)', fontsize=10, fontweight='bold')
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3, zorder=0)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PURPLE, label='CR'),
        Patch(facecolor=TEAL, label='PR'),
        Patch(facecolor=MUTED, label='SD/PD'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right', framealpha=0.9)

    fig.subplots_adjust(left=0.15)
    return fig_to_png(fig)


# ═══════════════════════════════════════════════════════
# WATERFALL PLOT — Tumor Size Change per Patient
# ═══════════════════════════════════════════════════════
def render_waterfall_plot(content):
    """Render waterfall plot showing best % change from baseline."""
    setup_style()

    patients = content.get('patients', content.get('changes', content.get('rows', [])))
    if not patients:
        return None

    # Extract values and sort descending
    data = []
    for pt in patients[:40]:  # max 40
        val = _parse_num(pt.get('change', pt.get('value', pt.get('pct_change', 0))))
        resp = pt.get('response', pt.get('status', ''))
        data.append((val, resp))
    data.sort(key=lambda x: x[0], reverse=True)

    fig, ax = plt.subplots(figsize=(11, 3.5))

    values = [d[0] for d in data]
    colors = []
    for val, resp in data:
        if resp in ('CR', 'Complete'):
            colors.append(PURPLE)
        elif resp in ('PR', 'Partial') or val <= -30:
            colors.append(TEAL)
        elif resp in ('SD', 'Stable') or (-30 < val <= 20):
            colors.append(GOLD)
        else:
            colors.append(ROSE)

    ax.bar(range(len(values)), values, color=colors, edgecolor='white', linewidth=0.3, zorder=2)

    # Reference lines
    ax.axhline(y=-30, color=TEAL, linestyle='--', linewidth=1, alpha=0.5, label='PR threshold (-30%)')
    ax.axhline(y=20, color=ROSE, linestyle='--', linewidth=1, alpha=0.5, label='PD threshold (+20%)')
    ax.axhline(y=0, color=DARK, linewidth=0.5)

    ax.set_ylabel('Best % Change from Baseline', fontsize=10, fontweight='bold')
    ax.set_xlabel('Patients', fontsize=9)
    ax.set_xticks([])
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.2, zorder=0)

    return fig_to_png(fig)


# ═══════════════════════════════════════════════════════
# GANTT CHART — Tactical Plan Timeline
# ═══════════════════════════════════════════════════════
def render_gantt_chart(content):
    """Render Gantt chart for tactical plan / timeline."""
    setup_style()

    rows = content.get('workstreams', content.get('activities', content.get('rows', [])))
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(11, max(2.5, len(rows) * 0.45 + 0.5)))

    # Parse timeline data
    for i, row in enumerate(reversed(rows[:10])):  # max 10 workstreams
        name = row.get('name', row.get('workstream', row.get('activity', f'WS {i+1}')))
        start = _parse_num(row.get('start_month', row.get('start', i * 2)))
        end = _parse_num(row.get('end_month', row.get('end', start + 6)))
        if end <= start:
            end = start + 3
        color = COLORS[i % len(COLORS)]

        ax.barh(i, end - start, left=start, height=0.5, color=color,
                edgecolor='white', linewidth=0.5, alpha=0.85, zorder=2)
        ax.text(start + (end - start) / 2, i, name[:30],
                ha='center', va='center', fontsize=7, fontweight='bold',
                color='white' if _is_dark(color) else DARK, zorder=3)

    # Month labels on x-axis
    max_month = max([_parse_num(r.get('end_month', r.get('end', 12))) for r in rows] + [12])
    ax.set_xlim(0, max_month + 1)
    month_labels = [f'M{m}' for m in range(0, int(max_month) + 2, 3)]
    ax.set_xticks(range(0, int(max_month) + 2, 3))
    ax.set_xticklabels(month_labels, fontsize=8)
    ax.set_xlabel('Timeline', fontsize=10, fontweight='bold')
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3, zorder=0)

    fig.subplots_adjust(left=0.05)
    return fig_to_png(fig)


# ═══════════════════════════════════════════════════════
# DISPATCHER — Route layout → chart function
# ═══════════════════════════════════════════════════════
def render_chart(layout, content):
    """
    Dispatch chart rendering based on layout ID.
    Returns PNG bytes or None.
    """
    renderers = {
        'PIVOTAL_STUDIES': render_bar_chart,
        'FOREST_PLOT': render_forest_plot,
        'SWIMMER_PLOT': render_swimmer_plot,
        'WATERFALL_PLOT': render_waterfall_plot,
        'TACTICAL_PLAN_4': render_gantt_chart,
        'TACTICAL_PLAN_6': render_gantt_chart,
        'TACTICAL_PLAN_8': render_gantt_chart,
        # KM_CURVE uses the existing km_curve module
    }

    renderer = renderers.get(layout)
    if not renderer:
        logger.info(f"No chart renderer for layout {layout}")
        return None

    try:
        return renderer(content)
    except Exception as e:
        logger.error(f"Chart render error for {layout}: {e}")
        return None


# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════
def _parse_num(val):
    """Safely parse a numeric value from string/int/float."""
    if val is None or val == '':
        return 0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        cleaned = str(val).replace('%', '').replace(',', '.').strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return 0


def _is_dark(hex_color):
    """Check if a hex color is dark (for text contrast)."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (r * 0.299 + g * 0.587 + b * 0.114) < 150
