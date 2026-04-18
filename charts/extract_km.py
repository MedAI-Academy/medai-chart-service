"""
Kaplan-Meier Curve Extractor — OpenCV pixel-level extraction.

Based on the Gemini-proven approach (VIALE-A reconstruction):
  1. HSV color separation per arm
  2. Topmost pixel per column = survival
  3. Pixel -> data coordinate transform
  4. Monotonic-decrease enforcement
  5. Censoring tick detection via morphological ops

Entry point: extract_km(image_bytes, study_name=None, arm_colors=None)
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Reference data (used when study_name matches a known study)
# ---------------------------------------------------------------------------

STUDY_REFERENCE = {
    "VIALE-A": {
        "endpoint": "Overall Survival",
        "x_range": (0, 33),
        "y_range": (0.0, 1.0),
        "plot_bounds": (521, 243, 1552, 873),
        "nar_times": [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        "arms": {
            "Azacitidine plus venetoclax": {
                "color_hint": "blue", "color_hex": "#2c6fb0",
                "n": 286, "published_median": 14.7,
                "nar": [286, 219, 198, 168, 143, 117, 101, 54, 23, 5, 3, 0],
            },
            "Azacitidine plus placebo": {
                "color_hint": "green", "color_hex": "#3bb273",
                "n": 145, "published_median": 9.6,
                "nar": [145, 109, 92, 74, 59, 38, 30, 14, 5, 1, 0, 0],
            },
        },
        "hr_text": "HR 0.66 (95% CI 0.52\u20130.85); P<0.001",
    },
    "AQUILA": {
        "endpoint": "Progression to MM (IRC)",
        "x_range": (0, 72),
        "y_range": (0.0, 1.0),
        "plot_bounds": (189, 170, 1058, 709),
        "nar_times": list(range(0, 73, 6)),
        "arms": {
            "Daratumumab SC": {
                "color_hint": "red", "color_hex": "#d1342f",
                "n": 194, "published_median": None,
                "nar": [194, 181, 166, 149, 142, 138, 129, 118, 106, 96, 90, 41, 6],
            },
            "Active Monitoring": {
                "color_hint": "gray", "color_hex": "#4a4a4a",
                "n": 196, "published_median": 41.5,
                "nar": [196, 175, 142, 120, 100, 87, 78, 67, 60, 51, 49, 19, 2],
            },
        },
        "hr_text": "HR 0.49 (95% CI 0.36\u20130.67); P<0.001",
    },
    "CLEAR": {
        "endpoint": "Progression-Free Survival",
        "x_range": (0, 40),
        "y_range": (0.0, 1.0),
        "plot_bounds": (415, 149, 1467, 622),
        "nar_times": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
        "arms": {
            "Lenvatinib + Pembrolizumab": {
                "color_hint": "red", "color_hex": "#d94f3a",
                "n": 355, "published_median": 23.9,
                "nar": [355, 300, 259, 218, 172, 138, 101, 56, 14, 1, 0],
            },
            "Lenvatinib + Everolimus": {
                "color_hint": "blue", "color_hex": "#3674b5",
                "n": 357, "published_median": 14.7,
                "nar": [357, 280, 193, 145, 124, 105, 79, 53, 17, 3, 1],
            },
            "Sunitinib": {
                "color_hint": "black", "color_hex": "#333333",
                "n": 357, "published_median": 9.2,
                "nar": [357, 228, 155, 117, 84, 69, 46, 32, 16, 9, 1],
            },
        },
        "hr_text": "HR 0.39 (L+P vs Sun, 95% CI 0.32\u20130.49); P<0.001",
    },
}


# Color hint -> HSV range. OpenCV hue is 0..180.
COLOR_HSV = {
    "blue":       ((100, 40, 40),  (130, 255, 255)),
    "dunkelblau": ((100, 40, 30),  (130, 255, 220)),
    "navy":       ((105, 60, 30),  (128, 255, 200)),
    "lightblue":  ((90, 30, 150),  (110, 180, 255)),
    "green":      ((35, 40, 40),   (85, 255, 255)),
    "waldgruen":  ((40, 40, 40),   (85, 255, 220)),
    "red":        ((0, 80, 60),    (10, 255, 255)),
    "red2":       ((170, 80, 60),  (180, 255, 255)),
    "orange":     ((10, 80, 80),   (25, 255, 255)),
    "yellow":     ((25, 80, 80),   (35, 255, 255)),
    "purple":     ((130, 40, 40),  (160, 255, 255)),
    "magenta":    ((140, 60, 60),  (170, 255, 255)),
    "black":      ((0, 0, 0),      (180, 70, 80)),
    "gray":       ((0, 0, 60),     (180, 40, 180)),
    "grau":       ((0, 0, 60),     (180, 40, 180)),
}


HINT_TO_HEX = {
    "blue": "#2166AC", "dunkelblau": "#2c6fb0", "navy": "#1b3a6b",
    "lightblue": "#6ba3d1",
    "green": "#4DAC26", "waldgruen": "#3bb273",
    "red": "#D6604D", "red2": "#D6604D",
    "orange": "#E08214", "yellow": "#F4A83B",
    "purple": "#9970AB", "magenta": "#C74B8B",
    "black": "#222222", "gray": "#666666", "grau": "#666666",
}


# ---------------------------------------------------------------------------
# Plot-bounds auto-detection
# ---------------------------------------------------------------------------

def detect_plot_bounds(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Find (xl, yt, xr, yb) of the data area of a KM plot.

    Two-stage heuristic:
      1. Find the long continuous axis lines (>=55 % of H/W). This rejects
         short strokes belonging to text or ticks.
      2. Refine the bounds by scanning for perpendicular tick marks that
         are anchored on the detected axes. A *plot frame* is a box around
         the data area but has no ticks on its inner side; the *real axis*
         is the one with ticks. The outermost tick positions also give
         exact (yt, yb) for y=1.0 / y=0.0 and (xl, xr) for x_min / x_max
         — which previously failed on AQUILA where the frame line sat
         above the real 100 % tick.
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Require >=55 % of H/W as a single continuous segment => ignores text.
    v_span = max(int(H * 0.55), 200)
    h_span = max(int(W * 0.55), 200)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_span))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_span, 1))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    border_skip = max(int(W * 0.02), 4)
    left_cap = W // 2
    col_density = v_lines.sum(axis=0)
    col_density[:border_skip] = 0
    col_density[left_cap:] = 0

    xl = int(W * 0.1)
    yt = int(H * 0.05)
    col_tops: dict[int, int] = {}
    col_bottoms: dict[int, int] = {}
    if col_density.max() > 0:
        cutoff = col_density.max() * 0.35
        candidates = np.where(col_density >= cutoff)[0]
        for c in candidates:
            rows = np.where(v_lines[:, c] > 0)[0]
            if len(rows):
                col_tops[int(c)] = int(rows[0])
                col_bottoms[int(c)] = int(rows[-1])
        if col_tops:
            max_top = max(col_tops.values())
            axis_cols = [c for c, t in col_tops.items() if t >= max_top - 15]
            xl = min(axis_cols)
            yt = col_tops[xl]

    row_density = h_lines.sum(axis=1)
    row_density[H - border_skip:] = 0
    yb = int(H * 0.88)
    picked_ticks: list[int] = []
    if row_density.max() > 0:
        cutoff_h = row_density.max() * 0.35
        h_candidates = sorted(np.where(row_density >= cutoff_h)[0].tolist())
        # Group near-adjacent rows into single "lines"
        line_groups: list[list[int]] = []
        for r in h_candidates:
            if line_groups and r - line_groups[-1][-1] <= 15:
                line_groups[-1].append(r)
            else:
                line_groups.append([r])
        # Each line's representative row = bottommost of its rows
        line_rows = [max(g) for g in line_groups]
        # Only lines strictly below yt (plot top) can be yb
        line_rows = [r for r in line_rows if r > yt + 80]
        # Prefer the topmost line with downward tick marks (x-axis signature);
        # NaR table / caption-box borders don't have ticks below them.
        picked = None
        picked_ticks: list[int] = []
        for cand in line_rows:
            ticks = _downward_tick_cols(binary, cand, W, H)
            if ticks:
                picked = cand
                picked_ticks = ticks
                break
        if picked is None and line_rows:
            # Fallback: topmost long horizontal line below yt
            picked = line_rows[0]
        if picked is not None:
            yb = picked

    # If we have downward ticks, derive xl and xr from their leftmost/
    # rightmost positions. This is more reliable than long-line detection
    # because the outer page frame or caption box can fool the vertical
    # line pass (too long to be the actual y-axis spine).
    if picked_ticks:
        xl = picked_ticks[0]
        xr = picked_ticks[-1]
        # And re-pick yt from y-axis ticks: the long-vertical-line
        # pass may have locked onto an outer page frame that extends
        # far past the real plot top, placing yt at the page margin
        # instead of the y=1.0 tick. The topmost leftward tick on the
        # y-axis spine at xl is the true y=1.0 position.
        y_ticks = _leftward_tick_rows(binary, xl, yb, H)
        if y_ticks:
            yt = y_ticks[0]
    else:
        axis_row = h_lines[yb, :] if yb < H else np.array([])
        h_cols = np.where(axis_row > 0)[0]
        xr = int(h_cols[-1]) if len(h_cols) else int(W * 0.98)

    # --- Tick-based refinement ---------------------------------------
    # KM figures sometimes have a thin plot frame ABOVE the real y=1.0
    # tick (NEJM's AQUILA). The detected axis line may be this frame.
    # Looking at the y-axis tick marks (short horizontal strokes to the
    # left of the y-axis spine) gives us the true yt and yb.
    xl, yt, xr, yb = _refine_bounds_with_ticks(binary, xl, yt, xr, yb, H, W)

    # --- Sanity clamps ---
    xl = max(border_skip, min(xl, int(W * 0.35)))
    yt = max(0, min(yt, int(H * 0.30)))
    # yb can be in the UPPER half of the crop for PDF figures that
    # include NaR tables and captions below the plot, so only clamp
    # relative to yt, not to H.
    yb = max(yt + 100, min(yb, H - 1))
    xr = max(int(W * 0.55), min(xr, W - 1))
    if xr <= xl + 50 or yb <= yt + 50:
        return (int(W * 0.10), int(H * 0.05), int(W * 0.98), int(H * 0.88))
    return (xl, yt, xr, yb)


def _refine_bounds_with_ticks(binary, xl, yt, xr, yb, H, W):
    """Refine yt (and only yt) when a plot frame sits above the real y=1.0.

    Conservative: we don't touch yb, xl, xr because the long-axis-line
    heuristic is already reliable for those. The pathological case is
    NEJM-style figures where there's a thin frame border a few tens of
    pixels above the real 100 % y-tick — the previous implementation
    picked this frame as yt, which pushed the whole y-axis calibration
    off by ~20 %.

    Method: look for short horizontal tick strokes immediately LEFT of
    the y-axis spine. The topmost tick cluster gives the true y=1.0.
    We only accept the refinement when:
      * the tick is clearly BELOW the detected yt (>= 10 px)
      * evenly-spaced ticks confirm this is a real tick ladder
    """
    # Look at a narrow band 4-8 px left of the y-axis — tick marks live
    # here; tick *labels* start further left.
    band_end = xl - 1
    band_start = max(0, xl - 8)
    if band_end - band_start < 3:
        return xl, yt, xr, yb
    region = binary[:yb, band_start:band_end]
    if region.size == 0:
        return xl, yt, xr, yb

    row_hits = region.sum(axis=1)
    # A tick row needs nearly the whole band filled (most of 4-8 px).
    thresh = max(2, (band_end - band_start) - 1) * 255
    rows_with_tick = np.where(row_hits >= thresh)[0]
    ygrp = _group_close(rows_with_tick, gap=5)
    if len(ygrp) < 4:
        return xl, yt, xr, yb

    # Check roughly-even tick spacing (signature of a real ladder).
    spacings = np.diff(ygrp)
    if len(spacings) < 3:
        return xl, yt, xr, yb
    median_sp = float(np.median(spacings))
    if median_sp < 6:
        return xl, yt, xr, yb
    # At least 60 % of spacings within ±25 % of median.
    ok = np.sum(np.abs(spacings - median_sp) <= median_sp * 0.25)
    if ok / len(spacings) < 0.6:
        return xl, yt, xr, yb

    yt_tick = ygrp[0]
    # Only refine yt if the tick ladder's top starts noticeably below
    # the currently-detected yt — that's the frame-above-ticks case.
    if yt_tick - yt >= 10 and yt_tick < yb - 100:
        yt = yt_tick

    return xl, yt, xr, yb


def _downward_tick_cols(binary, y_row, W, H) -> list[int]:
    """Return the x-column positions of downward tick marks just below
    y_row, or [] if the row does not have a regular tick ladder.

    Used both to CHOOSE yb (only the plot x-axis has downward ticks;
    NaR / caption rules don't) and to DERIVE xl/xr (leftmost and
    rightmost tick = plot x-range endpoints, bypassing unreliable
    long-line detection which can be fooled by an outer page frame).

    Also filters out page-margin / outer-frame columns: those extend
    the full image height, while real tick marks are short (≤25 px).
    """
    band_start = y_row + 2
    band_end = min(H, y_row + 10)
    if band_end - band_start < 3:
        return []
    col_start = 0
    col_end = min(W, int(W * 0.99))
    region = binary[band_start:band_end, col_start:col_end]
    if region.size == 0:
        return []
    col_hits = region.sum(axis=0)
    thresh = max(2, (band_end - band_start) - 1) * 255
    cols_with_tick = np.where(col_hits >= thresh)[0]

    # Drop columns that belong to a tall vertical line (page margin /
    # outer frame). A real tick stroke is ≤25 px tall and ends near the
    # axis; a page margin keeps going for hundreds of pixels.
    real_ticks: list[int] = []
    max_tick_len = 25
    y_check = y_row + 5
    if y_check >= H:
        return []
    for c in cols_with_tick:
        # Count how far the black run extends BELOW y_check.
        y = y_check
        run_down = 0
        while y < H and binary[y, c] > 0 and run_down < 200:
            run_down += 1
            y += 1
        if run_down <= max_tick_len:
            real_ticks.append(int(c))

    xgrp = _group_close(real_ticks, gap=5)
    if len(xgrp) < 4:
        return []
    spacings = np.diff(xgrp)
    if len(spacings) < 3:
        return []
    median_sp = float(np.median(spacings))
    if median_sp < 6:
        return []
    ok = int(np.sum(np.abs(spacings - median_sp) <= median_sp * 0.25))
    if ok / len(spacings) < 0.6:
        return []

    # Trim leading / trailing ticks whose spacing to the next tick is
    # off-ladder (< 70 % of median). These come from:
    #   - y-axis spine (short downward overhang just past yb)
    #   - a stray glyph from the x-axis tick *label* row if it sits
    #     close enough to the band
    # Without trimming, xl gets anchored to the y-axis spine rather
    # than the "0 months" tick — off by one tick spacing, which
    # biases the time calibration and undershoots medians by ~1 mo.
    def _trim_edges(ticks: list[int], med: float) -> list[int]:
        lo_bound = med * 0.70
        hi_bound = med * 1.30
        out = list(ticks)
        while len(out) >= 3:
            sp = out[1] - out[0]
            if sp < lo_bound or sp > hi_bound:
                out.pop(0)
            else:
                break
        while len(out) >= 3:
            sp = out[-1] - out[-2]
            if sp < lo_bound or sp > hi_bound:
                out.pop()
            else:
                break
        return out

    xgrp = _trim_edges(xgrp, median_sp)
    if len(xgrp) < 4:
        return []
    return xgrp


def _leftward_tick_rows(binary, x_col, yb, H) -> list[int]:
    """Return y-row positions of leftward tick marks on the y-axis
    spine at column x_col, or [] if no regular tick ladder is found.

    Used to derive yt (topmost tick = y=1.0) when the long-vertical-
    line detection is fooled by an outer page frame that extends past
    the real y-axis.
    """
    band_end = x_col - 1
    band_start = max(0, x_col - 10)
    if band_end - band_start < 3:
        return []
    region = binary[:yb, band_start:band_end]
    if region.size == 0:
        return []
    row_hits = region.sum(axis=1)
    thresh = max(2, (band_end - band_start) - 1) * 255
    rows_with_tick = np.where(row_hits >= thresh)[0]

    real_ticks: list[int] = []
    max_tick_len = 25
    x_check = x_col - 5
    if x_check < 0:
        return []
    for r in rows_with_tick:
        x = x_check
        run_left = 0
        while x >= 0 and binary[r, x] > 0 and run_left < 200:
            run_left += 1
            x -= 1
        if run_left <= max_tick_len:
            real_ticks.append(int(r))

    ygrp = _group_close(real_ticks, gap=5)
    if len(ygrp) < 4:
        return []
    spacings = np.diff(ygrp)
    if len(spacings) < 3:
        return []
    median_sp = float(np.median(spacings))
    if median_sp < 6:
        return []
    ok = int(np.sum(np.abs(spacings - median_sp) <= median_sp * 0.25))
    if ok / len(spacings) < 0.6:
        return []

    # Trim edge ticks with off-ladder spacing (same rationale as
    # _downward_tick_cols).
    lo, hi = median_sp * 0.70, median_sp * 1.30
    out = list(ygrp)
    while len(out) >= 3:
        sp = out[1] - out[0]
        if sp < lo or sp > hi:
            out.pop(0)
        else:
            break
    while len(out) >= 3:
        sp = out[-1] - out[-2]
        if sp < lo or sp > hi:
            out.pop()
        else:
            break
    return out if len(out) >= 4 else []


def _group_close(arr, gap: int = 6) -> list[int]:
    """Group consecutive integer positions whose spacing <= gap; return
    the mean of each group as an int."""
    if len(arr) == 0:
        return []
    groups = [[int(arr[0])]]
    for v in arr[1:]:
        if int(v) - groups[-1][-1] <= gap:
            groups[-1].append(int(v))
        else:
            groups.append([int(v)])
    return [int(round(sum(g) / len(g))) for g in groups]


# ---------------------------------------------------------------------------
# Pixel-to-data transform
# ---------------------------------------------------------------------------

def _px_to_data(x_px, y_px, plot_bounds, x_range, y_range):
    xl, yt, xr, yb = plot_bounds
    xmin, xmax = x_range
    ymin, ymax = y_range
    pw, ph = xr - xl, yb - yt
    t = xmin + (x_px - xl) / pw * (xmax - xmin)
    s = ymax - (y_px - yt) / ph * (ymax - ymin)
    return t, s


# ---------------------------------------------------------------------------
# Arm extraction
# ---------------------------------------------------------------------------

def extract_arm_curve(
    img_bgr: np.ndarray,
    plot_bounds: tuple,
    hsv_lower: tuple,
    hsv_upper: tuple,
    x_range: tuple,
    y_range: tuple,
    close_kernel: tuple = (5, 5),
    min_cc_width: int = 0,
    min_cc_height: int = 0,
    survival_band: Optional[tuple] = None,
    tail_anchors: Optional[list] = None,
) -> tuple[list, list]:
    """Return (coords [(t, s_pct)], censored_times [t]) for one arm."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    xl, yt, xr, yb = plot_bounds
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    plot_mask = np.zeros_like(mask)
    plot_mask[yt:yb, xl:xr] = mask[yt:yb, xl:xr]

    # Remove long horizontal/vertical axis runs
    pw, ph = xr - xl, yb - yt
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(pw // 2, 80), 1))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(ph // 2, 80)))
    h_runs = cv2.morphologyEx(plot_mask, cv2.MORPH_OPEN, h_k)
    v_runs = cv2.morphologyEx(plot_mask, cv2.MORPH_OPEN, v_k)
    h_runs = cv2.dilate(h_runs, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    v_runs = cv2.dilate(v_runs, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    plot_mask = cv2.subtract(plot_mask, h_runs)
    plot_mask = cv2.subtract(plot_mask, v_runs)

    # Snapshot *before* the closing step — closing fuses censoring ticks
    # into the main curve body, so CC analysis on the closed mask only
    # sees fringes. We keep this raw mask for censoring detection.
    raw_mask = plot_mask.copy()

    # Close small gaps in the curve (for the actual step-curve extraction)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
    plot_mask = cv2.morphologyEx(plot_mask, cv2.MORPH_CLOSE, k_close)

    # Drop speckle / narrow / short components
    num, lab, stats, _ = cv2.connectedComponentsWithStats(plot_mask)
    clean = np.zeros_like(plot_mask)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 12:
            continue
        if min_cc_width and w < min_cc_width:
            continue
        if min_cc_height and h < min_cc_height:
            continue
        clean[lab == i] = 255
    plot_mask = clean

    # Optional survival-band filter: zero-out rows outside [s_min, s_max] (%).
    # Useful for OS curves that stay near the top — rejects HR/annotation
    # text and axis labels in the lower portion of the plot.
    if survival_band is not None:
        s_min, s_max = survival_band
        y_scale_band = 100.0 if y_range[1] <= 1.001 else 1.0
        ph_px = yb - yt
        y_top = yt + int((1 - s_max / (y_range[1] * y_scale_band)) * ph_px)
        y_bot = yt + int((1 - s_min / (y_range[1] * y_scale_band)) * ph_px)
        y_top = max(y_top, yt)
        y_bot = min(y_bot, yb)
        band_mask = np.zeros_like(plot_mask)
        band_mask[y_top:y_bot, :] = 255
        plot_mask = cv2.bitwise_and(plot_mask, band_mask)

    y_scale = 100.0 if y_range[1] <= 1.001 else 1.0
    coords: list[tuple] = []
    for x_px in range(xl, xr):
        col = plot_mask[yt:yb, x_px]
        active = np.where(col > 0)[0]
        if len(active) == 0:
            continue
        y_abs = yt + int(active[0])
        t, s = _px_to_data(x_px, y_abs, plot_bounds, x_range, y_range)
        coords.append((round(float(t), 3), round(float(s) * y_scale, 3)))

    # Monotonic decreasing
    if coords:
        mono = [coords[0]]
        for t, s in coords[1:]:
            if s <= mono[-1][1]:
                mono.append((t, s))
        coords = mono

    y_max_pct = y_range[1] * (100.0 if y_range[1] <= 1.001 else 1.0)
    if coords:
        coords[0] = (0.0, y_max_pct)
    else:
        coords = [(0.0, y_max_pct)]

    # Smooth spike artifacts: split unnaturally large drops (mask loss)
    # into smaller intermediate steps.
    coords = smooth_spikes(coords, factor=3.0)

    # Optional tail anchors to complete a plateau or final drop that
    # pixel extraction missed (e.g. Ven plateau, Plac -> 0%).
    if tail_anchors:
        last_t, last_s = coords[-1]
        for t, s in tail_anchors:
            if t > last_t:
                if s > last_s:
                    s = last_s
                coords.append((float(t), float(s)))
                last_t, last_s = t, s

    # Censoring detection runs on the *raw* (pre-close) mask
    censored = _detect_censoring(raw_mask, plot_bounds, x_range, y_range)

    return coords, censored


def smooth_spikes(coords: list, factor: float = 3.0,
                  min_threshold_pct: float = 2.0) -> list:
    """Split oversized drops into micro-steps to smooth mask-loss artifacts.

    A drop s[i-1] -> s[i] greater than ``factor`` * median(drops) and at
    least ``min_threshold_pct`` % is spread linearly across the t-interval
    as several intermediate points. Real KM step drops are left alone
    because they are near the median drop size.
    """
    if len(coords) < 3:
        return coords
    drops = [coords[i-1][1] - coords[i][1]
             for i in range(1, len(coords))
             if coords[i-1][1] > coords[i][1]]
    if not drops:
        return coords
    drops_sorted = sorted(drops)
    median_drop = drops_sorted[len(drops_sorted) // 2]
    if median_drop <= 0:
        median_drop = 0.3
    threshold = max(median_drop * factor, min_threshold_pct)

    out = [coords[0]]
    for i in range(1, len(coords)):
        t_prev, s_prev = out[-1]
        t_curr, s_curr = coords[i]
        drop = s_prev - s_curr
        if drop > threshold:
            dt_total = max(t_curr - t_prev, 0.0)
            if dt_total < 0.1:
                dt_total = min(1.2, max(0.3, drop / 25.0))
            n_steps = min(60, max(2, int(round(drop / median_drop))))
            for k in range(1, n_steps):
                frac = k / n_steps
                t_k = t_prev + frac * dt_total
                s_k = s_prev - frac * drop
                out.append((round(float(t_k), 4), round(float(s_k), 3)))
        out.append((float(t_curr), float(s_curr)))
    return out


def _detect_censoring(mask, plot_bounds, x_range, y_range) -> list[float]:
    """Find censoring tick marks via local-baseline-deviation on the raw mask.

    For each column we record the topmost mask pixel (curve top). A
    censoring tick is a column where the mask spikes upward by >=2 px
    above the local median baseline. This approach is agnostic to whether
    the tick's mask is continuous with the curve body — which matters for
    KM plots where ticks are drawn in the same color as the line.
    """
    xl, yt, xr, yb = plot_bounds
    xmin, xmax = x_range
    pw = xr - xl

    # topmost y-pixel per column
    top_y = np.full(pw, -1, dtype=np.int32)
    for x_off in range(pw):
        col = mask[yt:yb, xl + x_off]
        active = np.where(col > 0)[0]
        if len(active):
            top_y[x_off] = int(active[0])

    # Local median baseline (small rolling window so real curve drops
    # are preserved in the baseline).
    win = max(11, pw // 60)
    if win % 2 == 0:
        win += 1
    half = win // 2
    baseline = np.full(pw, -1, dtype=np.int32)
    for i in range(pw):
        lo, hi = max(0, i - half), min(pw, i + half + 1)
        seg = top_y[lo:hi]
        seg = seg[seg >= 0]
        if len(seg) >= 3:
            baseline[i] = int(np.median(seg))

    tick_cols = np.where(
        (top_y >= 0) & (baseline >= 0) & (baseline - top_y >= 2)
    )[0]

    times: list[float] = []
    if len(tick_cols):
        groups = [[tick_cols[0]]]
        for c in tick_cols[1:]:
            if c - groups[-1][-1] <= 2:
                groups[-1].append(c)
            else:
                groups.append([c])
        for g in groups:
            if len(g) > 6:  # too wide — probably a real curve drop
                continue
            cx = xl + int(np.mean(g))
            t, _ = _px_to_data(cx, yt, plot_bounds, x_range, y_range)
            if xmin <= t <= xmax:
                times.append(round(float(t), 2))

    # Dedup near-coincident ticks (< 0.15 x-units apart).
    times.sort()
    dedup: list[float] = []
    last = -1e9
    for t in times:
        if t - last > 0.15:
            dedup.append(t)
            last = t
    return dedup


# ---------------------------------------------------------------------------
# Median estimation
# ---------------------------------------------------------------------------

def estimate_median(points: list) -> Optional[float]:
    prev_t, prev_s = 0.0, 100.0
    for t, s in points:
        if s <= 50.0:
            if prev_s == s:
                return float(t)
            frac = (prev_s - 50.0) / (prev_s - s) if prev_s != s else 0
            return float(prev_t + frac * (t - prev_t))
        prev_t, prev_s = t, s
    return None


# ---------------------------------------------------------------------------
# Reconstruction render (medaccur-style)
# ---------------------------------------------------------------------------

def render_reconstruction_png(
    arms: list,
    x_range: tuple,
    y_range: tuple,
    title: str,
    hr_text: str = "",
    nar_times: Optional[list] = None,
) -> bytes:
    y_max_pct = y_range[1] * (100.0 if y_range[1] <= 1.001 else 1.0)

    fig = plt.figure(figsize=(10, 7), dpi=140)
    if nar_times:
        gs = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.08)
        ax = fig.add_subplot(gs[0])
    else:
        ax = fig.add_subplot(1, 1, 1)

    for arm in arms:
        pts = np.array(arm["coordinates"])
        if pts.size == 0:
            continue
        ax.step(
            pts[:, 0], pts[:, 1], where="post",
            color=arm.get("color_hex", "#2166AC"), linewidth=2.0,
            label=f'{arm["name"]} (n={arm.get("n", "?")})' if arm.get("n") else arm["name"],
        )
        cens_t = arm.get("censored_times") or []
        cens_s = [_survival_at(arm["coordinates"], tc) for tc in cens_t]
        cens_t = [t for t, s in zip(cens_t, cens_s) if s is not None]
        cens_s = [s for s in cens_s if s is not None]
        if cens_t:
            ax.plot(
                cens_t, cens_s, linestyle="None", marker="|",
                color=arm.get("color_hex", "#2166AC"),
                markersize=6, markeredgewidth=1.2,
            )

    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(0, y_max_pct)
    ax.set_ylabel("Survival (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    if hr_text:
        ax.text(
            0.99, 0.98, hr_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888"),
        )

    if nar_times:
        ax.set_xticks(nar_times)
        ax.tick_params(axis="x", labelbottom=False)
        ax_nar = fig.add_subplot(gs[1], sharex=ax)
        _draw_nar_table(ax_nar, arms, nar_times, x_range)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _survival_at(points, t_target):
    prev_s = points[0][1] if points else None
    for t, s in points:
        if t >= t_target:
            return prev_s
        prev_s = s
    return prev_s


def _draw_nar_table(ax_nar, arms, nar_times, x_range):
    ax_nar.set_xlim(x_range[0], x_range[1])
    ax_nar.set_ylim(0, 1)
    ax_nar.set_xticks(nar_times)
    ax_nar.tick_params(axis="y", left=False, labelleft=False)
    ax_nar.tick_params(axis="x", top=False, bottom=False, labelbottom=False)
    for spine in ax_nar.spines.values():
        spine.set_visible(False)

    x_left = x_range[0] - 0.02 * (x_range[1] - x_range[0])
    for t in nar_times:
        ax_nar.text(t, 1.05, str(t), ha="center", va="bottom", fontsize=9)
    ax_nar.text(x_left, 1.05, "Months", ha="right", va="bottom", fontsize=9, fontweight="bold")
    ax_nar.text(x_left, 0.78, "No. at Risk", ha="right", va="center", fontsize=9, fontweight="bold")

    row_h = 0.60 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        y = 0.60 - (i + 0.5) * row_h
        ax_nar.text(
            x_left, y, arm.get("short_label") or arm["name"],
            ha="right", va="center", fontsize=8,
            color=arm.get("color_hex", "#333"), fontweight="bold",
        )
        for t, v in zip(nar_times, arm.get("nar") or []):
            ax_nar.text(t, y, str(v), ha="center", va="center", fontsize=8, color="#333")


# ---------------------------------------------------------------------------
# Color resolution
# ---------------------------------------------------------------------------

def _resolve_hsv(color_hint: Optional[str]) -> tuple:
    if not color_hint:
        return COLOR_HSV["blue"]
    key = color_hint.lower().strip()
    if key in COLOR_HSV:
        return COLOR_HSV[key]
    for k in COLOR_HSV:
        if k in key:
            return COLOR_HSV[k]
    return COLOR_HSV["blue"]


def _auto_detect_arm_colors(img_bgr: np.ndarray, plot_bounds: tuple, n_expected: int = 2) -> list:
    """If no colors given, pick the top-N most frequent saturated hues inside
    the plot area. Returns list of {'name', 'color_hint'}."""
    xl, yt, xr, yb = plot_bounds
    roi = img_bgr[yt:yb, xl:xr]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    saturated = (s > 60) & (v > 40) & (v < 230)

    hints_ranked: list = []
    for hint, (lower, upper) in COLOR_HSV.items():
        if hint in {"lightblue", "yellow", "red2", "dunkelblau", "waldgruen", "grau"}:
            continue
        lo = np.array(lower)
        up = np.array(upper)
        mask = (
            (h >= lo[0]) & (h <= up[0]) &
            (s >= lo[1]) & (s <= up[1]) &
            (v >= lo[2]) & (v <= up[2])
        ) & saturated
        hints_ranked.append((hint, int(mask.sum())))

    hints_ranked.sort(key=lambda x: x[1], reverse=True)
    winners = [h for h, c in hints_ranked[:n_expected] if c > 200]
    if not winners:
        winners = ["blue", "green"][:n_expected]
    return [{"name": f"Arm {i+1}", "color_hint": w} for i, w in enumerate(winners)]


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _decode_image(image_base64: str) -> np.ndarray:
    # Strip data URL prefix if present
    if image_base64.startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    raw = base64.b64decode(image_base64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from base64")
    return img


def extract_km(
    image_base64: str,
    study_name: Optional[str] = None,
    arm_colors: Optional[list] = None,
    x_range: Optional[tuple] = None,
    y_range: Optional[tuple] = None,
    plot_bounds: Optional[tuple] = None,
) -> dict:
    """Extract KM curves from a PNG/JPG image.

    Returns a dict shaped per the /extract-km response spec.
    """
    img = _decode_image(image_base64)

    # Resolve study metadata if known
    ref = None
    if study_name:
        key = study_name.strip().upper().replace(" ", "_").split("_")[0]
        for study_key, data in STUDY_REFERENCE.items():
            if study_key.upper().startswith(key):
                ref = data
                break

    # Axis ranges
    if ref:
        x_range = x_range or ref["x_range"]
        y_range = y_range or ref["y_range"]
    else:
        x_range = x_range or (0, 36)
        y_range = y_range or (0.0, 1.0)

    # Plot bounds: caller > reference (for canonical crops) > auto-detect
    if plot_bounds is None:
        if ref and "plot_bounds" in ref:
            plot_bounds = ref["plot_bounds"]
        else:
            plot_bounds = detect_plot_bounds(img)

    # Arms: caller-supplied > reference > auto-detect
    resolved_arms: list[dict] = []
    if arm_colors:
        for a in arm_colors:
            name = a.get("name") or "Arm"
            hint = a.get("color_hint")
            hsv_l, hsv_u = _resolve_hsv(hint)
            enrich = {"n": None, "nar": None, "published_median": None, "color_hex": None}
            if ref:
                for ref_name, ref_data in ref["arms"].items():
                    if (
                        name.lower() == ref_name.lower()
                        or name.lower() in ref_name.lower()
                        or ref_name.lower() in name.lower()
                    ):
                        enrich = {
                            "n": ref_data["n"],
                            "nar": ref_data["nar"],
                            "published_median": ref_data["published_median"],
                            "color_hex": ref_data["color_hex"],
                        }
                        break
            resolved_arms.append({
                "name": name, "color_hint": hint,
                "hsv_lower": hsv_l, "hsv_upper": hsv_u,
                "color_hex": enrich["color_hex"] or HINT_TO_HEX.get((hint or "").lower(), "#2166AC"),
                "n": enrich["n"], "nar": enrich["nar"],
                "published_median": enrich["published_median"],
            })
    elif ref:
        for name, data in ref["arms"].items():
            hsv_l, hsv_u = _resolve_hsv(data["color_hint"])
            resolved_arms.append({
                "name": name, "color_hint": data["color_hint"],
                "hsv_lower": hsv_l, "hsv_upper": hsv_u,
                "color_hex": data["color_hex"],
                "n": data["n"], "nar": data["nar"],
                "published_median": data["published_median"],
            })
    else:
        guessed = _auto_detect_arm_colors(img, plot_bounds, n_expected=2)
        for g in guessed:
            hsv_l, hsv_u = _resolve_hsv(g["color_hint"])
            resolved_arms.append({
                "name": g["name"], "color_hint": g["color_hint"],
                "hsv_lower": hsv_l, "hsv_upper": hsv_u,
                "color_hex": HINT_TO_HEX.get(g["color_hint"], "#2166AC"),
                "n": None, "nar": None, "published_median": None,
            })

    # Extract each arm
    extracted: list[dict] = []
    for arm in resolved_arms:
        coords, censored = extract_arm_curve(
            img, plot_bounds,
            arm["hsv_lower"], arm["hsv_upper"],
            x_range, y_range,
        )
        median = estimate_median(coords)
        extracted.append({
            "name": arm["name"],
            "coordinates": [[float(t), float(s)] for t, s in coords],
            "censored_times": censored,
            "median": round(median, 2) if median is not None else None,
            "n": arm.get("n"),
            "nar": arm.get("nar") or [],
            "color_hex": arm["color_hex"],
        })

    # Confidence tier
    tier = 3
    if ref:
        tier = 2
        all_pass = True
        any_check = False
        for arm_meta, res in zip(resolved_arms, extracted):
            pub = arm_meta.get("published_median")
            if pub is not None and res["median"] is not None:
                any_check = True
                if abs(res["median"] - pub) > 1.5:
                    all_pass = False
        if any_check and all_pass:
            tier = 1

    # Validation dict (short-key per response spec example)
    validation: dict = {}
    for res in extracted:
        nm_lower = res["name"].lower()
        if "ven" in nm_lower or "venetoclax" in nm_lower:
            validation["ven_median"] = res["median"]
        elif "plac" in nm_lower or "placebo" in nm_lower:
            validation["plac_median"] = res["median"]
    for i, res in enumerate(extracted):
        validation[f"arm_{i+1}_median"] = res["median"]

    # Reconstruction PNG
    title = (
        f"{study_name} \u2014 {ref['endpoint']}"
        if ref and study_name else (study_name or "Kaplan\u2013Meier Reconstruction")
    )
    hr_text = ref["hr_text"] if ref else ""
    nar_times = ref["nar_times"] if ref else None
    png_bytes = render_reconstruction_png(
        extracted, x_range, y_range, title, hr_text=hr_text, nar_times=nar_times,
    )
    png_b64 = base64.b64encode(png_bytes).decode("ascii")

    return {
        "arms": [
            {
                "name": r["name"],
                "coordinates": r["coordinates"],
                "censored_times": r["censored_times"],
                "median": r["median"],
            }
            for r in extracted
        ],
        "png_base64": png_b64,
        "validation": validation,
        "confidence_tier": tier,
        "plot_bounds": list(plot_bounds),
        "x_range": list(x_range),
        "y_range": list(y_range),
    }
