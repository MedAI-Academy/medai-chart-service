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
    """Find (xl, yt, xr, yb) of the plot area by locating the longest
    vertical and horizontal axis lines."""
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(H // 4, 40)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 4, 60), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Left axis: leftmost dense vertical line column in left half
    left_half = v_lines[:, :W // 2]
    col_density = left_half.sum(axis=0)
    if col_density.max() > 0:
        xl = int(np.argmax(col_density))
    else:
        xl = int(W * 0.08)

    # Bottom axis: bottommost dense horizontal row in bottom half
    bot_half = h_lines[H // 2:, :]
    row_density = bot_half.sum(axis=1)
    if row_density.max() > 0:
        yb = int(np.argmax(row_density) + H // 2)
    else:
        yb = int(H * 0.88)

    # Top of y-axis line
    v_col_rows = np.where(v_lines[:, xl] > 0)[0]
    if len(v_col_rows):
        yt = int(v_col_rows[0])
    else:
        yt = int(H * 0.05)

    # Right: extend to right of any horizontal line run
    h_row_cols = np.where(h_lines[yb, :] > 0)[0] if yb < H else np.array([])
    if len(h_row_cols):
        xr = int(h_row_cols[-1])
    else:
        xr = int(W * 0.98)

    # Sanity clamps
    xl = max(0, min(xl, int(W * 0.3)))
    yt = max(0, min(yt, int(H * 0.3)))
    yb = max(int(H * 0.5), min(yb, H - 1))
    xr = max(int(W * 0.5), min(xr, W - 1))
    if xr <= xl + 50 or yb <= yt + 50:
        return (int(W * 0.08), int(H * 0.05), int(W * 0.98), int(H * 0.88))
    return (xl, yt, xr, yb)


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

    # Close small gaps in the curve
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
    plot_mask = cv2.morphologyEx(plot_mask, cv2.MORPH_CLOSE, k_close)

    # Drop speckle / narrow components
    num, lab, stats, _ = cv2.connectedComponentsWithStats(plot_mask)
    clean = np.zeros_like(plot_mask)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 12:
            continue
        if min_cc_width and w < min_cc_width:
            continue
        clean[lab == i] = 255
    plot_mask = clean

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

    # Censoring ticks
    censored = _detect_censoring(plot_mask, plot_bounds, x_range, y_range)

    return coords, censored


def _detect_censoring(plot_mask, plot_bounds, x_range, y_range) -> list[float]:
    num, _, stats, cent = cv2.connectedComponentsWithStats(plot_mask)
    xmin, xmax = x_range
    times = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if 2 <= h <= 18 and w <= 5 and area <= 40:
            cx = cent[i][0]
            t, _ = _px_to_data(cx, cent[i][1], plot_bounds, x_range, y_range)
            if xmin <= t <= xmax:
                times.append(round(float(t), 2))
    return sorted(set(times))


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
        for tc in arm.get("censored_times", []):
            s = _survival_at(arm["coordinates"], tc)
            if s is not None:
                ax.plot(
                    [tc, tc], [s - 1.3, s + 1.3],
                    color=arm.get("color_hex", "#2166AC"), linewidth=1,
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

    # Plot bounds
    if plot_bounds is None:
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
