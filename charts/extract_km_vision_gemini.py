"""
Kaplan-Meier curve data extraction via Gemini Vision (Two-Pass).

Companion to extract_km_from_pdf.py: takes the 400 DPI PNG crop produced by
the keyword-scoring pipeline and asks Gemini to read off the structured KM
data. The output flows into the Guyot IPD reconstruction pipeline so the
slide is built from extracted data, not the original figure crop.

That gives us:
  - Tier 2 confidence (own visualisation reconstructed via Guyot from
    extracted publication data — the same methodology HTA agencies and
    meta-analyses use)
  - "Reconstructed from {paper} Figure {N} via IPD (Guyot 2012)" footer
    as liability marker
  - No copyright exposure (we never embed the paper's own figure)

Two-Pass design:
  Pass 1 — Metadata (title, arms with median/HR, axis ranges, hr-box text).
           This information is plot-level: title at top, arm names in
           legend, HR/CI text inside the plot area, axes labels.
  Pass 2 — Number-at-Risk table (per-arm counts at multiple time points).
           NaR tables are usually below or beside the plot in a tight
           tabular layout with small font — extracting them in isolation
           with a focused prompt yields better accuracy than asking for
           everything at once.

Both passes use the same primary→fallback model routing as the Forest
plot wrapper: gemini-3-flash-preview first, gemini-2.5-flash on
model-unavailable errors. Network/auth/timeout errors do NOT trigger a
fallback (they would just fail again).

Anti-hallucination posture is strict throughout:
  1. Gemini is told to OMIT values whose accuracy cannot be verified
     from the figure
  2. We re-validate every numeric value in code: positive, finite, in
     plausible range
  3. Cross-pass coherence check: if Pass 1 finds 2 arms but Pass 2
     returns NaR for 3 arms (or vice versa), we trust Pass 1's arm count
     and drop the extra Pass 2 row — the plot is the source of truth
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────
# Same primary→fallback strategy as extract_forest_subgroups_gemini.
# `or` (instead of os.environ.get's default arg) makes empty-string env
# vars fall back to the code default — important because Railway's UI
# can create env vars with empty values, which we hit on 2026-04-20.
# Demo-stability patch (2026-04-23):
# Invert primary/fallback. gemini-3-flash-preview was the original primary but
# we observed repeated 504 Deadline errors in production during real-world
# VIALE-A extraction on 2026-04-22 — the preview model's server queue appeared
# overloaded. gemini-2.5-flash is the stable GA model, runs on different
# infrastructure, and handles KM curve extraction identically in our tests.
# Using it as primary gives us predictable <20s per-pass response time; the
# preview model stays wired in as fallback so we can bounce back onto it by
# overriding via env var. To go back to preview-first, set
# GEMINI_KM_MODEL=gemini-3-flash-preview in Railway.
GEMINI_PRIMARY_MODEL  = os.environ.get("GEMINI_KM_MODEL")          or "gemini-2.5-flash"
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_KM_FALLBACK_MODEL") or "gemini-3-flash-preview"

# ── Timeout strategy (3 layers) ────────────────────────────────────
# Layer 1: per-call SDK timeout. We rely on the SDK's request_options
#          timeout, but it is NOT 100% reliable across all SDK versions.
# Layer 2: per-pass watchdog (concurrent.futures.Future.result(timeout=))
#          enforced on our side regardless of what the SDK does. This is
#          our actual ceiling for one (model, prompt) call.
# Layer 3: total-extraction watchdog. If both passes haven't returned
#          after this many seconds, we give up entirely. Protects against
#          edge-cases where the SDK hangs in C-level code that timeout=
#          can't interrupt.
#
# 2026-04-24 update: bumped per-call timeout from 25s → 60s after observing
# repeated 504s on Pass 3 (curve) even at 50-point target. Google's Vision
# API is slow for dense-image extraction; 25s was too aggressive. Total
# watchdog proportionally raised to 150s to accommodate a full retry cycle
# on the primary model (60s × 2 attempts = 120s, plus ~30s headroom).
# The three passes still run in PARALLEL, so real-world wall time is
# max(pass1, pass2, pass3), typically 45-70s when healthy — not 150s.
GEMINI_TIMEOUT_PER_CALL_SECONDS  = 60  # one (model, prompt) attempt (was 25 until 2026-04-24)
GEMINI_RETRIES_PER_PASS          = 1   # retry budget per pass (so up to 2 attempts)
GEMINI_TOTAL_EXTRACTION_SECONDS  = 150 # absolute ceiling, all passes combined (was 75 until 2026-04-24)

# Substrings (lowercased) in the exception message that indicate the
# model itself is the problem (gone, renamed, deprecated, no permission).
# Network/auth/timeout errors are NOT in this list — they would just
# fail again on the fallback model.
_MODEL_UNAVAILABLE_INDICATORS = (
    "404",
    "not found",
    "model not",
    "invalid model",
    "deprecated",
    "permission denied",
    "model unavailable",
    "unsupported model",
    "is not supported",
    "does not exist",
    "unexpected model name",  # Railway empty-env-var bug, 2026-04-20
)

# Transient server-side errors — the request failed but the problem is
# likely temporary (overloaded queue, server-side deadline, upstream hiccup).
# Policy: retry on SAME model first (might recover on next try), and if the
# retry also fails, escalate to the fallback model (the second model may be
# served from different infrastructure with independent load). This is
# DIFFERENT from _MODEL_UNAVAILABLE_INDICATORS, where the primary model is
# structurally broken and retry on the same model would just fail again.
#
# Added 2026-04-22 after observing "504 Deadline expired" on
# gemini-3-flash-preview during real-world VIALE-A extraction. The preview
# model's server queue appeared overloaded; a retry or fallback to
# gemini-2.5-flash (stable production model) is the correct response.
_TRANSIENT_SERVER_INDICATORS = (
    "504",
    "503",
    "500",
    "deadline expired",
    "deadline exceeded",
    "server error",
    "service unavailable",
    "overloaded",
    "try again later",
    "internal error",
    "timeout",
    "timed out",
)

# ─────────────────────────────────────────────────────────────────
# PROMPT 1 — Metadata extraction
# ─────────────────────────────────────────────────────────────────
# Reads everything visible in the plot area: title, arm names, colors,
# medians, hazard ratio box, axis labels and ranges. Does NOT touch the
# Number-at-Risk table — that is Pass 2.
METADATA_PROMPT = """You are a medical data extraction specialist. Extract Kaplan-Meier survival curve metadata from this figure with maximum accuracy.

CRITICAL ANTI-HALLUCINATION RULES (non-negotiable):
1. Extract ONLY values that are clearly visible and readable in the image.
2. If a value is unclear, partially obscured, or you have to guess, set it to null. Do NOT estimate.
3. Numeric values (median, hr, ci_low, ci_high, p_value, x_min, x_max, y_min, y_max) MUST come from text labels printed in the figure (legend, HR-box, axis ticks). NEVER estimate them by visually measuring positions in the plot.
4. If the plot has 2 arms but you can only confidently identify 1, return only the 1 you are sure of. Better to under-report than to invent.
5. For arm colors: report the color as it appears in the legend swatch. Use simple color words (blue, red, green, orange, purple, black, gray) NOT hex codes.

OUTPUT STRUCTURE (return JSON matching this shape exactly):
{
  "title": "Overall Survival",
  "endpoint": "OS",
  "arms": [
    {
      "name": "Venetoclax + Azacitidine",
      "n": 286,
      "color": "blue",
      "median_value": 14.7,
      "median_unit": "months"
    },
    {
      "name": "Placebo + Azacitidine",
      "n": 145,
      "color": "red",
      "median_value": 9.6,
      "median_unit": "months"
    }
  ],
  "hr_value": 0.66,
  "hr_ci_low": 0.52,
  "hr_ci_high": 0.85,
  "p_value": 0.001,
  "p_value_operator": "<",
  "x_label": "Months since Randomization",
  "x_min": 0,
  "x_max": 30,
  "y_label": "Overall Survival (%)",
  "y_min": 0,
  "y_max": 100,
  "extraction_notes": "Optional brief notes about anything unclear"
}

FIELD RULES:
- title: figure title or caption (e.g. "Overall Survival", "Progression-Free Survival"). If unclear, "".
- endpoint: short code — one of "OS", "PFS", "EFS", "DOR", "TTP", "DFS", "RFS", "TFI" or "" if unclear.
- arms: list each arm in the order shown in the legend (top to bottom typically).
  - name: as printed in the legend, including dose/regimen if shown.
  - n: total patients in that arm — read from legend (e.g. "n=286") or HR-box. If not printed, null.
  - color: simple color word as it appears in the legend swatch.
  - median_value: read from legend, plot annotation, or HR-box. Number only. If "Not reached" / "NR" / "NE", set to null.
  - median_unit: usually "months", sometimes "weeks" or "days". Default "months" if unit is implied but not printed.
- hr_value, hr_ci_low, hr_ci_high: from the HR-box text. e.g. "HR 0.66 (95% CI 0.52–0.85)" → 0.66, 0.52, 0.85.
- p_value: numeric only. e.g. "P<0.001" → 0.001, "P=0.0034" → 0.0034.
- p_value_operator: "<", "=", ">", or "" if not present.
- x_min, x_max: read from the X-axis tick range (usually 0 to 30/36/60 months).
- y_min, y_max: read from the Y-axis tick range. Usually 0 to 100 (percent) or 0 to 1.0 (fraction). Report as displayed.

If the image does NOT contain a Kaplan-Meier curve (e.g. it's a forest plot, table, or different chart type), return {"title": "", "arms": [], "extraction_notes": "Not a Kaplan-Meier curve"}.
"""

# ─────────────────────────────────────────────────────────────────
# PROMPT 2 — Number-at-Risk extraction
# ─────────────────────────────────────────────────────────────────
# Focused entirely on the NaR table that typically sits below the plot.
# These tables are dense, small-font, and arm-color-coded — easier to
# extract accurately when not competing with metadata extraction.
NAR_PROMPT = """You are a medical data extraction specialist. Extract the Number-at-Risk (NaR) table from this Kaplan-Meier figure.

The NaR table typically appears BELOW the plot, showing how many patients remain at risk at evenly-spaced time points (e.g. every 3 or 6 months). Each row corresponds to one treatment arm.

CRITICAL ANTI-HALLUCINATION RULES (non-negotiable):
1. Extract ONLY numbers that are clearly visible in the table. Do NOT interpolate.
2. If a column is partially obscured or numbers are blurry, OMIT that time point from `time_points` and leave the corresponding count out of `counts` for ALL arms (keep arrays aligned).
3. If you see arm labels (e.g. "Ven + Aza", "Placebo") next to the NaR rows, use them. If only color swatches are shown, label as "arm_1", "arm_2" in the order they appear top to bottom.
4. Numbers must monotonically decrease (or stay equal) across time within an arm. If you see a count that goes UP over time, you misread something — re-check or omit.

OUTPUT STRUCTURE (return JSON matching this shape exactly):
{
  "time_points": [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
  "time_unit": "months",
  "arms": [
    {
      "name": "Venetoclax + Azacitidine",
      "counts": [286, 256, 227, 196, 165, 138, 113, 87, 58, 31, 12]
    },
    {
      "name": "Placebo + Azacitidine",
      "counts": [145, 128, 105, 84, 62, 46, 33, 23, 14, 6, 2]
    }
  ],
  "extraction_notes": "Optional brief notes if any time points were dropped"
}

FIELD RULES:
- time_points: numeric only, in ascending order. The time values from the table column headers.
- time_unit: usually "months". If "weeks" or "days" set accordingly.
- arms: one entry per arm. Each `counts` array MUST have the same length as `time_points`.
- Arms in the same order as they appear in the NaR table (top to bottom).

If there is NO Number-at-Risk table visible in this image (some KM figures don't include one), return {"time_points": [], "arms": [], "extraction_notes": "No NaR table present"}.
"""

# ─────────────────────────────────────────────────────────────────
# PROMPT 3 — Curve coordinates (the visual KM line itself)
# ─────────────────────────────────────────────────────────────────
# This is the most accuracy-sensitive pass. We ask for ~80 (time, survival)
# coordinates per arm — enough to reflect fine-granularity step structure
# (NEJM-style curves have many small event drops) while staying within
# Pass 3's 60s per-call timeout. Adjusted upward 2026-04-24 from 50
# after observing the sequential pipeline handles 80 points reliably.
# The prompt:
#   - asks for SHAPE-PRESERVING sampling (not uniform)
#   - explicitly requires monotonic decrease per arm
#   - asks for axis range hints to anchor the coordinate space
#   - allows partial output if the model can't reach 80 confidently
CURVE_PROMPT = """You are a medical data extraction specialist. Read the Kaplan-Meier curve(s) from this figure and return (time, survival) coordinates that faithfully capture the STEP-FUNCTION character of each curve.

CRITICAL ANTI-HALLUCINATION RULES (non-negotiable):
1. Coordinates MUST come from carefully reading the actual plotted curve. Do NOT extrapolate, smooth, or fill in coordinates you cannot see.
2. If the curve becomes hard to read (e.g. multiple curves overlap, censoring marks obscure the path), provide fewer points rather than guessing.
3. Survival values MUST monotonically decrease (or stay flat) over time within each arm. KM curves NEVER go up. If you misread a point, OMIT it rather than recording an upward step.
4. Coordinates must respect the axis ranges shown in the figure. A point at (40 months, 95%) on a plot with x_max=30 is impossible — re-check.

OUTPUT TARGET: 80 coordinates per arm. CRITICAL — sample STRATEGICALLY, not uniformly:
  * Place 2 points at EVERY visible step/drop: one just BEFORE the drop (top of the step) and one just AFTER (bottom of the step). This preserves the step character.
  * On flat plateaus, place only 2-3 points (start and end of the plateau) — no need for dense sampling where nothing changes.
  * Concentrate MORE density around steep descent zones (where many events cluster).
  * This means: 80 points placed at MEANINGFUL locations is FAR better than 80 evenly-spaced points. A uniform sampling would average out the steps and destroy the KM shape.

Accept fewer points (down to ~50) if some sections are genuinely unreadable, but try to reach 80.

OUTPUT STRUCTURE (return JSON matching this shape exactly):
{
  "x_axis_unit": "months",
  "x_axis_min": 0,
  "x_axis_max": 30,
  "y_axis_scale": "percent",
  "y_axis_min": 0,
  "y_axis_max": 100,
  "arms": [
    {
      "name": "Venetoclax + Azacitidine",
      "color": "blue",
      "points": [
        {"t": 0.0, "s": 100.0},
        {"t": 1.5, "s": 96.5},
        {"t": 3.0, "s": 92.0},
        {"t": 4.5, "s": 87.5}
      ],
      "censoring_times": [2.1, 5.4, 8.7, 18.3]
    },
    {
      "name": "Placebo + Azacitidine",
      "color": "red",
      "points": [...],
      "censoring_times": [...]
    }
  ],
  "extraction_notes": "Optional brief notes about anything unclear or curve sections you skipped"
}

FIELD RULES:
- x_axis_unit: usually "months", sometimes "weeks" / "days" / "years". Read from x-axis label.
- x_axis_min / x_axis_max: the actual range of the X-axis ticks (e.g. 0 to 30 if ticks go 0, 5, 10, 15, 20, 25, 30).
- y_axis_scale: "percent" if Y-axis shows 0-100, or "fraction" if 0-1.0.
- y_axis_min / y_axis_max: read from the Y-axis tick range (typically 0-100 or 0-1.0).
- arms[].name: as printed in the legend, matching what you would extract for the metadata pass.
- arms[].color: simple color word (blue, red, green, ...) as it appears in the legend swatch.
- arms[].points[].t: time value, MUST fall within [x_axis_min, x_axis_max].
- arms[].points[].s: survival value, MUST fall within [y_axis_min, y_axis_max] AND monotonically decrease across the array.
- arms[].points: should start at or near (0, 100) — KM curves begin with all patients alive.
- arms[].censoring_times: list of times where censoring tick-marks ("|" or "+" symbols) appear on the curve. Only include marks you can clearly identify. Empty list [] if no marks visible.

If the image does NOT contain a Kaplan-Meier curve, return {"arms": [], "extraction_notes": "Not a Kaplan-Meier curve"}.
"""

# ─────────────────────────────────────────────────────────────────
# PROMPT 3-RETRY — Sharper version when first attempt produced
# coordinates whose implicit median doesn't match the published median
# ─────────────────────────────────────────────────────────────────
# Triggered when _validate_median_match returns "needs_reextract". The
# prompt is built dynamically with the published median value injected,
# so the model knows where to look more carefully. We also bump the
# point target slightly higher to give the renderer more material.
CURVE_RETRY_PROMPT_TEMPLATE = """You are a medical data extraction specialist. This is your SECOND attempt at extracting Kaplan-Meier curve coordinates from this figure. The first attempt produced coordinates that did not match the published median survival.

PUBLISHED ANCHORS (from the same publication — these are GROUND TRUTH):
{anchors_block}

YOUR TASK: Re-extract the curve coordinates with EXTRA CARE around the median crossing point. The coordinates you produce, when interpreted as a KM curve, MUST cross 50% survival at approximately the published median time.

CRITICAL ANTI-HALLUCINATION RULES (non-negotiable):
1. The published medians above are anchors, not data sources. Coordinates MUST come from carefully reading the plotted curve, not inferred from the median values.
2. If you find that you genuinely cannot read the curve precisely enough to match the published median, return your best honest extraction WITH a note in extraction_notes explaining the difficulty. Do NOT fabricate coordinates to force a match — that destroys the data integrity.
3. Survival values MUST monotonically decrease within each arm.

EXTRACTION FOCUS:
- Sample densely around the median crossing zone for each arm (where curve approaches and crosses 50%).
- Use the anchors above as a sanity check — if your reading puts the median far from the published value, look again at the curve before that region.
- Target 90 coordinates per arm in this retry (was 80 in the first attempt). Accept ≥60 if some sections are unreadable. Prioritize shape preservation (step character) over point density.

OUTPUT STRUCTURE: identical to the first-attempt prompt — same JSON schema with x_axis_unit, x_axis_min/max, y_axis_scale, y_axis_min/max, arms[]{{name, color, points[]{{t, s}}, censoring_times}}, extraction_notes.
"""


# ─────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────

def extract_km_vision(
    image_base64: str,
    source_hint: Optional[str] = None,
) -> dict:
    """Extract structured KM data from a curve PNG via Gemini Vision (two-pass).

    Pass 1 reads metadata (title, arms, HR, axis ranges).
    Pass 2 reads the Number-at-Risk table.
    Both passes use primary→fallback model routing for resilience.

    Args:
        image_base64: PNG (or JPEG) of the KM curve crop, base64-encoded.
                      Optionally with "data:image/png;base64," prefix.
        source_hint:  Optional study/paper name for logging context. Not sent
                      to the model — the model must read the image, not rely
                      on training memory.

    Returns:
        dict with these keys (always present, even on failure):
          - title:             figure title or ""
          - endpoint:          short endpoint code (OS, PFS, ...) or ""
          - arms:              list of arm dicts (see _validate_arms)
          - hr_value, hr_ci_low, hr_ci_high, p_value, p_value_operator
          - x_label, x_min, x_max, y_label, y_min, y_max
          - nar_time_points:   list of time values for NaR table
          - nar_time_unit:     "months" / "weeks" / "days"
          - nar_arms:          list of {name, counts[]} per arm
          - extraction_method_metadata: model id used for Pass 1
          - extraction_method_nar:      model id used for Pass 2
          - error:             None on success, or a string describing the
                               failure that prevented USABLE output. Soft
                               failures (e.g. NaR table missing, but
                               metadata present) leave error=None and just
                               return empty NaR arrays.
    """
    # ---- Setup ----
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _empty_result(error="GEMINI_API_KEY environment variable not set")

    try:
        import google.generativeai as genai
    except ImportError as e:
        return _empty_result(error=f"google-generativeai not installed: {e}")

    if image_base64.startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]

    try:
        png_bytes = base64.b64decode(image_base64)
    except Exception as e:
        return _empty_result(error=f"Invalid base64 image: {e}")

    if len(png_bytes) < 1000:
        return _empty_result(error=f"Image too small ({len(png_bytes)} bytes) — likely empty or corrupted")

    genai.configure(api_key=api_key)

    logger.info(
        f"KM extraction starting for source_hint='{source_hint}' "
        f"primary={GEMINI_PRIMARY_MODEL} fallback={GEMINI_FALLBACK_MODEL} "
        f"per-call timeout={GEMINI_TIMEOUT_PER_CALL_SECONDS}s "
        f"total budget={GEMINI_TOTAL_EXTRACTION_SECONDS}s"
    )

    # ──────────────────────────────────────────────────────────────
    # Run passes SEQUENTIALLY (2026-04-24 refactor from parallel).
    #
    # Why sequential over parallel?
    #   - Pass 3 (curve) is the heaviest and repeatedly 504-timed out in
    #     production, even after the 100→50 point reduction and the
    #     25s→60s timeout bump. Our hypothesis: parallel execution puts
    #     3 concurrent requests on the same (overloaded) Google endpoint
    #     queue at once, and Pass 3 — the longest — is the most likely
    #     victim of server-side deadline exhaustion.
    #   - Pass 1 (metadata) and Pass 2 (NaR) consistently succeed within
    #     10-20s each. By running them first, sequentially, we free up
    #     Google-side capacity for Pass 3 by the time it fires.
    #   - Sequential also enables GRACEFUL DEGRADATION: if Pass 1 fails
    #     we abort immediately (no wasted calls); if Pass 2 fails the
    #     slide still renders from metadata; if Pass 3 fails we keep
    #     metadata + NaR.
    #
    # Cost: wall time rises from ~60s (max of three) to ~90-120s (sum).
    # Still well within Railway's 300s edge timeout. For an on-demand
    # extraction the user is waiting on, this is an acceptable trade.
    #
    # We keep the _run_pass helper (with its per-call watchdog, retry,
    # and fallback-model logic) — it's already single-request-safe and
    # doesn't need changes for sequential use. The executor wrapping is
    # what goes away.
    # ──────────────────────────────────────────────────────────────
    deadline = time.monotonic() + GEMINI_TOTAL_EXTRACTION_SECONDS

    def _remaining_budget() -> float:
        """Remaining wall-clock budget before the total watchdog fires."""
        return max(0.0, deadline - time.monotonic())

    def _run_pass_with_deadline(prompt, label):
        """Run one pass. If the total deadline has already fired, skip this
        pass entirely and return a watchdog-timeout result — no point
        starting a call that can't complete within budget."""
        if _remaining_budget() < 2.0:
            logger.error(
                f"{label}: total watchdog ({GEMINI_TOTAL_EXTRACTION_SECONDS}s) "
                f"expired before this pass could start — skipping"
            )
            return (
                None, GEMINI_PRIMARY_MODEL,
                f"Total watchdog expired before {label} started",
            )
        return _run_pass(genai, png_bytes, prompt, label, source_hint)

    # ---- Pass 1: Metadata (title, arms, HR, axis ranges) ----
    # Runs first because: (a) without metadata we can't validate or render
    # anything, so early failure here means we abort the whole extraction;
    # (b) it's the lightest pass (typically 10-15s), so we spend little
    # budget even in the pathological case.
    logger.info(f"Running Pass 1 (metadata) sequentially for source='{source_hint}'")
    metadata_data, metadata_model, metadata_error = _run_pass_with_deadline(
        METADATA_PROMPT, "Pass 1 (metadata)"
    )

    # ---- Early-abort on metadata failure ----
    # Unlike the parallel version, we abort here BEFORE wasting budget on
    # NaR and Curve passes that would produce unusable output without arm
    # names / axis ranges.
    if metadata_error and not metadata_data:
        logger.error(
            f"Pass 1 (metadata) failed — aborting extraction. "
            f"NaR/curve passes skipped to save budget. Error: {metadata_error}"
        )
        return _empty_result(
            error=f"Pass 1 (metadata) failed: {metadata_error}",
            metadata_model=metadata_model,
        )

    # ---- Pass 2: Number-at-Risk table ----
    # Runs second. Soft-fail is acceptable: a KM slide without NaR is still
    # informative; the renderer simply omits the per-time-point counts strip.
    logger.info(
        f"Pass 1 (metadata) complete in {GEMINI_TOTAL_EXTRACTION_SECONDS - _remaining_budget():.1f}s — "
        f"running Pass 2 (NaR)"
    )
    nar_data, nar_model, nar_error = _run_pass_with_deadline(
        NAR_PROMPT, "Pass 2 (NaR)"
    )

    # ---- Pass 3: Curve coordinates ----
    # Runs last so that (a) earlier passes succeeded under lighter API load,
    # (b) Pass 3 gets whatever budget remains — typically 90-120s if
    # Pass 1 + Pass 2 were healthy.
    logger.info(
        f"Pass 2 (NaR) complete, remaining budget {_remaining_budget():.1f}s — "
        f"running Pass 3 (curve)"
    )
    curve_data, curve_model, curve_error = _run_pass_with_deadline(
        CURVE_PROMPT, "Pass 3 (curve)"
    )

    # Defensive: Gemini occasionally returns a JSON list at the top level
    # instead of the object we asked for (schema violation). The rest of
    # this function assumes metadata_data is a dict (or None), so we
    # coerce any non-dict shape to None and treat it as a soft failure.
    # This was observed 2026-04-23 during a VIALE-A run that crashed with
    # "'list' object has no attribute 'get'" at the _validate_arms call.
    if metadata_data is not None and not isinstance(metadata_data, dict):
        logger.warning(
            f"Pass 1 (metadata) returned non-dict type "
            f"{type(metadata_data).__name__} instead of object — "
            f"treating as empty metadata. Raw start: "
            f"{str(metadata_data)[:200]!r}"
        )
        metadata_data = None

    # ---- Validate, normalise, cross-check ----
    arms_validated, arms_dropped = _validate_arms(metadata_data.get("arms") if metadata_data else [])
    nar_validated, nar_dropped = _validate_nar(
        nar_data,
        arm_count_hint=len(arms_validated),
    )

    # ---- Final assembly ----
    md = metadata_data or {}
    result = {
        "title":            _safe_str(md.get("title"), "", max_len=200),
        "endpoint":         _safe_str(md.get("endpoint"), "", max_len=10).upper(),
        "arms":             arms_validated,
        "arms_dropped":     arms_dropped,
        "hr_value":         _to_float(md.get("hr_value")),
        "hr_ci_low":        _to_float(md.get("hr_ci_low")),
        "hr_ci_high":       _to_float(md.get("hr_ci_high")),
        "p_value":          _to_float(md.get("p_value")),
        "p_value_operator": _safe_str(md.get("p_value_operator"), "", max_len=2),
        "x_label":          _safe_str(md.get("x_label"), "Time (months)", max_len=80),
        "x_min":            _to_float(md.get("x_min")) or 0.0,
        "x_max":             _to_float(md.get("x_max")) or 36.0,
        "y_label":          _safe_str(md.get("y_label"), "Survival (%)", max_len=80),
        "y_min":             _to_float(md.get("y_min")) or 0.0,
        "y_max":             _to_float(md.get("y_max")) or 100.0,
        "nar_time_points":  nar_validated.get("time_points", []),
        "nar_time_unit":    nar_validated.get("time_unit", "months"),
        "nar_arms":         nar_validated.get("arms", []),
        "nar_dropped":      nar_dropped,
        # Curve fields — populated below after Pass 3 validation + possible retry
        "curve_x_axis_unit":  "months",
        "curve_x_axis_min":   0.0,
        "curve_x_axis_max":   36.0,
        "curve_y_axis_min":   0.0,
        "curve_y_axis_max":   100.0,
        "curve_arms":         [],
        "curves_dropped":     [],
        "median_validation":  None,  # dict set after validation runs
        "curve_reextract_attempted": False,
        "extraction_method_metadata": metadata_model,
        "extraction_method_nar":      nar_model,
        "extraction_method_curve":    curve_model,
        "error":            None,
    }

    # ──────────────────────────────────────────────────────────────
    # Pass 3 validation + median-match + re-extract logic
    # ──────────────────────────────────────────────────────────────
    #
    # Decision tree:
    #   1. If curve_error and no data → leave curves empty, but keep the
    #      slide buildable from metadata + NaR (renderer will note tier 3).
    #   2. If curve data present → validate against arm_count from Pass 1.
    #   3. Compute implicit median per arm, compare with published median.
    #   4. Verdict:
    #        "match"           → accept curves, done.
    #        "needs_reextract" → rerun Pass 3 with the sharper retry prompt
    #                            (injecting published medians as anchors).
    #                            Re-validate the retry result. If it still
    #                            fails, fall through to hard_fail handling.
    #        "hard_fail"       → set error with clear user-facing message,
    #                            leave curves empty so no misleading slide
    #                            is rendered. Frontend shows the error.
    # ──────────────────────────────────────────────────────────────

    if curve_data and not curve_error:
        curves_validated, curves_dropped = _validate_curves(
            curve_data, expected_arm_count=len(arms_validated),
        )
        validation = _validate_median_match(
            curves_validated, metadata_data or {}, source_hint,
        )

        logger.info(
            f"KM curve Pass 3 first attempt for source='{source_hint}': "
            f"{validation['summary']}"
        )

        # Re-extract branch ────────────────────────────────────────
        if validation["verdict"] == "needs_reextract":
            logger.warning(
                f"KM curve for source='{source_hint}' — median mismatch in "
                f"tolerance window (10-25%). Re-extracting with sharper prompt."
            )
            result["curve_reextract_attempted"] = True

            retry_prompt = _build_retry_prompt(metadata_data or {}, validation)

            # Retry uses same parallel-safe pattern but standalone.
            # Budget: whatever's left of the total watchdog.
            remaining_budget = max(5.0, deadline - time.monotonic())
            try:
                retry_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="km-vision-retry"
                )
                try:
                    retry_future = retry_executor.submit(
                        _run_pass, genai, png_bytes, retry_prompt,
                        "curve-retry", source_hint,
                    )
                    retry_curve_data, retry_curve_model, retry_curve_error = (
                        retry_future.result(timeout=remaining_budget)
                    )
                finally:
                    retry_executor.shutdown(wait=False)
            except concurrent.futures.TimeoutError:
                logger.error(
                    f"KM curve retry for source='{source_hint}' exceeded "
                    f"remaining budget ({remaining_budget:.1f}s)"
                )
                retry_curve_data, retry_curve_model, retry_curve_error = (
                    None, curve_model, "retry timeout"
                )

            if retry_curve_data and not retry_curve_error:
                retry_curves_validated, retry_curves_dropped = _validate_curves(
                    retry_curve_data, expected_arm_count=len(arms_validated),
                )
                retry_validation = _validate_median_match(
                    retry_curves_validated, metadata_data or {}, source_hint,
                )
                logger.info(
                    f"KM curve Pass 3 RETRY for source='{source_hint}': "
                    f"{retry_validation['summary']}"
                )

                # Only accept retry if it's strictly better (or match).
                # If retry is still needs_reextract, we accept it — the
                # sharper prompt did its best and we don't retry again.
                if retry_validation["verdict"] in ("match", "needs_reextract"):
                    curves_validated = retry_curves_validated
                    curves_dropped = retry_curves_dropped + curves_dropped
                    validation = retry_validation
                    result["extraction_method_curve"] = (
                        f"{curve_model} → retry:{retry_curve_model}"
                    )
                elif retry_validation["verdict"] == "hard_fail":
                    # Retry made it worse. Keep first attempt's validation
                    # for the hard_fail decision below.
                    logger.warning(
                        f"KM curve retry for source='{source_hint}' produced "
                        f"hard_fail; reverting to first-attempt validation."
                    )
                    validation = retry_validation  # still hard_fail
            else:
                logger.warning(
                    f"KM curve retry for source='{source_hint}' returned no "
                    f"usable data: {retry_curve_error}. Falling through to "
                    f"first-attempt validation (was: needs_reextract)."
                )
                # Keep first attempt's curves — they were within the soft
                # tolerance band anyway, so they're usable with a warning.

        # Final verdict handling ──────────────────────────────────
        if validation["verdict"] == "hard_fail":
            logger.error(
                f"KM curve for source='{source_hint}' hard_fail after "
                f"{'retry' if result['curve_reextract_attempted'] else 'first attempt'}: "
                f"{validation['summary']}. Not rendering curve — user message set."
            )
            result["error"] = (
                "Curve reconstruction failed: the extracted coordinates do not "
                "match the publication's reported median survival (>25% deviation). "
                "This usually means the source image resolution is too low for "
                "accurate reading. Please re-upload a higher-resolution version "
                "of the publication PDF, or skip the KM slide."
            )
            result["median_validation"] = validation
            # Leave curve_arms empty so the renderer produces no slide
        else:
            # Accept curves (match OR soft-warn in tolerance band)
            result["curve_x_axis_unit"] = curves_validated["x_axis_unit"]
            result["curve_x_axis_min"]  = curves_validated["x_axis_min"]
            result["curve_x_axis_max"]  = curves_validated["x_axis_max"]
            result["curve_y_axis_min"]  = curves_validated["y_axis_min"]
            result["curve_y_axis_max"]  = curves_validated["y_axis_max"]
            result["curve_arms"]        = curves_validated["arms"]
            result["curves_dropped"]    = curves_dropped
            result["median_validation"] = validation
    else:
        logger.warning(
            f"KM curve Pass 3 failed for source='{source_hint}': {curve_error}. "
            f"Slide can still be rendered from metadata + NaR but without curve geometry."
        )
        # curve_arms stays empty; error message about curve failure is
        # optional — metadata/NaR-only slides are still useful.

    # ---- HR consistency check ----
    # If hr_value is outside its CI, log it but don't fail — Gemini may
    # have read the box correctly but the CI was unusually wide; drop the
    # HR rather than the CI so the rendered figure is at least internally
    # consistent.
    if (
        result["hr_value"] is not None
        and result["hr_ci_low"] is not None
        and result["hr_ci_high"] is not None
    ):
        lo, hi = result["hr_ci_low"], result["hr_ci_high"]
        if lo > hi:
            lo, hi = hi, lo
            result["hr_ci_low"], result["hr_ci_high"] = lo, hi
        if not (lo * 0.95 <= result["hr_value"] <= hi * 1.05):
            logger.warning(
                f"HR {result['hr_value']} outside CI [{lo}, {hi}] for "
                f"source='{source_hint}' — keeping HR, dropping CI"
            )
            result["hr_ci_low"] = None
            result["hr_ci_high"] = None

    logger.info(
        f"KM extraction done for source='{source_hint}': "
        f"{len(arms_validated)} arms (metadata via {metadata_model}), "
        f"{len(result['nar_arms'])} NaR rows × "
        f"{len(result['nar_time_points'])} time pts (NaR via {nar_model}), "
        f"{len(result['curve_arms'])} curve arms × "
        f"{sum(len(a.get('points', [])) for a in result['curve_arms'])} total pts "
        f"(curve via {result['extraction_method_curve']}); "
        f"verdict={result['median_validation']['verdict'] if result['median_validation'] else 'n/a'}"
    )
    return result


# ─────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────

def _run_pass(genai_module, png_bytes: bytes, prompt: str, label: str, source_hint: Optional[str]) -> tuple:
    """Run one Gemini Vision pass with primary→fallback routing AND retry.

    Per-attempt timeline:
      attempt 1: primary model, watchdog GEMINI_TIMEOUT_PER_CALL_SECONDS
      attempt 2: same primary model retry (only on Timeout)
      attempt 3: fallback model (only on model-unavailable error)
      attempt 4: fallback model retry (only on Timeout)
    Worst case: ~4 × 25s = 100s, but the outer watchdog caps at
    GEMINI_TOTAL_EXTRACTION_SECONDS (75s) so this is bounded.

    Returns (parsed_dict_or_None, model_id_used, error_message_or_None).
    """
    # ── Try primary model with retry-on-timeout ──
    primary_exc = None
    for attempt in range(1 + GEMINI_RETRIES_PER_PASS):
        try:
            data = _call_gemini_with_watchdog(
                genai_module, GEMINI_PRIMARY_MODEL, png_bytes, prompt
            )
            if attempt > 0:
                logger.info(
                    f"KM {label}: succeeded on retry attempt {attempt+1} "
                    f"with {GEMINI_PRIMARY_MODEL}"
                )
            return data, GEMINI_PRIMARY_MODEL, None
        except concurrent.futures.TimeoutError as te:
            primary_exc = te
            if attempt < GEMINI_RETRIES_PER_PASS:
                logger.warning(
                    f"KM {label}: timed out on attempt {attempt+1} with "
                    f"{GEMINI_PRIMARY_MODEL} — retrying"
                )
                continue
            # Timeout exhausted retries; fall through to fallback model
            logger.warning(
                f"KM {label}: timed out {attempt+1}× on {GEMINI_PRIMARY_MODEL} — "
                f"trying fallback model"
            )
            break
        except Exception as exc:
            primary_exc = exc
            if _is_model_unavailable_error(exc):
                # Don't retry on a model-unavailable error — it won't recover
                logger.error(
                    f"⚠️  PRIMARY MODEL UNAVAILABLE for KM {label}: "
                    f"'{GEMINI_PRIMARY_MODEL}' failed with: {exc}. "
                    f"Falling back to '{GEMINI_FALLBACK_MODEL}'. ACTION: investigate "
                    f"primary model status and update GEMINI_KM_MODEL env var if "
                    f"it was renamed/deprecated."
                )
                break
            if _is_transient_server_error(exc):
                # Transient: server-side deadline / 504 / overloaded. Retry
                # on the SAME model (might clear), and if we've exhausted
                # retries, fall through to the fallback model.
                if attempt < GEMINI_RETRIES_PER_PASS:
                    logger.warning(
                        f"KM {label}: transient server error on attempt {attempt+1} "
                        f"with {GEMINI_PRIMARY_MODEL} ({exc}) — retrying same model"
                    )
                    continue
                logger.warning(
                    f"KM {label}: transient server error {attempt+1}× on "
                    f"{GEMINI_PRIMARY_MODEL} ({exc}) — trying fallback model"
                )
                break
            # Other exceptions (auth, quota, malformed response, etc.) —
            # retrying the same model won't help, and the fallback model
            # would hit the same auth/quota issue. Return early.
            logger.warning(
                f"KM {label}: failed on {GEMINI_PRIMARY_MODEL} "
                f"(no retry — not a transient error): {exc}"
            )
            return None, GEMINI_PRIMARY_MODEL, str(exc)

    # ── Try fallback model with retry-on-timeout ──
    if not GEMINI_FALLBACK_MODEL or GEMINI_FALLBACK_MODEL == GEMINI_PRIMARY_MODEL:
        return None, GEMINI_PRIMARY_MODEL, f"Primary model failed: {primary_exc}"

    fallback_exc = None
    for attempt in range(1 + GEMINI_RETRIES_PER_PASS):
        try:
            data = _call_gemini_with_watchdog(
                genai_module, GEMINI_FALLBACK_MODEL, png_bytes, prompt
            )
            if attempt > 0:
                logger.info(
                    f"KM {label}: succeeded on retry attempt {attempt+1} "
                    f"with fallback {GEMINI_FALLBACK_MODEL}"
                )
            return data, GEMINI_FALLBACK_MODEL, None
        except concurrent.futures.TimeoutError as te:
            fallback_exc = te
            if attempt < GEMINI_RETRIES_PER_PASS:
                logger.warning(
                    f"KM {label}: timed out on attempt {attempt+1} with "
                    f"fallback {GEMINI_FALLBACK_MODEL} — retrying"
                )
                continue
            break
        except Exception as exc:
            fallback_exc = exc
            if _is_transient_server_error(exc) and attempt < GEMINI_RETRIES_PER_PASS:
                # Fallback model also hit a transient issue. One retry
                # before giving up — two transient errors in a row on a
                # production model is rare but worth the cost.
                logger.warning(
                    f"KM {label}: transient server error on attempt {attempt+1} "
                    f"with fallback {GEMINI_FALLBACK_MODEL} ({exc}) — retrying"
                )
                continue
            break

    logger.error(
        f"BOTH MODELS FAILED for KM {label} — primary "
        f"'{GEMINI_PRIMARY_MODEL}': {primary_exc}; fallback "
        f"'{GEMINI_FALLBACK_MODEL}': {fallback_exc}"
    )
    return (
        None, GEMINI_FALLBACK_MODEL,
        f"Both models failed. Primary: {primary_exc}. Fallback: {fallback_exc}",
    )


def _call_gemini_with_watchdog(genai_module, model_id: str, png_bytes: bytes, prompt: str) -> dict:
    """Wrap _call_gemini in a per-call watchdog using a SECOND-level
    ThreadPoolExecutor.

    Why a nested executor instead of relying on the SDK's request_options
    timeout? Across SDK versions and grpc internals, the SDK timeout is
    sometimes not honoured (the underlying grpc call hangs in C code that
    Python's GIL can't interrupt). A wall-clock Future.result(timeout=)
    gives us a guaranteed ceiling: if the worker thread doesn't return
    within the budget, we raise TimeoutError and let the caller decide
    (retry / fallback / give up). The orphaned thread will eventually
    exit on its own.

    Raises:
      concurrent.futures.TimeoutError — call exceeded per-call budget
      ValueError — invalid response from model (empty / unparseable JSON)
      Exception — any other SDK error (auth, quota, model unavailable, ...)
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"km-call-{model_id}"
    ) as inner:
        fut = inner.submit(_call_gemini, genai_module, model_id, png_bytes, prompt)
        return fut.result(timeout=GEMINI_TIMEOUT_PER_CALL_SECONDS)
        # NOTE: if timeout fires, we exit the `with` block without cancelling
        # the running future. ThreadPoolExecutor's __exit__ calls shutdown(
        # wait=True) which would block. We accept that pattern here because
        # the call only happens at the leaf of the call tree and the parent
        # outer watchdog (GEMINI_TOTAL_EXTRACTION_SECONDS) protects us.
        # In practice the inner thread completes shortly after we abandon
        # it because grpc respects its own request deadline.


def _call_gemini(genai_module, model_id: str, png_bytes: bytes, prompt: str) -> dict:
    """Single Gemini Vision call. Returns parsed JSON dict or raises.

    Identical contract to the Forest wrapper's _call_gemini — the prompt
    is a parameter so this is reusable across the two passes.

    Layer 1 of the timeout strategy: ask the SDK to abort the underlying
    HTTP request if it exceeds GEMINI_TIMEOUT_PER_CALL_SECONDS. This is
    the polite path — when honoured, the request is cancelled cleanly and
    no thread leaks. Layer 2 (the watchdog in _call_gemini_with_watchdog)
    is the enforcement.
    """
    model = genai_module.GenerativeModel(model_id)
    response = model.generate_content(
        [
            prompt,
            {"mime_type": "image/png", "data": png_bytes},
        ],
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.0,
        },
        request_options={"timeout": GEMINI_TIMEOUT_PER_CALL_SECONDS},
    )

    raw_text = (getattr(response, "text", None) or "").strip()
    if not raw_text:
        raise ValueError(f"Empty response from {model_id}")

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        # Strip ```json ... ``` code-fence wrappers some models add
        cleaned = raw_text.strip().lstrip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
        cleaned = cleaned.rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            raise ValueError(
                f"Invalid JSON from {model_id}: {e}. "
                f"First 200 chars: {raw_text[:200]!r}"
            )


def _is_model_unavailable_error(exc: Exception) -> bool:
    """Detect 'model is gone/renamed/unavailable' errors that should
    trigger fallback. Network/auth/timeout errors do NOT match — they
    would just fail again on the fallback model."""
    msg = str(exc).lower()
    return any(indicator in msg for indicator in _MODEL_UNAVAILABLE_INDICATORS)


def _is_transient_server_error(exc: Exception) -> bool:
    """Detect transient server-side errors (504, 503, overloaded queues,
    server-side timeout) that should trigger retry-then-fallback.

    Different from _is_model_unavailable_error: the model itself is fine,
    but this specific request hit infrastructure trouble. Retrying may
    succeed; if it doesn't, the fallback model (likely on different
    infrastructure) is our best recovery path.

    Critical distinction: we check for transient BEFORE model-unavailable
    in the call site, because a 504 response might coincidentally match
    less-specific indicators. Keeping the two sets conceptually separate
    makes the logic easier to reason about."""
    msg = str(exc).lower()
    return any(indicator in msg for indicator in _TRANSIENT_SERVER_INDICATORS)


def _validate_arms(raw_arms) -> tuple[list, list]:
    """Validate Pass 1 arms list. Returns (cleaned, dropped_with_reasons).

    Each kept arm has: name (str), n (int|None), color (str), median_value
    (float|None), median_unit (str). Median can be None for "Not Reached".
    """
    cleaned: list[dict] = []
    dropped: list[dict] = []

    if not isinstance(raw_arms, list):
        return cleaned, [{"raw": raw_arms, "reason": "arms field is not a list"}]

    for arm in raw_arms:
        if not isinstance(arm, dict):
            dropped.append({"raw": arm, "reason": "arm is not an object"})
            continue

        name = _safe_str(arm.get("name"), "", max_len=80)
        if not name:
            dropped.append({"raw": arm, "reason": "arm missing name"})
            continue

        # n: optional integer. Negative or absurd values rejected.
        n_raw = arm.get("n")
        n: Optional[int] = None
        if n_raw is not None:
            try:
                n_int = int(float(n_raw))
                if 0 < n_int < 100000:
                    n = n_int
            except (TypeError, ValueError):
                pass

        # color: simple word, lowercased
        color = _safe_str(arm.get("color"), "", max_len=20).lower()

        # median: positive finite float, plausible upper bound (years
        # could be 100+, months could be 60+, weeks could be many; cap
        # at 500 units which covers any realistic study)
        median = _to_float(arm.get("median_value"))
        if median is not None and not (0 < median <= 500):
            median = None

        median_unit = _safe_str(arm.get("median_unit"), "months", max_len=20).lower()

        cleaned.append({
            "name":         name,
            "n":            n,
            "color":        color,
            "median_value": median,
            "median_unit":  median_unit,
        })

    return cleaned, dropped


def _validate_nar(raw_nar, arm_count_hint: int) -> tuple[dict, list]:
    """Validate Pass 2 NaR data. Returns (cleaned_nar_dict, dropped_rows).

    Cleaned dict has: time_points (list[float]), time_unit (str),
    arms (list of {name, counts}). counts arrays are aligned with
    time_points and validated for monotonic decrease.

    arm_count_hint is the number of arms Pass 1 confirmed. If Pass 2
    returned more, we keep only the first N (Pass 1 is source of truth
    for arm count). If Pass 2 returned fewer, that's fine — NaR is
    optional, and missing rows just result in a renderer that draws
    arms without per-time-point counts.
    """
    dropped: list[dict] = []
    empty_result = {"time_points": [], "time_unit": "months", "arms": []}

    if not isinstance(raw_nar, dict):
        return empty_result, [{"raw": raw_nar, "reason": "NaR response is not a dict"}]

    raw_time_points = raw_nar.get("time_points")
    if not isinstance(raw_time_points, list):
        return empty_result, [{"raw": raw_nar, "reason": "time_points is not a list"}]

    # Coerce time points to floats; drop non-numeric and ensure ascending
    time_points: list[float] = []
    for tp in raw_time_points:
        f = _to_float(tp)
        if f is not None and 0 <= f <= 1000:
            time_points.append(f)

    if not time_points:
        return empty_result, [{"raw": raw_nar, "reason": "no valid numeric time_points"}]

    # Sort ascending and dedupe (some models repeat columns)
    time_points = sorted(set(time_points))

    time_unit = _safe_str(raw_nar.get("time_unit"), "months", max_len=20).lower()

    raw_arms = raw_nar.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        return empty_result, [{"raw": raw_nar, "reason": "arms list missing or empty"}]

    cleaned_arms: list[dict] = []
    for arm in raw_arms:
        if not isinstance(arm, dict):
            dropped.append({"raw": arm, "reason": "NaR arm is not an object"})
            continue

        name = _safe_str(arm.get("name"), "", max_len=80)
        if not name:
            name = f"arm_{len(cleaned_arms) + 1}"

        raw_counts = arm.get("counts")
        if not isinstance(raw_counts, list):
            dropped.append({"raw": arm, "reason": "counts is not a list"})
            continue

        # Coerce to ints, reject negatives
        counts: list[int] = []
        for c in raw_counts:
            try:
                ci = int(float(c))
                if ci < 0 or ci > 100000:
                    counts = []
                    break
                counts.append(ci)
            except (TypeError, ValueError):
                counts = []
                break

        if not counts:
            dropped.append({"raw": arm, "reason": "non-numeric or out-of-range count value"})
            continue

        # Truncate to time_points length (Gemini sometimes returns extra
        # columns or misaligned arrays). If too short, pad with the last
        # known value — but log it as a minor issue.
        if len(counts) > len(time_points):
            counts = counts[:len(time_points)]
        elif len(counts) < len(time_points):
            dropped.append({
                "raw": arm,
                "reason": (
                    f"counts length {len(counts)} < time_points length "
                    f"{len(time_points)} — partial NaR row dropped to keep "
                    f"data consistent"
                ),
            })
            continue

        # Enforce monotonic decrease — small upticks (1-2 patients) can
        # be OCR errors, snap them down to the previous value rather than
        # dropping the entire arm.
        for i in range(1, len(counts)):
            if counts[i] > counts[i-1]:
                counts[i] = counts[i-1]

        cleaned_arms.append({"name": name, "counts": counts})

    # Cross-check vs Pass 1 arm count
    if arm_count_hint > 0 and len(cleaned_arms) > arm_count_hint:
        excess = cleaned_arms[arm_count_hint:]
        cleaned_arms = cleaned_arms[:arm_count_hint]
        for ex in excess:
            dropped.append({
                "raw": ex,
                "reason": (
                    f"Pass 1 found {arm_count_hint} arms, Pass 2 returned more — "
                    f"trimmed extra row to keep cross-pass consistency"
                ),
            })

    return {
        "time_points": time_points,
        "time_unit":   time_unit,
        "arms":        cleaned_arms,
    }, dropped


# ─────────────────────────────────────────────────────────────────
# Curve validation, median computation, and match-checking
# ─────────────────────────────────────────────────────────────────

# Per-arm point count thresholds. Below MIN_USABLE the arm is dropped.
# Values adjusted 2026-04-24 to 50/80 after confirming the sequential
# pipeline + 60s per-call timeout can handle 80 points reliably.
# Previous iteration (30/50) produced visibly jagged curves; the NEJM
# VIALE-A reference has fine granularity with many small event drops,
# and we want our reconstruction to reflect that step character.
# 80 points captures the fine-granularity step structure without
# timing out; the renderer's step function will look noticeably
# smoother than at 50 points while still being an honest step plot
# (never smoothed or interpolated between extracted points).
CURVE_MIN_USABLE_POINTS_PER_ARM = 50
CURVE_TARGET_POINTS_PER_ARM     = 80

# Median-match validation thresholds (relative tolerance vs published value)
MEDIAN_MATCH_TOLERANCE       = 0.10   # ±10% → MATCH
MEDIAN_MATCH_HARD_FAIL       = 0.25   # ±>25% after retry → REJECT


def _validate_curves(raw_curve_data, expected_arm_count: int) -> tuple[dict, list]:
    """Validate Pass 3 curve data. Returns (cleaned_curve_dict, dropped_with_reasons).

    Cleaned dict has:
      - x_axis_unit, x_axis_min, x_axis_max, y_axis_scale, y_axis_min, y_axis_max
      - arms: list of {name, color, points:[{t,s}], censoring_times:[]}
    Each arm's points are sorted by time, monotonically decreasing in survival,
    and clipped to the axis bounds. Arms with fewer than CURVE_MIN_USABLE_POINTS_PER_ARM
    valid points after cleaning are dropped — better no curve than a 5-point fragment.

    Survival values are normalised to PERCENT (0-100) regardless of input scale,
    so the renderer has a consistent input contract.
    """
    dropped: list[dict] = []
    empty_result = {
        "x_axis_unit": "months", "x_axis_min": 0.0, "x_axis_max": 36.0,
        "y_axis_scale": "percent", "y_axis_min": 0.0, "y_axis_max": 100.0,
        "arms": [],
    }

    if not isinstance(raw_curve_data, dict):
        return empty_result, [{"raw": raw_curve_data, "reason": "curve response is not a dict"}]

    # Axis metadata — fall back to defaults on missing/invalid
    x_min = _to_float(raw_curve_data.get("x_axis_min")) or 0.0
    x_max = _to_float(raw_curve_data.get("x_axis_max")) or 36.0
    if x_max <= x_min:
        x_max = x_min + 36.0  # absurd range — coerce to something usable

    y_min = _to_float(raw_curve_data.get("y_axis_min")) or 0.0
    y_max_raw = _to_float(raw_curve_data.get("y_axis_max")) or 100.0

    y_axis_scale = _safe_str(raw_curve_data.get("y_axis_scale"), "percent", max_len=20).lower()
    # If scale is fraction (0-1.0), we'll multiply incoming s by 100 to normalise.
    is_fraction_scale = (y_axis_scale == "fraction") or (0 < y_max_raw <= 1.5)
    y_max_pct = 100.0  # always normalise to percent

    cleaned_arms: list[dict] = []
    raw_arms = raw_curve_data.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        return empty_result, [{"raw": raw_curve_data, "reason": "no arms in curve data"}]

    for arm in raw_arms:
        if not isinstance(arm, dict):
            dropped.append({"raw": arm, "reason": "curve arm is not an object"})
            continue

        name = _safe_str(arm.get("name"), "", max_len=80)
        if not name:
            name = f"arm_{len(cleaned_arms) + 1}"
        color = _safe_str(arm.get("color"), "", max_len=20).lower()

        raw_points = arm.get("points")
        if not isinstance(raw_points, list) or not raw_points:
            dropped.append({"raw": arm, "reason": f"arm '{name}' has no points list"})
            continue

        # Coerce points to (t, s) pairs in PERCENT scale
        pts: list[tuple[float, float]] = []
        for p in raw_points:
            if not isinstance(p, dict):
                continue
            t = _to_float(p.get("t"))
            s = _to_float(p.get("s"))
            if t is None or s is None:
                continue
            if is_fraction_scale and s <= 1.5:
                s *= 100.0
            # Clip to plausible bounds
            if t < x_min - 0.01 or t > x_max + 0.01:
                continue  # out of x-axis range, drop silently
            if s < -0.01 or s > 100.5:
                continue  # impossible survival, drop silently
            s = max(0.0, min(100.0, s))
            pts.append((t, s))

        if not pts:
            dropped.append({"raw": arm, "reason": f"arm '{name}' had no valid points after coercion"})
            continue

        # Sort by time
        pts.sort(key=lambda p: p[0])

        # Enforce monotonic decrease — small upticks (1-2%) are likely OCR
        # noise, snap them down. Large upticks are likely misreads, drop them.
        cleaned_pts: list[tuple[float, float]] = [pts[0]]
        for (t, s) in pts[1:]:
            prev_s = cleaned_pts[-1][1]
            if s <= prev_s:
                cleaned_pts.append((t, s))
            elif s - prev_s <= 2.0:  # small uptick → snap down
                cleaned_pts.append((t, prev_s))
            else:
                # Large uptick: drop the offending point rather than corrupt the curve
                continue

        # Censoring marks (optional)
        raw_cens = arm.get("censoring_times") or []
        cens_times: list[float] = []
        if isinstance(raw_cens, list):
            for ct in raw_cens:
                f = _to_float(ct)
                if f is not None and x_min <= f <= x_max:
                    cens_times.append(f)
            cens_times.sort()

        if len(cleaned_pts) < CURVE_MIN_USABLE_POINTS_PER_ARM:
            dropped.append({
                "raw": {"name": name, "n_points_raw": len(raw_points), "n_points_cleaned": len(cleaned_pts)},
                "reason": (
                    f"arm '{name}' has only {len(cleaned_pts)} usable points "
                    f"(min {CURVE_MIN_USABLE_POINTS_PER_ARM} required) — dropped"
                ),
            })
            continue

        cleaned_arms.append({
            "name": name,
            "color": color,
            "points": [{"t": t, "s": s} for (t, s) in cleaned_pts],
            "censoring_times": cens_times,
        })

    # Cross-check arm count against metadata pass
    if expected_arm_count > 0 and len(cleaned_arms) > expected_arm_count:
        excess = cleaned_arms[expected_arm_count:]
        cleaned_arms = cleaned_arms[:expected_arm_count]
        for ex in excess:
            dropped.append({
                "raw": {"name": ex["name"]},
                "reason": (
                    f"Pass 1 found {expected_arm_count} arms, Pass 3 returned more — "
                    f"trimmed extra arm '{ex['name']}' for cross-pass consistency"
                ),
            })

    return {
        "x_axis_unit": _safe_str(raw_curve_data.get("x_axis_unit"), "months", max_len=20).lower(),
        "x_axis_min": x_min,
        "x_axis_max": x_max,
        "y_axis_scale": "percent",  # always normalised
        "y_axis_min": 0.0,
        "y_axis_max": y_max_pct,
        "arms": cleaned_arms,
    }, dropped


def _compute_implicit_median(arm_points: list) -> Optional[float]:
    """Find the time t* where the curve crosses survival = 50%.

    Uses linear interpolation between the two adjacent points that bracket
    50%. Returns None if the curve never reaches 50% (i.e. median NR/NE).

    Input: list of {"t": float, "s": float} dicts in monotonic-decreasing
    survival order, with s in PERCENT.
    """
    if not arm_points or len(arm_points) < 2:
        return None
    # Find first index where s ≤ 50
    for i, p in enumerate(arm_points):
        if p["s"] <= 50.0:
            if i == 0:
                # Curve already below 50 at first point — pathological
                return p["t"]
            prev = arm_points[i - 1]
            curr = p
            # Linear interp: at what t did s cross 50?
            ds = curr["s"] - prev["s"]
            if abs(ds) < 1e-9:
                return curr["t"]
            frac = (50.0 - prev["s"]) / ds  # 0..1 between prev and curr
            return prev["t"] + frac * (curr["t"] - prev["t"])
    # Never crossed 50 — median not reached in observation window
    return None


def _validate_median_match(curves_validated: dict, metadata: dict, source_hint: Optional[str]) -> dict:
    """Cross-validate Pass 3 curve coordinates against Pass 1 published medians.

    For each arm in `curves_validated`:
      1. Compute the implicit median from the curve points.
      2. Look up the published median for the same arm in `metadata['arms']`
         (matched by case-insensitive name substring).
      3. Compare and classify the arm match.

    Returns a dict:
      {
        "verdict":    "match" | "needs_reextract" | "hard_fail",
        "arm_results": [
          {"name", "implicit_median", "published_median",
           "relative_error", "status"  # "match"|"warn"|"fail"|"unverifiable"|"nr"
          }
        ],
        "summary":    human-readable summary string,
      }

    Verdict rules:
      - "hard_fail":         any arm has rel_error > MEDIAN_MATCH_HARD_FAIL (>25%)
      - "needs_reextract":   any arm has rel_error in (TOLERANCE, HARD_FAIL] (10-25%)
      - "match":             all arms within tolerance, OR all arms have published
                             median NR (no anchor to validate against) — the latter
                             is treated as "match" because we have no basis to fail
      - Arms with no published median (NR/NE) are marked "nr" and do NOT trigger
        a re-extract on their own.
      - Arms where the curve never reaches 50% but the publication says it should
        have are marked "fail" — that's a strong signal of bad coordinates.
    """
    arm_results: list[dict] = []
    has_hard_fail = False
    needs_reextract = False
    any_validated_arm = False

    metadata_arms = metadata.get("arms") or []

    for cv_arm in curves_validated.get("arms", []):
        name = cv_arm.get("name", "")
        implicit = _compute_implicit_median(cv_arm.get("points") or [])

        # Find matching published arm by name substring
        published = None
        for md_arm in metadata_arms:
            md_name = (md_arm.get("name") or "").lower()
            cv_name = name.lower()
            if not md_name or not cv_name:
                continue
            if md_name == cv_name or md_name in cv_name or cv_name in md_name:
                published = md_arm.get("median_value")
                break

        if published is None:
            # No published median to compare against
            arm_results.append({
                "name": name,
                "implicit_median": implicit,
                "published_median": None,
                "relative_error": None,
                "status": "unverifiable",
            })
            continue

        if implicit is None:
            # Curve never reached 50% but publication says median exists
            # — this is a strong signal of bad coordinates
            arm_results.append({
                "name": name,
                "implicit_median": None,
                "published_median": published,
                "relative_error": None,
                "status": "fail",
            })
            has_hard_fail = True
            continue

        rel_err = abs(implicit - published) / published if published > 0 else float("inf")
        any_validated_arm = True

        if rel_err <= MEDIAN_MATCH_TOLERANCE:
            status = "match"
        elif rel_err <= MEDIAN_MATCH_HARD_FAIL:
            status = "warn"
            needs_reextract = True
        else:
            status = "fail"
            has_hard_fail = True

        arm_results.append({
            "name": name,
            "implicit_median": implicit,
            "published_median": published,
            "relative_error": rel_err,
            "status": status,
        })

    # Determine overall verdict
    if has_hard_fail:
        verdict = "hard_fail"
    elif needs_reextract:
        verdict = "needs_reextract"
    else:
        verdict = "match"

    # Build human-readable summary
    summary_parts = [f"verdict={verdict}"]
    for ar in arm_results:
        if ar["implicit_median"] is None and ar["published_median"] is None:
            summary_parts.append(f"{ar['name']}: no anchor")
        elif ar["implicit_median"] is None:
            summary_parts.append(
                f"{ar['name']}: extracted=NR, published={ar['published_median']:.1f} → fail"
            )
        elif ar["published_median"] is None:
            summary_parts.append(
                f"{ar['name']}: extracted={ar['implicit_median']:.1f}, no published anchor"
            )
        else:
            summary_parts.append(
                f"{ar['name']}: extracted={ar['implicit_median']:.1f}, "
                f"published={ar['published_median']:.1f}, err={ar['relative_error']*100:.1f}% → {ar['status']}"
            )
    summary = "; ".join(summary_parts)

    if not any_validated_arm and not has_hard_fail:
        # No arm could be validated (all NR or unverifiable). Treat as match
        # — we have no basis to fail. Renderer will note tier 3 confidence.
        logger.info(
            f"Median-match validation for source='{source_hint}': "
            f"no validated arms (all unverifiable/NR), treating as match. {summary}"
        )

    return {
        "verdict": verdict,
        "arm_results": arm_results,
        "summary": summary,
    }


def _build_retry_prompt(metadata: dict, validation: dict) -> str:
    """Build a sharper prompt for the curve re-extract, injecting the
    published medians as anchors. The retry prompt is more focused: it
    tells the model exactly which arms had bad medians and where to look
    again.
    """
    anchor_lines = []
    for arm in metadata.get("arms") or []:
        name = arm.get("name") or "?"
        med = arm.get("median_value")
        unit = arm.get("median_unit") or "months"
        if med is not None:
            anchor_lines.append(f"  - {name}: median = {med} {unit}")
        else:
            anchor_lines.append(f"  - {name}: median = Not Reached")

    hr = metadata.get("hr_value")
    if hr is not None:
        anchor_lines.append(f"  - Hazard Ratio (between arms): {hr}")

    # Add per-arm validation feedback to focus attention
    feedback_lines = []
    for ar in validation.get("arm_results") or []:
        if ar["status"] in ("warn", "fail"):
            implicit = ar["implicit_median"]
            published = ar["published_median"]
            if implicit is not None and published is not None:
                feedback_lines.append(
                    f"  - Arm '{ar['name']}': your first attempt placed median at "
                    f"~{implicit:.1f}, but published value is {published:.1f}. "
                    f"Re-read the curve crossing through 50% survival."
                )
            elif implicit is None and published is not None:
                feedback_lines.append(
                    f"  - Arm '{ar['name']}': your first attempt's coordinates never "
                    f"reached 50% survival, but published median is {published:.1f}. "
                    f"The curve does cross 50% — extract more points in that region."
                )

    anchors_block = "\n".join(anchor_lines)
    if feedback_lines:
        anchors_block += "\n\nPER-ARM FEEDBACK FROM FIRST ATTEMPT:\n" + "\n".join(feedback_lines)

    return CURVE_RETRY_PROMPT_TEMPLATE.format(anchors_block=anchors_block)


# ── Tiny helpers ──────────────────────────────────────────────────

def _to_float(v) -> Optional[float]:
    """Coerce to float or None. Rejects NaN, inf."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _safe_str(v, default: str, max_len: int = 100) -> str:
    """Coerce to trimmed string with length cap. Returns default on empty."""
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    return s[:max_len]


def _empty_result(
    error: Optional[str] = None,
    metadata_model: Optional[str] = None,
    nar_model: Optional[str] = None,
    curve_model: Optional[str] = None,
) -> dict:
    """Standard empty-result shape — used for every failure path so the
    caller can rely on a consistent schema."""
    return {
        "title":             "",
        "endpoint":          "",
        "arms":              [],
        "arms_dropped":      [],
        "hr_value":          None,
        "hr_ci_low":         None,
        "hr_ci_high":        None,
        "p_value":           None,
        "p_value_operator":  "",
        "x_label":           "Time (months)",
        "x_min":             0.0,
        "x_max":             36.0,
        "y_label":           "Survival (%)",
        "y_min":             0.0,
        "y_max":             100.0,
        "nar_time_points":   [],
        "nar_time_unit":     "months",
        "nar_arms":          [],
        "nar_dropped":       [],
        # Curve fields (always present so schema is stable)
        "curve_x_axis_unit": "months",
        "curve_x_axis_min":  0.0,
        "curve_x_axis_max":  36.0,
        "curve_y_axis_min":  0.0,
        "curve_y_axis_max":  100.0,
        "curve_arms":        [],
        "curves_dropped":    [],
        "median_validation": None,
        "curve_reextract_attempted": False,
        "extraction_method_metadata": metadata_model or GEMINI_PRIMARY_MODEL,
        "extraction_method_nar":      nar_model      or GEMINI_PRIMARY_MODEL,
        "extraction_method_curve":    curve_model    or GEMINI_PRIMARY_MODEL,
        "error":             error,
    }
