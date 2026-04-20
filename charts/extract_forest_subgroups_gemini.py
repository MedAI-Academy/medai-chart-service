"""
Forest Plot subgroup extraction via Gemini Vision.

Companion to extract_forest_from_pdf.py: takes the 400 DPI PNG crop the
keyword-scoring pipeline already produces and asks Gemini to read off the
subgroup labels, sample sizes, hazard ratios and 95% CI bounds.

The output flows into /charts/forest-plot (Phase 4A reconstruction renderer)
so the slide is built from extracted data, not the original figure crop.
That gives us:
  - Tier 2 confidence (own visualisation of extracted publication data)
  - "Reconstructed from {paper} Figure {N}" footer as liability marker
  - No copyright exposure (we never embed the paper's own figure)

Anti-hallucination posture is strict and explicit:
  1. Gemini is told to OMIT subgroups whose values are unclear, never estimate
  2. We re-validate every row in code: numeric HR/CI, plausible magnitudes,
     ci_low ≤ hr ≤ ci_high (with rounding tolerance)
  3. Dropped rows are returned with a reason so the caller can log diagnostics

If GEMINI_API_KEY is missing, the network call fails, or the model returns
junk, the function never raises — it returns an empty subgroups list with an
error string. The PDF crop pipeline upstream is unaffected; the caller
decides whether to render the slide without structured data.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────
# Two-tier model strategy with automatic fallback.
#
# PRIMARY: gemini-3-flash-preview
#   Bleeding-edge multimodal model (released Dec 2025). Frontier-level
#   reasoning at Flash speed. Free tier available on the Gemini API.
#   BUT: it's a preview model — Google may rename, deprecate, or shut it
#   down with as little as 2 weeks' notice. Without protection, that would
#   silently break the forest-plot pipeline.
#
# FALLBACK: gemini-2.5-flash
#   Stable, generally available, free-tier, fully multimodal. Slightly
#   weaker on complex reasoning but more than capable for forest-plot text
#   extraction (which is OCR-heavy, not reasoning-heavy).
#
# When the primary call fails with a "model unavailable" style error
# (404 not found, deprecated, permission denied, invalid model), we
# automatically retry on the fallback and emit a loud ERROR log so it
# shows up in Railway monitoring. Other failures (network, auth, timeout,
# bad image) do NOT trigger a retry — they're not the model's fault and
# the fallback would just fail too.
#
# To monitor a future stable release: when "gemini-3-flash" or similar
# becomes officially available, set GEMINI_FOREST_MODEL in Railway to
# the new string — no code change required. The result dict's
# `extraction_method` field will always show which model actually served
# the request, so debugging is trivial.
# `or` (instead of os.environ.get's default arg) ensures the code default
# kicks in for BOTH cases:
#   1. env var not set at all
#   2. env var set to empty string (e.g. operator created the Railway var
#      with an empty value field) ← we hit this 2026-04-20, hence the `or`
GEMINI_PRIMARY_MODEL  = os.environ.get("GEMINI_FOREST_MODEL")          or "gemini-3-flash-preview"
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FOREST_FALLBACK_MODEL") or "gemini-2.5-flash"
GEMINI_TIMEOUT_SECONDS = 60

# Substrings (lowercased) in the exception message that mean "this model
# is gone / renamed / unavailable to this API key". Matching any of these
# triggers the fallback retry. Network-level errors (ConnectionError,
# Timeout, etc.) do NOT match any of these on purpose.
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
    "unexpected model name",   # added 2026-04-20: empty/malformed model id
)

# Strict extraction prompt. Two design choices worth noting:
#   - We ask for JSON output via response_mime_type rather than free-text;
#     Gemini honours this reliably.
#   - We tell the model that omission is preferred over estimation. A short
#     subgroup list with high accuracy beats a long list with one wrong HR
#     in a pharma deliverable.
EXTRACTION_PROMPT = """You are a medical data extraction specialist. Extract subgroup analysis data from this Forest Plot figure with maximum accuracy.

CRITICAL ANTI-HALLUCINATION RULES (non-negotiable):
1. Extract ONLY values that are clearly visible and readable in the image.
2. If a numeric value is unclear, partially visible, or ambiguous, OMIT that entire subgroup row. Do NOT estimate.
3. Do NOT invent subgroups that are not visible in the figure.
4. Numeric values (hr, ci_low, ci_high) MUST come from the figure's text labels — typically the right-hand "HR (95% CI)" column. NEVER estimate them from the position of diamonds, squares, or whisker endpoints.
5. Do NOT include narrative strings (e.g. "Favors X", "NS") as numeric values. If a row has no numeric HR, omit it.
6. If the figure contains both "Number of events" and total N (e.g. "161/286"), put the full string in `n` (e.g. "161/286 vs 109/145"). If only N is shown, use just the N (e.g. "286 vs 145"). Be faithful to what is printed.

OUTPUT STRUCTURE (return JSON matching this shape exactly):
{
  "subgroups": [
    {
      "is_header": true,
      "category": "Sex"
    },
    {
      "name": "Female",
      "n": "114 vs 58",
      "hr": 0.61,
      "ci_low": 0.41,
      "ci_high": 0.91,
      "is_overall": false
    },
    {
      "name": "All patients",
      "n": "286 vs 145",
      "hr": 0.66,
      "ci_low": 0.52,
      "ci_high": 0.85,
      "is_overall": true
    }
  ],
  "favours_left": "Favors experimental",
  "favours_right": "Favors control",
  "title": "Subgroup Analysis of Overall Survival",
  "extraction_notes": "Optional brief notes if any rows were dropped due to unclear values"
}

ROW RULES:
- For category separator rows (e.g. "Sex", "Age", "ECOG performance status") set `is_header: true` and put the label in `category`. Leave hr/ci_low/ci_high empty.
- For the All Patients / Overall / Total row, set `is_overall: true`.
- For data rows, fill `name`, `n`, `hr`, `ci_low`, `ci_high`.
- `favours_left` / `favours_right`: read the arrow labels at the bottom of the plot (e.g. "Favors Ven+Aza" / "Favors Placebo"). If not present, use "experimental better" / "control better".
- `title`: the figure caption or heading. If unclear, use empty string.

If the image does NOT contain a Forest Plot, return {"subgroups": [], "extraction_notes": "No forest plot detected in image"}.
"""


def extract_forest_subgroups_gemini(
    image_base64: str,
    source_hint: Optional[str] = None,
) -> dict:
    """Extract structured subgroup data from a Forest Plot PNG via Gemini Vision.

    Uses a two-tier model strategy: GEMINI_PRIMARY_MODEL is tried first;
    if it returns a "model unavailable" style error (404, deprecated,
    not found, …), we automatically retry on GEMINI_FALLBACK_MODEL and
    emit a loud ERROR log so the deprecation surfaces in monitoring.
    Other failure modes (network, auth, timeout, bad image, malformed
    response) do NOT trigger a retry — they're not the model's fault.

    Args:
        image_base64: PNG (or JPEG) of the forest plot, base64-encoded.
                      Optionally with "data:image/png;base64," prefix.
        source_hint:  Optional study/paper name for logging context. Not sent
                      to the model — the model must read the image, not rely
                      on training memory.

    Returns:
        dict with these keys:
          - subgroups:        list of normalised dicts (see _validate_rows)
          - favours_left:     str   ("experimental better" if not detected)
          - favours_right:    str   ("control better" if not detected)
          - title:            str   (figure title from image, or "")
          - dropped_rows:     list[{"raw":..., "reason":...}] for diagnostics
          - extraction_method: str  (model id ACTUALLY used — primary or fallback)
          - error:            str | None
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _empty_result(
            error="GEMINI_API_KEY environment variable not set",
            model_used=GEMINI_PRIMARY_MODEL,
        )

    # Lazy import so the chart-service still boots if the package is missing.
    try:
        import google.generativeai as genai
    except ImportError as e:
        return _empty_result(
            error=f"google-generativeai not installed: {e}",
            model_used=GEMINI_PRIMARY_MODEL,
        )

    # Strip data-URL prefix if present
    if image_base64.startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]

    try:
        png_bytes = base64.b64decode(image_base64)
    except Exception as e:
        return _empty_result(
            error=f"Invalid base64 image: {e}",
            model_used=GEMINI_PRIMARY_MODEL,
        )

    if len(png_bytes) < 1000:
        return _empty_result(
            error=f"Image too small ({len(png_bytes)} bytes) — likely empty or corrupted",
            model_used=GEMINI_PRIMARY_MODEL,
        )

    genai.configure(api_key=api_key)

    # Note: we do NOT include source_hint in the prompt to the model —
    # we want it to read the image, not pattern-match against memory of
    # famous trials. We log the hint server-side only.
    logger.info(
        f"Forest extraction starting for source_hint='{source_hint}' "
        f"primary={GEMINI_PRIMARY_MODEL} fallback={GEMINI_FALLBACK_MODEL}"
    )

    # ---- Tier 1: try the primary (preview) model ----------------------
    model_used = GEMINI_PRIMARY_MODEL
    try:
        data = _call_gemini(genai, GEMINI_PRIMARY_MODEL, png_bytes)
    except Exception as primary_exc:
        if (
            _is_model_unavailable_error(primary_exc)
            and GEMINI_FALLBACK_MODEL
            and GEMINI_FALLBACK_MODEL != GEMINI_PRIMARY_MODEL
        ):
            # Loud log — Railway/monitoring should pick this up so the operator
            # knows to investigate the primary model's status (renamed?
            # deprecated? rate-limited?) and update GEMINI_FOREST_MODEL.
            logger.error(
                f"⚠️  PRIMARY MODEL UNAVAILABLE: '{GEMINI_PRIMARY_MODEL}' failed "
                f"with: {primary_exc}. Falling back to '{GEMINI_FALLBACK_MODEL}'. "
                f"ACTION: investigate primary model status and update "
                f"GEMINI_FOREST_MODEL env var if it was renamed/deprecated."
            )
            try:
                data = _call_gemini(genai, GEMINI_FALLBACK_MODEL, png_bytes)
                model_used = GEMINI_FALLBACK_MODEL
            except Exception as fallback_exc:
                logger.error(
                    f"BOTH MODELS FAILED — primary '{GEMINI_PRIMARY_MODEL}': "
                    f"{primary_exc}; fallback '{GEMINI_FALLBACK_MODEL}': {fallback_exc}"
                )
                return _empty_result(
                    error=(
                        f"Both Gemini models failed. "
                        f"Primary ({GEMINI_PRIMARY_MODEL}): {primary_exc}. "
                        f"Fallback ({GEMINI_FALLBACK_MODEL}): {fallback_exc}"
                    ),
                    model_used=model_used,
                )
        else:
            # Non-model-related failure — no retry, just report.
            logger.warning(
                f"Gemini call to '{GEMINI_PRIMARY_MODEL}' failed for "
                f"source='{source_hint}' (no fallback — not a model availability error): {primary_exc}"
            )
            return _empty_result(
                error=f"Gemini call to {GEMINI_PRIMARY_MODEL} failed: {primary_exc}",
                model_used=model_used,
            )

    # ---- Validate and normalise the response --------------------------
    if not isinstance(data, dict):
        return _empty_result(
            error="Gemini response is not a JSON object",
            model_used=model_used,
        )

    cleaned_rows, dropped = _validate_rows(data.get("subgroups") or [])

    result = {
        "subgroups":         cleaned_rows,
        "favours_left":      _safe_str(data.get("favours_left"), "experimental better", max_len=40),
        "favours_right":     _safe_str(data.get("favours_right"), "control better", max_len=40),
        "title":             _safe_str(data.get("title"), "", max_len=200),
        "dropped_rows":      dropped,
        "extraction_method": model_used,  # ← ACTUAL model used, not the default
        "error":             None,
    }

    data_row_count = sum(1 for sg in cleaned_rows if not sg.get("is_header"))
    logger.info(
        f"Forest extraction done for source='{source_hint}' via {model_used}: "
        f"{data_row_count} data row(s), {len(dropped)} dropped"
    )
    return result


def _call_gemini(genai_module, model_id: str, png_bytes: bytes) -> dict:
    """Single Gemini Vision call. Returns parsed JSON dict or raises.

    Isolated from the main function so the fallback chain can call it
    twice (once per model) without code duplication. Raises on any
    failure — the caller decides whether to retry.
    """
    model = genai_module.GenerativeModel(model_id)
    response = model.generate_content(
        [
            EXTRACTION_PROMPT,
            {"mime_type": "image/png", "data": png_bytes},
        ],
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.0,  # deterministic — same image → same output
        },
        request_options={"timeout": GEMINI_TIMEOUT_SECONDS},
    )

    raw_text = (getattr(response, "text", None) or "").strip()
    if not raw_text:
        raise ValueError(f"Empty response from {model_id}")

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        # Last-ditch: strip ```json … ``` code-fence wrappers some models add
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
    """Detect 'model is gone / renamed / unavailable' style errors that
    should trigger a fallback retry. Network/auth/timeout errors do NOT
    match — those would just fail again on the fallback model too."""
    msg = str(exc).lower()
    return any(indicator in msg for indicator in _MODEL_UNAVAILABLE_INDICATORS)


def _validate_rows(raw_subgroups: list) -> tuple[list, list]:
    """Validate AI output, normalise field names, drop bad rows.

    Returns (cleaned_rows, dropped_with_reasons).

    A "good" data row needs:
      - name         (non-empty after strip)
      - hr           (positive finite float, ≤ 100)
      - ci_low       (positive finite float, ≤ 100)
      - ci_high      (positive finite float, ≤ 100)
      - ci_low ≤ hr ≤ ci_high  (with 5% rounding tolerance both sides)

    A "good" header row needs:
      - is_header == True
      - category (or name/group) non-empty
    """
    cleaned: list[dict] = []
    dropped: list[dict] = []

    if not isinstance(raw_subgroups, list):
        return cleaned, [{"raw": raw_subgroups, "reason": "subgroups field is not a list"}]

    for sg in raw_subgroups:
        if not isinstance(sg, dict):
            dropped.append({"raw": sg, "reason": "row is not an object"})
            continue

        # Header rows — keep if there's a category label
        if sg.get("is_header"):
            cat = _safe_str(
                sg.get("category") or sg.get("group") or sg.get("name"),
                "",
                max_len=80,
            )
            if cat:
                cleaned.append({"is_header": True, "category": cat})
            else:
                dropped.append({"raw": sg, "reason": "header without category label"})
            continue

        # Data rows — strict numeric validation
        hr = _to_float(sg.get("hr"))
        ci_low = _to_float(sg.get("ci_low") or sg.get("ciLow") or sg.get("ci_lower"))
        ci_high = _to_float(sg.get("ci_high") or sg.get("ciHigh") or sg.get("ci_upper"))

        if hr is None:
            dropped.append({"raw": sg, "reason": "hr missing or non-numeric"})
            continue
        if ci_low is None or ci_high is None:
            dropped.append({"raw": sg, "reason": "ci_low or ci_high missing or non-numeric"})
            continue
        if not (0 < hr <= 100 and 0 < ci_low <= 100 and 0 < ci_high <= 100):
            dropped.append({"raw": sg, "reason": f"hr/ci out of plausible range (hr={hr}, ci=[{ci_low},{ci_high}])"})
            continue
        if ci_low > ci_high:
            # Some figures print "0.85–0.52" by mistake or the model swapped them.
            # Auto-correct rather than drop — the data is recoverable.
            ci_low, ci_high = ci_high, ci_low
        # 5% rounding tolerance: hr printed as 0.66, CI as (0.52–0.85) — published
        # values are usually rounded so a strict ci_low ≤ hr ≤ ci_high check
        # occasionally fails by 0.005-0.01.
        if not (ci_low * 0.95 <= hr <= ci_high * 1.05):
            dropped.append({
                "raw": sg,
                "reason": f"hr {hr} outside CI [{ci_low}, {ci_high}] beyond rounding tolerance",
            })
            continue

        name = _safe_str(sg.get("name"), "", max_len=80)
        if not name:
            dropped.append({"raw": sg, "reason": "data row missing name"})
            continue

        cleaned.append({
            "name":       name,
            "n":          _safe_str(sg.get("n"), "", max_len=40),
            "hr":         hr,
            "ci_low":     ci_low,
            "ci_high":    ci_high,
            "is_overall": bool(sg.get("is_overall", False)),
        })

    return cleaned, dropped


def _to_float(v) -> Optional[float]:
    """Coerce a value to float. Returns None on failure or non-finite."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # Reject NaN, ±inf
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


def _empty_result(error: Optional[str] = None, model_used: Optional[str] = None) -> dict:
    """Standard empty-result shape — used for every failure path so the
    caller can rely on the schema regardless of which error tripped.

    `model_used` defaults to the primary model id; pass the actual model
    used when the failure occurred during/after a fallback attempt so
    debugging can distinguish "primary failed before retry" from
    "fallback also failed"."""
    return {
        "subgroups":         [],
        "favours_left":      "experimental better",
        "favours_right":     "control better",
        "title":             "",
        "dropped_rows":      [],
        "extraction_method": model_used or GEMINI_PRIMARY_MODEL,
        "error":             error,
    }
