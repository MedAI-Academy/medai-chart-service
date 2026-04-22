"""
Kaplan-Meier extraction from full PDF publications via Gemini Vision.

This is the Phase 5 replacement for the OpenCV pixel-tracing pipeline that
shipped before. The new pipeline:

  1. Scan all pages, score each by KM-figure keyword density (UNCHANGED
     from the previous pipeline — keyword scoring works well).
  2. Pick the best page.
  3. Find the largest figure bounding box via PyMuPDF drawings (UNCHANGED).
  4. Render that bbox at 400 DPI as PNG (UNCHANGED).
  5. NEW: send the crop to Gemini Vision (extract_km_vision_gemini, three
     parallel passes: metadata, NaR, curve coordinates).
  6. NEW: validate the curve coordinates against published medians.
  7. NEW: render an own NEJM-style PNG from the validated coordinates
     (km_render_nejm). The rendered PNG is what the slide embeds —
     never the original figure crop, which avoids copyright issues and
     gives us full control over visual quality.

Confidence tier mapping:
    Tier 1: not used (would require comparing to known reference data —
            we don't maintain a STUDY_REFERENCE table any more)
    Tier 2: vision extraction succeeded AND median validation matched
            (verdict == "match" within ±10% tolerance)
    Tier 3: vision extraction succeeded but median validation flagged
            warn (10-25% deviation, retry already applied) OR no
            published anchor available
    None  : extraction failed entirely (hard_fail or worse) — no PNG
            returned, error message set so the frontend can show a
            user-facing message.

Backwards compatibility: response shape is intentionally a superset of
the previous shape. The frontend only consumed `png_base64`, `page_number`,
`error`, and `confidence_tier`, all of which are still here. New fields
(`vision_data`, `median_validation`, `extraction_methods`) are optional —
the frontend will eventually use them for richer UI but doesn't need to
yet.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import fitz  # PyMuPDF

from charts.extract_km_vision_gemini import extract_km_vision
from charts.km_render_nejm import render_km_from_vision

logger = logging.getLogger(__name__)


# Same keyword set as before — KM figures consistently use these terms.
KM_KEYWORDS = {
    "no. at risk": 4,
    "number at risk": 4,
    "kaplan-meier": 4,
    "hazard ratio": 3,
    "overall survival": 3,
    "progression-free survival": 3,
    "progression free survival": 3,
    "median follow-up": 2,
    "probability of": 2,
    "survival": 1,
    "months": 1,
    "95% ci": 1,
}


def _score_page(text: str) -> int:
    t = text.lower()
    score = 0
    for kw, w in KM_KEYWORDS.items():
        if kw in t:
            score += w
    return score


def _find_figure_bbox(page) -> Optional[fitz.Rect]:
    """Return the union bbox of the largest drawing cluster on the page,
    or None if drawings cannot reliably identify the figure.

    KM figures are typically composed of many line/rect drawings (axes,
    curve segments, tick marks). We cluster drawings by proximity and
    return the largest cluster's union bbox.

    UNCHANGED from the old pipeline — this part works well.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    rects: list[fitz.Rect] = []
    page_rect = page.rect
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        if r.width < 1 and r.height < 1:
            continue
        if r.width > page_rect.width * 0.9 and r.height > page_rect.height * 0.9:
            continue
        if r.width > page_rect.width * 0.9 or r.height > page_rect.height * 0.9:
            continue
        rects.append(r)

    if not rects:
        return None

    # Cluster rects within a 12pt gap
    gap = 12.0
    clusters: list[list[fitz.Rect]] = []
    for r in rects:
        pad = fitz.Rect(r.x0 - gap, r.y0 - gap, r.x1 + gap, r.y1 + gap)
        placed = False
        for cl in clusters:
            if any(pad.intersects(x) for x in cl):
                cl.append(r)
                placed = True
                break
        if not placed:
            clusters.append([r])

    # Merge overlapping clusters
    for _ in range(2):
        merged: list[list[fitz.Rect]] = []
        for cl in clusters:
            union = fitz.Rect(cl[0])
            for r in cl[1:]:
                union.include_rect(r)
            placed = False
            for m in merged:
                m_union = fitz.Rect(m[0])
                for r in m[1:]:
                    m_union.include_rect(r)
                if union.intersects(m_union):
                    m.extend(cl)
                    placed = True
                    break
            if not placed:
                merged.append(cl)
        clusters = merged

    # Pick the largest qualifying cluster
    best: Optional[fitz.Rect] = None
    best_area = 0.0
    for cl in clusters:
        if len(cl) < 15:
            continue
        union = fitz.Rect(cl[0])
        for r in cl[1:]:
            union.include_rect(r)
        area = union.width * union.height
        page_area = page_rect.width * page_rect.height
        if area < page_area * 0.08 or area > page_area * 0.85:
            continue
        if area > best_area:
            best_area = area
            best = union

    if best is None:
        return None

    # Pad slightly to include axis labels
    pad = 12
    return fitz.Rect(
        max(page.rect.x0, best.x0 - pad),
        max(page.rect.y0, best.y0 - pad),
        min(page.rect.x1, best.x1 + pad),
        min(page.rect.y1, best.y1 + pad),
    )


def _render_page_crop(page, bbox: Optional[fitz.Rect], dpi: int = 400) -> bytes:
    """Render the page (or a bbox) to PNG bytes at the given DPI."""
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)
    if bbox is not None:
        pix = page.get_pixmap(matrix=mat, clip=bbox, alpha=False)
    else:
        pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def _compute_confidence_tier(vision_data: dict) -> int:
    """Map the vision_data state to the integer confidence tier.

    Tier 2 = matched against published median (best we can offer
             without reference IPD)
    Tier 3 = extracted but validation soft-failed or no anchor
    """
    mv = vision_data.get("median_validation") or {}
    verdict = mv.get("verdict")
    if verdict == "match":
        # Did we actually validate against any published anchor, or did
        # all arms come back "unverifiable" (no published median)?
        any_validated = any(
            (ar.get("status") == "match")
            for ar in (mv.get("arm_results") or [])
        )
        return 2 if any_validated else 3
    elif verdict == "needs_reextract":
        # Soft warn — kept the extraction but median didn't fully match
        return 3
    elif verdict == "hard_fail":
        # Should not be called in the success path — we set png to None
        # before this. Returning 3 as defensive default.
        return 3
    else:
        return 3


def extract_km_from_pdf(
    pdf_base64: str,
    study_name: Optional[str] = None,
    min_score: int = 4,
) -> dict:
    """Detect and extract the KM figure from a full PDF publication, then
    re-render via Gemini Vision + Vision-renderer.

    Args:
        pdf_base64: full PDF as base64 string (with or without data: prefix)
        study_name: optional paper/study name for the validation footer
                    on the rendered PNG (e.g. "DiNardo et al., NEJM 2020").
                    Also used as `source_hint` for vision extraction logging.
        min_score: minimum keyword score for accepting a page as KM-bearing.

    Returns:
        dict with these keys (always present, even on failure):
          - png_base64:           base64 of the rendered NEJM-style PNG.
                                  EMPTY STRING if extraction failed; check
                                  `error` for the reason.
          - page_number:          1-indexed page where the figure was found,
                                  or None if no KM page detected.
          - figure_crop_base64:   base64 of the source crop sent to Vision
                                  (useful for debugging — frontend doesn't
                                  embed this).
          - pdf_keyword_score:    keyword score of the selected page.
          - page_scores:          list of keyword scores per page.
          - confidence_tier:      2 (matched against published median),
                                  3 (extracted but soft warn / no anchor),
                                  or None (extraction failed).
          - vision_data:          full dict from extract_km_vision (metadata,
                                  NaR, curve_arms, etc.). Useful for the
                                  frontend if it wants to display NaR-only
                                  fallback when the renderer returns no PNG.
          - median_validation:    convenience copy of vision_data.median_validation
                                  for frontend access without traversing.
          - extraction_methods:   convenience dict of model IDs used per pass.
          - error:                None on success, or a string describing
                                  why the slide cannot be built. Frontend
                                  should display this to the user.

    Raises:
        ValueError on invalid PDF or no KM page detected (caught by the
        FastAPI endpoint and returned as 400).
    """
    # ---- 1. Decode + open PDF ----
    if pdf_base64.startswith("data:"):
        pdf_base64 = pdf_base64.split(",", 1)[1]
    pdf_bytes = base64.b64decode(pdf_base64)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Invalid PDF: {e}")

    # ---- 2. Score all pages ----
    best_idx = -1
    best_score = 0
    scores: list[int] = []
    for i, page in enumerate(doc):
        try:
            text = page.get_text("text")
        except Exception:
            text = ""
        s = _score_page(text)
        scores.append(s)
        if s > best_score:
            best_score = s
            best_idx = i

    if best_idx < 0 or best_score < min_score:
        doc.close()
        raise ValueError(
            f"No KM figure detected (best keyword score = {best_score}, "
            f"threshold = {min_score})"
        )

    page = doc[best_idx]

    # ---- 3. Figure bbox via vector drawings ----
    bbox = _find_figure_bbox(page)

    # ---- 4. Render at 400 DPI ----
    png_bytes = _render_page_crop(page, bbox, dpi=400)
    crop_b64 = base64.b64encode(png_bytes).decode("ascii")

    doc.close()

    logger.info(
        f"KM page detected: page {best_idx + 1} (score {best_score}), "
        f"crop {len(png_bytes)} bytes — sending to Vision"
    )

    # ---- 5. Vision extraction (3 parallel passes) ----
    vision_data = extract_km_vision(
        image_base64=crop_b64,
        source_hint=study_name or f"PDF page {best_idx + 1}",
    )

    # Common return scaffolding — populated below
    response = {
        "png_base64":         "",
        "page_number":        best_idx + 1,
        "figure_crop_base64": crop_b64,
        "pdf_keyword_score":  best_score,
        "page_scores":        scores,
        "confidence_tier":    None,
        "vision_data":        vision_data,
        "median_validation":  vision_data.get("median_validation"),
        "extraction_methods": {
            "metadata": vision_data.get("extraction_method_metadata"),
            "nar":      vision_data.get("extraction_method_nar"),
            "curve":    vision_data.get("extraction_method_curve"),
        },
        "error":              None,
    }

    # ---- 5a. Extraction-level errors ----
    # If vision extraction set its own error (hard_fail or model-down), pass
    # it through verbatim — the wrapper formats user-friendly messages.
    if vision_data.get("error"):
        logger.warning(
            f"Vision extraction returned error for page {best_idx + 1}: "
            f"{vision_data['error']}"
        )
        response["error"] = vision_data["error"]
        return response

    # ---- 5b. No usable curve arms ----
    # Vision succeeded but produced no curve coordinates we could validate.
    # This happens if the figure is too low-resolution or the model
    # hallucinated and we dropped everything in validation. Don't render
    # a misleading slide — return the metadata so the frontend can decide.
    if not vision_data.get("curve_arms"):
        msg = (
            "Vision extraction did not produce usable curve coordinates. "
            "Title and metadata may still be available, but the curve itself "
            "could not be reconstructed. Try a higher-resolution PDF, or "
            "skip this KM slide."
        )
        logger.warning(f"Page {best_idx + 1}: {msg}")
        response["error"] = msg
        return response

    # ---- 6. Render the NEJM-style PNG from the extracted coordinates ----
    try:
        rendered_png = render_km_from_vision(
            vision_data,
            source_name=study_name,
        )
    except Exception as e:
        logger.exception(f"Render failed for page {best_idx + 1}: {e}")
        response["error"] = f"Render failed: {e}"
        return response

    if rendered_png is None:
        # render_km_from_vision returns None if curve_arms is empty.
        # Should not happen here (we checked above) but defensive.
        response["error"] = "Renderer returned no PNG (no curve data)"
        return response

    response["png_base64"]      = base64.b64encode(rendered_png).decode("ascii")
    response["confidence_tier"] = _compute_confidence_tier(vision_data)

    logger.info(
        f"KM extraction complete for page {best_idx + 1}: "
        f"{len(rendered_png)} bytes rendered, tier {response['confidence_tier']}, "
        f"verdict={(response['median_validation'] or {}).get('verdict', 'n/a')}"
    )
    return response
