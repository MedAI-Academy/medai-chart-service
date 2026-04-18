"""
Kaplan-Meier extraction from full PDF publications.

Pipeline:
  1. Iterate all pages, score each by KM-figure keyword density.
  2. Pick the best page.
  3. Find the largest figure bounding box via PyMuPDF drawings (vector
     shapes) with a fallback to the whole page.
  4. Render that bbox at 400 DPI as PNG.
  5. Feed the crop into extract_km (HSV pixel extraction).
"""

from __future__ import annotations

import base64
from typing import Optional

import fitz  # PyMuPDF

import cv2
import numpy as np

from charts.extract_km import extract_km, detect_plot_bounds


# Keywords used to score pages. Weighted: the ones that almost never
# appear outside KM figures score higher.
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
        # Accept lines (w or h == 0). Drop zero-sized points only.
        if r.width < 1 and r.height < 1:
            continue
        # Skip drawings that span (almost) the whole page
        if r.width > page_rect.width * 0.9 and r.height > page_rect.height * 0.9:
            continue
        # Skip zero-width/height lines that are page borders
        if r.width > page_rect.width * 0.9 or r.height > page_rect.height * 0.9:
            continue
        rects.append(r)

    if not rects:
        return None

    # Cluster rects: two rects belong to the same cluster if they are
    # within a small gap of each other. Use a fixed 12 pt gap — KM
    # figures consist of tightly-spaced line segments.
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

    # Merge overlapping clusters (2-pass union)
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

    # Pick the cluster whose union bbox has the largest area AND
    # contains at least 15 drawings (smaller threshold helps figures
    # drawn as a handful of long polylines).
    best: Optional[fitz.Rect] = None
    best_area = 0.0
    for cl in clusters:
        if len(cl) < 15:
            continue
        union = fitz.Rect(cl[0])
        for r in cl[1:]:
            union.include_rect(r)
        area = union.width * union.height
        # Figure must occupy at least 8 % of the page area but not > 85 %.
        page_area = page_rect.width * page_rect.height
        if area < page_area * 0.08 or area > page_area * 0.85:
            continue
        if area > best_area:
            best_area = area
            best = union

    if best is None:
        return None

    # Pad the bbox slightly so tick labels and axis titles that live
    # just outside the drawing cluster are included in the crop.
    pad = 12  # points
    padded = fitz.Rect(
        max(page.rect.x0, best.x0 - pad),
        max(page.rect.y0, best.y0 - pad),
        min(page.rect.x1, best.x1 + pad),
        min(page.rect.y1, best.y1 + pad),
    )
    return padded


def _render_page_crop(page, bbox: Optional[fitz.Rect], dpi: int = 400) -> bytes:
    """Render the page (or a bbox) to PNG bytes at the given DPI."""
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)
    if bbox is not None:
        pix = page.get_pixmap(matrix=mat, clip=bbox, alpha=False)
    else:
        pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def extract_km_from_pdf(
    pdf_base64: str,
    study_name: Optional[str] = None,
    min_score: int = 4,
) -> dict:
    """Detect and extract the KM figure from a full PDF publication.

    Returns the same shape as extract_km() plus:
      - page_number:           1-indexed page where the figure was found
      - figure_crop_base64:    the 400 DPI crop that was sent to extract_km
      - pdf_keyword_score:     the keyword score of the selected page
    """
    if pdf_base64.startswith("data:"):
        pdf_base64 = pdf_base64.split(",", 1)[1]
    pdf_bytes = base64.b64decode(pdf_base64)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Invalid PDF: {e}")

    # ---- 1. Score all pages ----
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

    # ---- 2. Figure bbox via vector drawings ----
    bbox = _find_figure_bbox(page)

    # ---- 3. Render at 400 DPI ----
    png_bytes = _render_page_crop(page, bbox, dpi=400)
    png_b64 = base64.b64encode(png_bytes).decode("ascii")

    doc.close()

    # ---- 4. Auto-detect plot bounds ON THE NEW CROP ----
    # STUDY_REFERENCE plot_bounds apply to the canonical PNG crops, not
    # to freshly-rendered PDF pages. Detect on the rendered image and
    # pass explicitly so the reference lookup for bounds is bypassed
    # while name/colors/NaR/axis-ranges from ref are still used.
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    detected_bounds = detect_plot_bounds(img)

    # ---- 5. Delegate to existing extract_km ----
    try:
        result = extract_km(
            image_base64=png_b64,
            study_name=study_name,
            plot_bounds=detected_bounds,
        )
    except Exception as e:
        # Even if extraction fails, still return the crop so the caller
        # can fall back to manual tooling.
        return {
            "error": f"extract_km failed on rendered crop: {e}",
            "page_number": best_idx + 1,
            "figure_crop_base64": png_b64,
            "pdf_keyword_score": best_score,
            "page_scores": scores,
        }

    result["page_number"] = best_idx + 1
    result["figure_crop_base64"] = png_b64
    result["pdf_keyword_score"] = best_score
    result["page_scores"] = scores
    return result
