"""
Forest Plot extraction from full PDF publications.

Pipeline:
  1. Iterate all pages, score each by forest-plot keyword density.
  2. Pick the best page (distinct from KM pages via KM-penalty).
  3. Find the largest figure bounding box via PyMuPDF drawings.
  4. Render that bbox at 400 DPI as PNG.
  5. Return the crop — no pixel tracing needed.

Unlike KM extraction, we do NOT reconstruct the plot. The original
figure already contains all the necessary information (subgroup labels,
N-values, HR values, diamonds, whiskers). The crop is good enough for
embedding directly into a forest_pic_light slide.
"""

from __future__ import annotations

import base64
import re
from typing import Optional

import fitz  # PyMuPDF


# Phase 3B fix — strong positive signal.
# Matches figure captions like "Figure 3. Subgroup Analysis of Overall Survival"
# or "Figure 4. Forest Plot of ...". Limited to 120 chars after the period so we
# don't accidentally match prose that mentions "figure" earlier in the paragraph.
FIGURE_CAPTION_PAT = re.compile(
    r'figure\s+\d+\.\s*[^\n.]{0,120}?(subgroup|forest)',
    re.IGNORECASE,
)


# Keywords used to score pages. Weighted: the ones that almost never
# appear outside forest plots score higher.
FOREST_KEYWORDS = {
    # Forest-plot-specific (high weight)
    "subgroup analysis": 5,
    "forest plot": 5,
    "favours": 4,
    "favors": 4,
    "subgroups": 3,
    "subgroup": 2,
    # Common to forest plots + some KM tables
    "hazard ratio": 2,
    "95% ci": 2,
    "95% confidence": 2,
    "odds ratio": 2,
    "relative risk": 2,
    # Low-weight helpers
    "favor experimental": 2,
    "favor control": 2,
    "hr (95%": 2,
    "or (95%": 2,
}

# Keywords that indicate a page is PRIMARILY a KM page, not a forest plot.
# We subtract these from the forest score to avoid false positives.
KM_INDICATORS = {
    "no. at risk": 3,
    "number at risk": 3,
    "kaplan-meier": 3,
    "probability of survival": 2,
    "overall survival": 1,  # weak signal — forest plots also mention OS
    "progression-free survival": 1,
}


def _score_page(text: str, drawing_count: int = 0) -> dict:
    """Return detailed page-scoring breakdown.

    Phase 3B change — returns a dict instead of tuple so the caller can log and
    tie-break on individual components. Two strong new signals:

      - figure_caption_bonus: +15 when a "Figure N. ... (Subgroup|Forest)" caption
        is detected. Near-perfect binary: only the real forest-plot page matches
        in typical journal articles.
      - drawing_bonus: +3 for pages with ≥30 vector drawings, +6 for ≥100.
        Forest plots are drawing-dense (diamonds, whiskers, ticks per subgroup).
        Prose pages have 0-5 drawings.

    These two signals are what separates a real figure page from a prose page
    that happens to mention "subgroup" and "95% CI" in running text.
    """
    t = text.lower()
    forest_score = 0
    for kw, w in FOREST_KEYWORDS.items():
        if kw in t:
            forest_score += w
    km_penalty = 0
    for kw, w in KM_INDICATORS.items():
        if kw in t:
            km_penalty += w
    # Figure caption match (strong +15)
    cap_bonus = 15 if FIGURE_CAPTION_PAT.search(t) else 0
    # Graduated drawing-count bonus (weak but useful for tie-breaking)
    if drawing_count >= 100:
        drawing_bonus = 6
    elif drawing_count >= 30:
        drawing_bonus = 3
    else:
        drawing_bonus = 0
    effective = forest_score - km_penalty + cap_bonus + drawing_bonus
    return {
        "forest": forest_score,
        "km_penalty": km_penalty,
        "figure_caption_bonus": cap_bonus,
        "drawing_bonus": drawing_bonus,
        "drawing_count": drawing_count,
        "effective": effective,
    }


def _find_figure_bbox(page) -> Optional[fitz.Rect]:
    """Return the union bbox of the largest drawing cluster on the page,
    or None if drawings cannot reliably identify the figure.

    Forest plots consist of many line/rect drawings (horizontal whiskers,
    diamonds, reference line, axis ticks). We cluster drawings by proximity
    and return the largest cluster's union bbox.

    This is identical to the KM version — forest plots have the same
    topology (many clustered small vector shapes).
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

    # Cluster rects: two rects belong to the same cluster if they are
    # within a small gap. Forest plots have wider horizontal spread than
    # KM curves (the subgroup table sits to the left of the plot), so
    # use a larger gap.
    gap = 18.0
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
    # contains at least 10 drawings. Forest plots can be sparser than
    # KM figures — diamond + whiskers for each subgroup row.
    best: Optional[fitz.Rect] = None
    best_area = 0.0
    for cl in clusters:
        if len(cl) < 10:
            continue
        union = fitz.Rect(cl[0])
        for r in cl[1:]:
            union.include_rect(r)
        area = union.width * union.height
        # Figure must occupy at least 8 % of the page area but not > 90 %.
        # Forest plots tend to be larger (fill most of the page with
        # subgroup table + plot) so upper bound is 90 % not 85 %.
        page_area = page_rect.width * page_rect.height
        if area < page_area * 0.08 or area > page_area * 0.90:
            continue
        if area > best_area:
            best_area = area
            best = union

    if best is None:
        return None

    # Pad the bbox so subgroup labels on the left (which are text, not
    # drawings) get included in the crop. Use a larger pad than KM
    # because forest plots have wide-spread text labels.
    pad_x = 80  # points — wide enough to include "Favours Experimental" labels
    pad_y = 20  # points
    padded = fitz.Rect(
        max(page.rect.x0, best.x0 - pad_x),
        max(page.rect.y0, best.y0 - pad_y),
        min(page.rect.x1, best.x1 + pad_x),
        min(page.rect.y1, best.y1 + pad_y),
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


def extract_forest_from_pdf(
    pdf_base64: str,
    study_name: Optional[str] = None,
    min_score: int = 8,
) -> dict:
    """Detect and extract the Forest Plot figure from a full PDF publication.

    Unlike KM extraction, this does NOT reconstruct the plot — it simply
    returns the cropped PNG which already contains all subgroup labels,
    N-values, HR values, diamonds, and whiskers.

    Phase 3B — stricter detection. Default min_score raised from 4 to 8 so prose
    pages that merely mention "subgroup" + "95% CI" in running text do NOT qualify.
    A real forest-plot page will score ≥20 thanks to the figure_caption_bonus and
    drawing_bonus additions.

    Returns:
      - png_base64:            the 400 DPI crop (same as figure_crop_base64)
      - figure_crop_base64:    the 400 DPI crop
      - page_number:           1-indexed page where the figure was found
      - pdf_keyword_score:     the effective score of the selected page
      - page_scores:           list of per-page scoring detail dicts (see _score_page)
      - confidence_tier:       3 (always — we cannot validate forest data
                                  against text in the same way we do for KM
                                  medians, so we always report tier 3)
      - source:                file source marker (filled in by caller)
      - bbox_found:            whether a figure bounding box could be isolated
    """
    if pdf_base64.startswith("data:"):
        pdf_base64 = pdf_base64.split(",", 1)[1]
    pdf_bytes = base64.b64decode(pdf_base64)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Invalid PDF: {e}")

    # ---- 1. Score all pages (Phase 3B: now with drawing count + caption bonus) ----
    scores: list[dict] = []
    for i, page in enumerate(doc):
        try:
            text = page.get_text("text")
        except Exception:
            text = ""
        try:
            drawing_count = len(page.get_drawings())
        except Exception:
            drawing_count = 0
        detail = _score_page(text, drawing_count)
        detail["page"] = i + 1
        scores.append(detail)

    # ---- 2. Pick winner with explicit tie-breaker ----
    # Primary:   highest effective score
    # Tiebreak1: page with a figure_caption_bonus wins over one without
    # Tiebreak2: more drawings wins (real figure > prose)
    # Tiebreak3: earlier page wins (stable)
    ranked = sorted(
        scores,
        key=lambda s: (-s["effective"], -s["figure_caption_bonus"], -s["drawing_count"], s["page"]),
    )
    winner = ranked[0] if ranked else None

    if winner is None or winner["effective"] < min_score:
        doc.close()
        best_eff = winner["effective"] if winner else 0
        raise ValueError(
            f"No Forest Plot figure detected "
            f"(best effective score = {best_eff}, threshold = {min_score}). "
            f"Frontend will skip the Forest Plot slide."
        )

    best_idx = winner["page"] - 1
    best_score = winner["effective"]
    page = doc[best_idx]

    # ---- 2. Figure bbox via vector drawings ----
    bbox = _find_figure_bbox(page)

    # ---- 3. Render at 400 DPI ----
    png_bytes = _render_page_crop(page, bbox, dpi=400)
    png_b64 = base64.b64encode(png_bytes).decode("ascii")

    doc.close()

    # ---- 4. Return result — NO extract_km-style pixel tracing ----
    # The crop itself IS the output. The Forest Plot figure in the PDF
    # already has all labels, diamonds, and numbers rendered at publication
    # quality. We hand it straight to the PPTX renderer via figurePng.
    return {
        "png_base64": png_b64,
        "figure_crop_base64": png_b64,
        "page_number": best_idx + 1,
        "pdf_keyword_score": best_score,
        "page_scores": scores,
        "confidence_tier": 3,  # Always tier 3 — original figure, not reconstructed
        "source": study_name or "",
        "bbox_found": bbox is not None,
    }
