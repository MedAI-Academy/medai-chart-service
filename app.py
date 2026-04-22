"""
MedAI Chart Service — Publication-Quality Scientific Charts + PPTX Deck Renderer
FastAPI microservice on Railway.

Endpoints:
  POST /charts/kaplan-meier   → KM survival curve PNG
  POST /render-deck           → Complete PPTX from recipe JSON + templates
  GET  /render-deck/health    → Deck renderer health check
  GET  /health                → General health check
"""

from fastapi import FastAPI, Response, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import io
import os
import traceback
import logging
import numpy as np

# Phase 5: KM extraction is now Vision-based (Gemini). The old kaplan_meier
# (matplotlib IPD renderer) and extract_km (OpenCV pixel tracer) modules
# have been removed — see charts/extract_km_from_pdf.py for the new
# pipeline (PDF → keyword scoring → Vision passes → NEJM-style render).
from charts.extract_km_from_pdf import extract_km_from_pdf
from charts.extract_forest_from_pdf import extract_forest_from_pdf
from charts.forest_plot_nejm import render_forest_nejm

logger = logging.getLogger(__name__)

app = FastAPI(title="MedAI Chart Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Models ──────────────────────────────────────────────
# Phase 5: KMArm + KMRequest models removed along with the
# /charts/kaplan-meier endpoint — they served the old IPD-input
# matplotlib renderer which is no longer in the call graph.


# ══════════════════════════════════════════════════════════════════
# CHART ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "engine": "FastAPI + python-pptx"}



# ══════════════════════════════════════════════════════════════════
# KM CURVE EXTRACTION — Vision-based pipeline
# ══════════════════════════════════════════════════════════════════
# Phase 5: ArmColorHint + ExtractKMRequest models removed along with the
# /extract-km endpoint — they served the old OpenCV pixel-tracing pipeline
# which has been replaced by the Vision pipeline (see /extract-km-from-pdf).
# Pre-cropped image extraction is no longer a public endpoint; if you need
# to test on a single image, build a tiny PDF wrapping it and POST that.


class ExtractKMFromPDFRequest(BaseModel):
    pdf_base64: str = Field(..., description="Full PDF document as base64")
    study_name: Optional[str] = Field(
        None, description="Optional study name for reference lookup"
    )
    min_score: int = Field(
        4,
        description="Minimum keyword score required to accept a page as "
                    "a KM figure page",
    )


@app.post("/extract-km-from-pdf")
def extract_km_from_pdf_endpoint(req: ExtractKMFromPDFRequest):
    """
    POST /extract-km-from-pdf — Extract KM curves from a full PDF.

    Scans every page for KM-figure keywords ("no. at risk", "hazard
    ratio", "overall survival", ...), picks the best-scoring page,
    auto-crops the figure, renders at 400 DPI, and runs the existing
    extract_km pipeline on the crop.

    Response: same shape as /extract-km plus page_number,
    figure_crop_base64, and pdf_keyword_score.
    """
    try:
        result = extract_km_from_pdf(
            pdf_base64=req.pdf_base64,
            study_name=req.study_name,
            min_score=req.min_score,
        )
        return JSONResponse(result)
    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"extract-km-from-pdf error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ══════════════════════════════════════════════════════════════════
# FOREST PLOT EXTRACTION — Crop from PDF (no pixel tracing)
# ══════════════════════════════════════════════════════════════════

class ExtractForestFromPDFRequest(BaseModel):
    pdf_base64: str = Field(..., description="Full PDF document as base64")
    study_name: Optional[str] = Field(
        None, description="Optional study name for reference lookup"
    )
    min_score: int = Field(
        4,
        description="Minimum effective keyword score (forest - km_penalty) "
                    "required to accept a page as a Forest Plot page",
    )


# ── Phase 4A: reconstruction-based forest plot ────────────────────
# Unlike /extract-forest-from-pdf (which crops the original figure),
# /charts/forest-plot takes STRUCTURED subgroup data and renders a
# NEJM-style forest plot in the platform's own style. Liability
# model matches KM curves: own visualisation, "Reconstructed from [ref]"
# footer, confidence tier 2. No copyright exposure.

class ForestSubgroup(BaseModel):
    # Header row (group separator)
    is_header: Optional[bool] = Field(None, description="True if this is a group-heading row")
    category: Optional[str] = Field(None, description="Group label when is_header=True")
    # Data row
    name: Optional[str] = Field(None, description="Subgroup label (e.g. 'Female', '<75 yr')")
    n: Optional[str] = Field(None, description="Sample size annotation (e.g. '286 vs 145')")
    hr: Optional[float] = Field(None, description="Hazard ratio (required for data rows)")
    ci_low: Optional[float] = Field(None, description="95% CI lower bound")
    ci_high: Optional[float] = Field(None, description="95% CI upper bound")
    is_overall: Optional[bool] = Field(
        None, description="True if this is the 'All patients' row (rendered as diamond)"
    )
    hr_text: Optional[str] = Field(
        None,
        description="Optional pre-formatted HR text; falls back to '{hr:.2f} ({ci_low:.2f}–{ci_high:.2f})'",
    )


class ForestPlotRequest(BaseModel):
    subgroups: list[ForestSubgroup] = Field(
        ..., min_length=1, description="Ordered list of subgroup rows (headers + data rows)"
    )
    title: Optional[str] = Field("", description="Plot title")
    subtitle: Optional[str] = Field("", description="Plot subtitle")
    favours_left: Optional[str] = Field(
        "experimental better", description="Short generic label under the left arrow (max ~20 chars, e.g. 'experimental better')"
    )
    favours_right: Optional[str] = Field(
        "control better", description="Short generic label under the right arrow (max ~20 chars, e.g. 'control better')"
    )
    reference_line: Optional[float] = Field(1.0, description="Null-effect HR (default 1.0)")
    source: Optional[str] = Field(
        "", description="Citation used in the 'Reconstructed from …' footer (liability marker)"
    )
    dpi: Optional[int] = Field(300, description="Output DPI")


@app.post("/extract-forest-from-pdf")
def extract_forest_from_pdf_endpoint(req: ExtractForestFromPDFRequest):
    """
    POST /extract-forest-from-pdf — Extract Forest Plot figures from a full PDF.

    Scans every page for forest-plot-specific keywords ("subgroup analysis",
    "favours", "forest plot", "hazard ratio", ...), subtracts KM-page
    indicators to avoid false positives, picks the best-scoring page,
    auto-crops the figure at 400 DPI, and returns the crop directly.

    Unlike /extract-km-from-pdf, this does NOT reconstruct the plot —
    the original figure already contains all subgroup labels, N-values,
    HR values, diamonds, and whiskers at publication quality.

    Response:
      - png_base64:            the 400 DPI PNG crop
      - figure_crop_base64:    same (for compatibility with KM response shape)
      - page_number:           1-indexed page where the figure was found
      - pdf_keyword_score:     the effective score of the selected page
      - page_scores:           list per page with forest/km_penalty/effective
      - confidence_tier:       always 3 (original figure, not reconstructed)
      - source:                study_name if provided
      - bbox_found:            true if a figure cluster was detected, false
                               if we fell back to rendering the whole page
    """
    try:
        result = extract_forest_from_pdf(
            pdf_base64=req.pdf_base64,
            study_name=req.study_name,
            min_score=req.min_score,
        )
        return JSONResponse(result)
    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"extract-forest-from-pdf error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/charts/forest-plot")
def forest_plot_endpoint(req: ForestPlotRequest):
    """
    POST /charts/forest-plot — Render a NEJM-style subgroup forest plot from
    STRUCTURED data. This is the reconstruction-based counterpart to
    /extract-forest-from-pdf.

    The caller supplies subgroup rows (headers + data) with HR and 95% CI.
    The renderer produces the platform's own visualisation; the "Reconstructed
    from {source}" footer makes the liability model explicit (matches KM
    curves, confidence tier 2). No copyright exposure — own visualisation of
    data the caller already extracted from the paper.

    Response: binary PNG (image/png), 300 DPI by default.
    """
    try:
        png_bytes = render_forest_nejm(
            subgroups=[sg.model_dump(exclude_none=True) for sg in req.subgroups],
            title=req.title or "",
            subtitle=req.subtitle or "",
            favours_left=req.favours_left or "experimental better",
            favours_right=req.favours_right or "control better",
            reference_line=req.reference_line if req.reference_line is not None else 1.0,
            source=req.source or "",
            dpi=req.dpi or 300,
        )
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"forest-plot render error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# Phase 5: /extract-km endpoint removed — old OpenCV pixel-tracing pipeline
# replaced by the Vision pipeline in /extract-km-from-pdf. To process a
# bare image, wrap it in a 1-page PDF first.


# ══════════════════════════════════════════════════════════════════
# DECK RENDERER — Template Clone+Swap + Native Shapes
# ══════════════════════════════════════════════════════════════════

@app.get("/render-deck/health")
def deck_health():
    """Health check for deck renderer — shows template count and manifest."""
    try:
        from deck_renderer import load_manifest, TEMPLATE_DIR
        manifest = load_manifest()
        td = TEMPLATE_DIR
        tc = len([f for f in os.listdir(td) if f.endswith('.pptx')]) if os.path.isdir(td) else 0
        return {
            "status": "ok",
            "engine": "python-pptx + native shapes",
            "template_dir": td,
            "templates_found": tc,
            "manifest_version": manifest.get('version', '?') if manifest else 'none',
            "manifest_layouts": len(manifest.get('layout_map', {})) if manifest else 0
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/render-deck")
async def render_deck_endpoint(request: Request):
    """
    POST /render-deck — Render a complete PPTX from recipe JSON.

    Uses medaccur templates with {{placeholder}} replacement.
    Adds native PowerPoint shapes for Forest Plot, Waterfall, Swimmer, ORR bars.
    KM curves rendered as matplotlib PNG (only exception).
    """
    try:
        recipe = await request.json()
        if not recipe:
            return JSONResponse({"error": "No JSON body"}, status_code=400)
        if 'slides' not in recipe:
            return JSONResponse({"error": "Missing 'slides' in recipe"}, status_code=400)

        from deck_renderer import render_deck, load_manifest
        from charts.shape_renderer import add_chart_shapes
        from charts.chart_renderer import render_chart

        load_manifest()

        pptx_buf = render_deck(
            recipe,
            chart_renderer=render_chart,
            shape_renderer=add_chart_shapes
        )

        meta = recipe.get('metadata', {})
        drug = meta.get('drug', 'MAP').replace(' ', '_')
        country = meta.get('country', 'EMEA').replace(' ', '_')
        year = meta.get('year', '2027')
        filename = f"{drug}_MAP_{country}_{year}.pptx"

        return Response(
            content=pptx_buf.read(),
            media_type='application/vnd.openxmlformats-officedocument.presentationml.presentation',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Access-Control-Allow-Origin': '*',
            }
        )

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Deck render error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
