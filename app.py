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

from charts.kaplan_meier import render_kaplan_meier
from charts.extract_km import extract_km
from charts.extract_km_from_pdf import extract_km_from_pdf

logger = logging.getLogger(__name__)

app = FastAPI(title="MedAI Chart Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Models ──────────────────────────────────────────────

class KMArm(BaseModel):
    label: str = Field(..., description="Arm name, e.g. 'Pembrolizumab + Chemo'")
    times: list[float] = Field(..., description="Event/censor times in months")
    events: list[int] = Field(..., description="1 = event (death), 0 = censored")
    color: Optional[str] = Field(None, description="Hex color override")


class KMRequest(BaseModel):
    arms: list[KMArm] = Field(..., min_length=1, max_length=4)
    title: Optional[str] = Field(None, description="Chart title")
    xlabel: Optional[str] = Field("Time (months)", description="X-axis label")
    ylabel: Optional[str] = Field("Overall Survival (%)", description="Y-axis label")
    show_ci: bool = Field(True, description="Show 95% confidence intervals")
    show_censoring: bool = Field(True, description="Show censoring tick marks")
    show_at_risk: bool = Field(True, description="Show number-at-risk table")
    show_median: bool = Field(True, description="Show median survival lines")
    hr_text: Optional[str] = Field(None, description="HR box text, e.g. 'HR 0.68 (95% CI 0.56–0.84); p<0.001'")
    width: float = Field(10, description="Figure width in inches")
    height: float = Field(7, description="Figure height in inches")
    dpi: int = Field(300, description="Resolution")


# ══════════════════════════════════════════════════════════════════
# CHART ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "engine": "FastAPI + python-pptx"}


@app.post("/charts/kaplan-meier")
def kaplan_meier(req: KMRequest):
    buf = render_kaplan_meier(req)
    return Response(content=buf.getvalue(), media_type="image/png")


# ══════════════════════════════════════════════════════════════════
# KM CURVE EXTRACTION — OpenCV pixel-level reader
# ══════════════════════════════════════════════════════════════════

class ArmColorHint(BaseModel):
    name: str = Field(..., description="Arm name")
    color_hint: Optional[str] = Field(
        None,
        description="Color hint: blue | green | red | orange | yellow | "
                    "purple | black | gray (or German: dunkelblau, "
                    "waldgruen, grau)",
    )


class ExtractKMRequest(BaseModel):
    image_base64: str = Field(..., description="PNG or JPG as base64")
    study_name: Optional[str] = Field(
        None,
        description="Optional study name (e.g. 'VIALE-A', 'AQUILA', 'CLEAR') "
                    "for metadata lookup",
    )
    arm_colors: Optional[list[ArmColorHint]] = Field(
        None,
        description="Optional arm color hints. If omitted and study_name "
                    "is unknown, arms are auto-detected.",
    )
    x_range: Optional[list[float]] = Field(
        None, description="Optional axis range override [min, max]"
    )
    y_range: Optional[list[float]] = Field(
        None, description="Optional Y-axis range override [min, max]"
    )
    plot_bounds: Optional[list[int]] = Field(
        None,
        description="Optional plot pixel bounds override [xl, yt, xr, yb]",
    )


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


@app.post("/extract-km")
def extract_km_endpoint(req: ExtractKMRequest):
    """
    POST /extract-km — Extract Kaplan-Meier curves from a publication figure.

    Input:  image (base64), optional study_name, optional arm_colors.
    Output: arm coordinates [[time, survival%], ...], censored times, median,
            a reconstructed medaccur-style PNG, confidence tier 1-3.
    """
    try:
        arm_colors = (
            [a.model_dump() for a in req.arm_colors] if req.arm_colors else None
        )
        result = extract_km(
            image_base64=req.image_base64,
            study_name=req.study_name,
            arm_colors=arm_colors,
            x_range=tuple(req.x_range) if req.x_range else None,
            y_range=tuple(req.y_range) if req.y_range else None,
            plot_bounds=tuple(req.plot_bounds) if req.plot_bounds else None,
        )
        return JSONResponse(result)
    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"extract-km error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


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
