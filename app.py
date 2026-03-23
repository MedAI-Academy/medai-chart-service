"""
MedAI Chart Service — Publication-Quality Scientific Charts
FastAPI microservice for generating NEJM/Lancet-grade chart PNGs.

Endpoints:
  POST /charts/kaplan-meier   → KM survival curve PNG
  POST /charts/forest-plot    → Forest plot PNG  (coming soon)
  POST /charts/waterfall      → Waterfall chart PNG (coming soon)
  POST /charts/swimmer        → Swimmer plot PNG (coming soon)
  GET  /health                → Health check
"""

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import io
import numpy as np

from charts.kaplan_meier import render_kaplan_meier

app = FastAPI(title="MedAI Chart Service", version="1.0.0")

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


# ── Endpoints ───────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/charts/kaplan-meier")
def kaplan_meier(req: KMRequest):
    buf = render_kaplan_meier(req)
    return Response(content=buf.getvalue(), media_type="image/png")
