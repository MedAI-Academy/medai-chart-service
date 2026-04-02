"""
Deck Route — FastAPI endpoint for /render-deck

Add to your app.py:
    from deck_route import register_deck_route
    register_deck_route(app)
"""

import traceback
import logging
import os
from fastapi import Request
from fastapi.responses import JSONResponse, Response as FastAPIResponse

logger = logging.getLogger(__name__)


def register_deck_route(app):
    """Register the /render-deck endpoints on the FastAPI app."""

    @app.post("/render-deck")
    async def render_deck_endpoint(request: Request):
        """
        POST /render-deck

        Receives a recipe JSON, renders a complete PPTX using
        medaccur templates + native shapes + KM PNG.

        Returns: PPTX binary
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

            return FastAPIResponse(
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

    @app.get("/render-deck/health")
    def deck_health():
        """Health check for deck renderer."""
        from deck_renderer import load_manifest, TEMPLATE_DIR

        manifest = load_manifest()
        template_dir = TEMPLATE_DIR
        templates_exist = os.path.isdir(template_dir)

        template_count = 0
        if templates_exist:
            template_count = len([f for f in os.listdir(template_dir) if f.endswith('.pptx')])

        return {
            "status": "ok",
            "engine": "python-pptx + native shapes",
            "template_dir": template_dir,
            "templates_found": template_count,
            "manifest_version": manifest.get('version', 'unknown') if manifest else 'not loaded',
            "manifest_layouts": len(manifest.get('layout_map', {})) if manifest else 0,
        }
