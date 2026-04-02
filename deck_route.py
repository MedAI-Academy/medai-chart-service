"""
Deck Route — Flask endpoint for /render-deck

Add to your existing app.py:
    from deck_route import register_deck_route
    register_deck_route(app)
"""

import traceback
import logging
from flask import request, jsonify, Response

logger = logging.getLogger(__name__)


def register_deck_route(app):
    """Register the /render-deck endpoint on the Flask app."""

    @app.route("/render-deck", methods=["POST"])
    def render_deck_endpoint():
        """
        POST /render-deck
        
        Receives a recipe JSON, renders a complete PPTX using
        medaccur templates + chart PNGs.
        
        Returns: PPTX binary (application/octet-stream)
        """
        try:
            recipe = request.get_json(force=True)
            if not recipe:
                return jsonify({"error": "No JSON body"}), 400
            if 'slides' not in recipe:
                return jsonify({"error": "Missing 'slides' in recipe"}), 400

            # Import here to avoid circular imports
            from deck_renderer import render_deck, load_manifest
            from charts.shape_renderer import add_chart_shapes
            from charts.chart_renderer import render_chart

            # Ensure manifest is loaded
            load_manifest()

            # Render the deck (native shapes + KM PNG)
            pptx_buf = render_deck(
                recipe,
                chart_renderer=render_chart,      # KM curve PNG only
                shape_renderer=add_chart_shapes    # Forest, Waterfall, Swimmer, ORR bars
            )

            # Build filename
            meta = recipe.get('metadata', {})
            drug = meta.get('drug', 'MAP').replace(' ', '_')
            country = meta.get('country', 'EMEA').replace(' ', '_')
            year = meta.get('year', '2027')
            filename = f"{drug}_MAP_{country}_{year}.pptx"

            return Response(
                pptx_buf.read(),
                mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation',
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}"',
                    'Access-Control-Allow-Origin': '*',
                }
            )

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Deck render error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/render-deck/health", methods=["GET"])
    def deck_health():
        """Health check for deck renderer."""
        import os
        from deck_renderer import load_manifest, TEMPLATE_DIR

        manifest = load_manifest()
        template_dir = TEMPLATE_DIR
        templates_exist = os.path.isdir(template_dir)

        # Count available templates
        template_count = 0
        if templates_exist:
            template_count = len([f for f in os.listdir(template_dir) if f.endswith('.pptx')])

        return jsonify({
            "status": "ok",
            "engine": "python-pptx + matplotlib",
            "template_dir": template_dir,
            "templates_found": template_count,
            "manifest_version": manifest.get('version', 'unknown') if manifest else 'not loaded',
            "manifest_layouts": len(manifest.get('layout_map', {})) if manifest else 0,
        })
