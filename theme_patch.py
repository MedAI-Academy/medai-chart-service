"""
═══════════════════════════════════════════════════════════════
THEME PATCH for deck_renderer.py — Railway medaccur Chart Service
═══════════════════════════════════════════════════════════════

INTEGRATION:
1. Add this import at the top of deck_renderer.py:
     from pptx.util import Inches, Pt, Emu
     from pptx.dml.color import RGBColor
     from pptx.enum.dml import MSO_THEME_COLOR
     import copy

2. Add the apply_theme() function below to deck_renderer.py

3. In render_deck(), after cloning + placeholder replacement + auto_shrink,
   call apply_theme() on each slide:

     # After existing code:
     #   replace_placeholders(slide, merged)
     #   enable_auto_shrink(slide)
     
     # Add:
     theme = recipe.get('theme', {})
     color_swap = theme.get('color_swap', {})
     if color_swap:
         apply_theme(slide, color_swap)

═══════════════════════════════════════════════════════════════
"""

from pptx.dml.color import RGBColor
import logging

log = logging.getLogger(__name__)


def hex_to_rgb(hex_str: str) -> RGBColor:
    """Convert hex string like 'FF5F7E' to RGBColor."""
    hex_str = hex_str.lstrip('#')
    return RGBColor(
        int(hex_str[0:2], 16),
        int(hex_str[2:4], 16),
        int(hex_str[4:6], 16)
    )


def rgb_to_hex(rgb: RGBColor) -> str:
    """Convert RGBColor to uppercase hex string like '7C6FFF'."""
    return f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def colors_match(c1_hex: str, c2_hex: str, tolerance: int = 8) -> bool:
    """
    Check if two hex colors are close enough to be considered a match.
    Tolerance accounts for slight variations from PowerPoint's color handling.
    """
    r1, g1, b1 = int(c1_hex[0:2], 16), int(c1_hex[2:4], 16), int(c1_hex[4:6], 16)
    r2, g2, b2 = int(c2_hex[0:2], 16), int(c2_hex[2:4], 16), int(c2_hex[4:6], 16)
    return abs(r1 - r2) <= tolerance and abs(g1 - g2) <= tolerance and abs(b1 - b2) <= tolerance


def apply_theme(slide, color_swap: dict):
    """
    Apply theme colors to a slide by swapping base template colors
    with the theme's replacement colors.
    
    color_swap: dict mapping source hex → target hex
                e.g. {'0B1A3B': 'FFFFFF', '7C6FFF': '0D9488', ...}
    
    Handles:
    - Shape solid fills (backgrounds)
    - Shape line/border colors
    - Text run font colors
    - Table cell fills
    - Gradient fill stop colors
    """
    if not color_swap:
        return
    
    swapped = 0
    
    for shape in slide.shapes:
        swapped += _apply_to_shape(shape, color_swap)
        
        # Handle table cells
        if shape.has_table:
            for row in shape.table.rows:
                for cell in row.cells:
                    swapped += _apply_to_cell(cell, color_swap)
                    # Cell text
                    if cell.text_frame:
                        swapped += _apply_to_text_frame(cell.text_frame, color_swap)
    
    # Handle slide background
    try:
        bg = slide.background
        if bg.fill and bg.fill.type is not None:
            swapped += _apply_to_fill(bg.fill, color_swap)
    except Exception:
        pass
    
    if swapped > 0:
        log.info(f"  Theme: {swapped} color swaps applied")


def _apply_to_shape(shape, color_swap: dict) -> int:
    """Apply color swaps to a single shape (fill, line, text)."""
    swapped = 0
    
    # Shape fill
    try:
        if hasattr(shape, 'fill') and shape.fill.type is not None:
            swapped += _apply_to_fill(shape.fill, color_swap)
    except Exception:
        pass
    
    # Shape line/border
    try:
        line = shape.line
        if line.fill and line.fill.type is not None:
            swapped += _apply_to_fill(line.fill, color_swap)
        elif line.color and line.color.rgb:
            hex_val = rgb_to_hex(line.color.rgb)
            new_hex = _find_swap(hex_val, color_swap)
            if new_hex:
                line.color.rgb = hex_to_rgb(new_hex)
                swapped += 1
    except Exception:
        pass
    
    # Text in shape
    try:
        if shape.has_text_frame:
            swapped += _apply_to_text_frame(shape.text_frame, color_swap)
    except Exception:
        pass
    
    # Group shapes — recurse
    try:
        if hasattr(shape, 'shapes'):
            for child in shape.shapes:
                swapped += _apply_to_shape(child, color_swap)
    except Exception:
        pass
    
    return swapped


def _apply_to_fill(fill, color_swap: dict) -> int:
    """Apply color swaps to a fill (solid or gradient)."""
    swapped = 0
    
    try:
        fill_type = fill.type
    except Exception:
        return 0
    
    # Solid fill
    if fill_type == 1:  # MSO_FILL.SOLID = 1
        try:
            rgb = fill.fore_color.rgb
            if rgb:
                hex_val = rgb_to_hex(rgb)
                new_hex = _find_swap(hex_val, color_swap)
                if new_hex:
                    fill.solid()
                    fill.fore_color.rgb = hex_to_rgb(new_hex)
                    swapped += 1
        except Exception:
            pass
    
    # Gradient fill
    elif fill_type == 3:  # MSO_FILL.GRADIENT = 3
        try:
            for stop in fill.gradient_stops:
                try:
                    rgb = stop.color.rgb
                    if rgb:
                        hex_val = rgb_to_hex(rgb)
                        new_hex = _find_swap(hex_val, color_swap)
                        if new_hex:
                            stop.color.rgb = hex_to_rgb(new_hex)
                            swapped += 1
                except Exception:
                    pass
        except Exception:
            pass
    
    return swapped


def _apply_to_text_frame(tf, color_swap: dict) -> int:
    """Apply color swaps to all text runs in a text frame."""
    swapped = 0
    for para in tf.paragraphs:
        for run in para.runs:
            try:
                font_color = run.font.color
                if font_color and font_color.rgb:
                    hex_val = rgb_to_hex(font_color.rgb)
                    new_hex = _find_swap(hex_val, color_swap)
                    if new_hex:
                        font_color.rgb = hex_to_rgb(new_hex)
                        swapped += 1
            except Exception:
                pass
    return swapped


def _apply_to_cell(cell, color_swap: dict) -> int:
    """Apply color swaps to a table cell's fill."""
    swapped = 0
    try:
        # Access cell fill via XML (python-pptx cell.fill can be tricky)
        from lxml import etree
        tcPr = cell._tc.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}tcPr')
        if tcPr is not None:
            solidFill = tcPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}solidFill')
            if solidFill is not None:
                srgbClr = solidFill.find('{http://schemas.openxmlformats.org/drawingml/2006/main}srgbClr')
                if srgbClr is not None:
                    hex_val = srgbClr.get('val', '').upper()
                    new_hex = _find_swap(hex_val, color_swap)
                    if new_hex:
                        srgbClr.set('val', new_hex)
                        swapped += 1
    except Exception:
        pass
    return swapped


def _find_swap(hex_val: str, color_swap: dict) -> str | None:
    """
    Find a matching swap color. Uses tolerance matching to handle
    slight color variations from PowerPoint's internal rendering.
    """
    hex_val = hex_val.upper()
    
    # Exact match first (fast path)
    for src, tgt in color_swap.items():
        if hex_val == src.upper():
            return tgt.upper()
    
    # Fuzzy match (tolerance 8 per channel — catches PPT rounding)
    for src, tgt in color_swap.items():
        if colors_match(hex_val, src.upper(), tolerance=8):
            return tgt.upper()
    
    return None
