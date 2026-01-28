"""
Raster2Vector - Convert raster line drawings to vectors for pen plotters.

This library converts line drawings in raster format (PNG, JPG, etc.) to
vector format (SVG) suitable for pen plotters. The output consists of
single-line strokes with optional sketchy, human-like qualities.
"""

__version__ = "0.1.0"

from .core import convert, RasterToVectorConverter
from .config import ConversionConfig

__all__ = ["convert", "RasterToVectorConverter", "ConversionConfig", "__version__"]
