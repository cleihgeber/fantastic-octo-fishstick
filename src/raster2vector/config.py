"""Configuration settings for raster to vector conversion."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class ThresholdMethod(Enum):
    """Methods for converting grayscale to binary."""

    OTSU = "otsu"  # Automatic threshold using Otsu's method
    ADAPTIVE = "adaptive"  # Adaptive threshold for varying lighting
    FIXED = "fixed"  # Fixed threshold value


class SketchStyle(Enum):
    """Style presets for the sketchy output."""

    CLEAN = "clean"  # Minimal processing, smooth curves
    NATURAL = "natural"  # Slight variation, like careful hand drawing
    SKETCHY = "sketchy"  # More variation, like quick sketches
    ROUGH = "rough"  # Maximum variation, very hand-drawn look


@dataclass
class ConversionConfig:
    """Configuration for raster to vector conversion.

    Attributes:
        threshold_method: Method for binarization
        threshold_value: Fixed threshold value (0-255) if using FIXED method
        invert: Whether to invert the image (for white lines on dark background)
        denoise: Apply denoising before processing
        min_path_length: Minimum path length in pixels to include

        sketch_style: Preset style for output
        wobble_amplitude: Amount of random wobble added to lines (0.0-5.0)
        wobble_frequency: Frequency of wobble variations
        corner_rounding: How much to round sharp corners (0.0-1.0)

        simplify_tolerance: Tolerance for path simplification (higher = fewer points)
        curve_fitting: Whether to fit bezier curves to paths

        output_width: Output SVG width in mm (None = use image dimensions)
        output_height: Output SVG height in mm (None = use image dimensions)
        stroke_width: Stroke width in the output SVG

        line_sorting: Sort paths to minimize pen travel
        group_by_color: Group strokes by detected color regions

        preserve_thickness: Generate multiple parallel strokes for thick lines
        stroke_spacing: Spacing between parallel strokes when preserving thickness
        min_width_for_multi: Minimum stroke width to trigger multiple strokes
        max_parallel_strokes: Maximum number of parallel strokes

        fill_hatching: Generate hatching lines for solid filled areas
        hatch_angle: Angle of hatch lines in degrees
        hatch_spacing: Spacing between hatch lines
        min_hatch_area: Minimum area to apply hatching
        cross_hatch: Add perpendicular hatch lines for denser fill
    """

    # Preprocessing
    threshold_method: ThresholdMethod = ThresholdMethod.OTSU
    threshold_value: int = 128
    invert: bool = False
    denoise: bool = True
    blur_kernel: int = 3

    # Path filtering
    min_path_length: int = 5
    remove_small_objects: int = 10

    # Sketch style
    sketch_style: SketchStyle = SketchStyle.NATURAL
    wobble_amplitude: float = 0.5
    wobble_frequency: float = 0.1
    corner_rounding: float = 0.3
    line_variation: float = 0.2

    # Path processing
    simplify_tolerance: float = 1.0
    curve_fitting: bool = True
    smoothing_window: int = 5

    # Output dimensions
    output_width: Optional[float] = None
    output_height: Optional[float] = None
    stroke_width: float = 0.5
    stroke_color: str = "black"

    # Optimization
    line_sorting: bool = True
    merge_nearby_endpoints: bool = True
    merge_distance: float = 3.0

    # Thickness preservation (for rendering mass with multiple strokes)
    preserve_thickness: bool = False
    stroke_spacing: float = 1.5  # Spacing between parallel strokes
    min_width_for_multi: float = 4.0  # Minimum width to generate multiple strokes
    max_parallel_strokes: int = 10  # Maximum parallel strokes for very thick lines

    # Hatching for filled areas
    fill_hatching: bool = False  # Generate hatching for solid filled regions
    hatch_angle: float = 45.0  # Angle of hatch lines in degrees
    hatch_spacing: float = 2.0  # Spacing between hatch lines
    min_hatch_area: int = 200  # Minimum area (in pixels) to apply hatching
    cross_hatch: bool = False  # Add second set of hatch lines at perpendicular angle

    def __post_init__(self):
        """Apply sketch style presets."""
        self._apply_style_preset()

    def _apply_style_preset(self):
        """Apply preset values based on sketch_style."""
        presets = {
            SketchStyle.CLEAN: {
                "wobble_amplitude": 0.0,
                "wobble_frequency": 0.0,
                "corner_rounding": 0.0,
                "line_variation": 0.0,
                "simplify_tolerance": 0.5,
            },
            SketchStyle.NATURAL: {
                "wobble_amplitude": 0.3,
                "wobble_frequency": 0.08,
                "corner_rounding": 0.2,
                "line_variation": 0.15,
                "simplify_tolerance": 1.0,
            },
            SketchStyle.SKETCHY: {
                "wobble_amplitude": 0.8,
                "wobble_frequency": 0.15,
                "corner_rounding": 0.4,
                "line_variation": 0.3,
                "simplify_tolerance": 1.5,
            },
            SketchStyle.ROUGH: {
                "wobble_amplitude": 1.5,
                "wobble_frequency": 0.25,
                "corner_rounding": 0.6,
                "line_variation": 0.5,
                "simplify_tolerance": 2.0,
            },
        }

        if self.sketch_style in presets:
            preset = presets[self.sketch_style]
            # Only apply preset if values are at default
            for key, value in preset.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    @classmethod
    def for_pen_plotter(cls, width_mm: float = 210, height_mm: float = 297) -> "ConversionConfig":
        """Create config optimized for pen plotters (A4 default)."""
        return cls(
            output_width=width_mm,
            output_height=height_mm,
            stroke_width=0.4,
            line_sorting=True,
            sketch_style=SketchStyle.NATURAL,
            simplify_tolerance=1.0,
            curve_fitting=True,
        )

    @classmethod
    def for_laser_cutter(cls, width_mm: float = 300, height_mm: float = 200) -> "ConversionConfig":
        """Create config optimized for laser cutters."""
        return cls(
            output_width=width_mm,
            output_height=height_mm,
            stroke_width=0.1,
            sketch_style=SketchStyle.CLEAN,
            line_sorting=True,
            simplify_tolerance=0.5,
            curve_fitting=True,
        )

    @classmethod
    def with_thickness_preservation(
        cls,
        width_mm: float = 210,
        height_mm: float = 297,
        stroke_spacing: float = 1.5,
    ) -> "ConversionConfig":
        """Create config that preserves stroke thickness with multiple parallel lines.

        Use this when you want thicker strokes in the original to be rendered
        as multiple pen strokes to maintain visual weight/mass.

        Args:
            width_mm: Output width in mm
            height_mm: Output height in mm
            stroke_spacing: Spacing between parallel strokes (smaller = denser)
        """
        return cls(
            output_width=width_mm,
            output_height=height_mm,
            stroke_width=0.4,
            line_sorting=True,
            sketch_style=SketchStyle.NATURAL,
            preserve_thickness=True,
            stroke_spacing=stroke_spacing,
            min_width_for_multi=4.0,
        )

    @classmethod
    def with_fill_hatching(
        cls,
        width_mm: float = 210,
        height_mm: float = 297,
        hatch_spacing: float = 2.0,
        cross_hatch: bool = False,
    ) -> "ConversionConfig":
        """Create config that fills solid areas with hatching lines.

        Use this when your drawing has filled/solid areas that should be
        rendered as hatching patterns for the pen plotter.

        Args:
            width_mm: Output width in mm
            height_mm: Output height in mm
            hatch_spacing: Spacing between hatch lines
            cross_hatch: Whether to add perpendicular hatch lines
        """
        return cls(
            output_width=width_mm,
            output_height=height_mm,
            stroke_width=0.4,
            line_sorting=True,
            sketch_style=SketchStyle.NATURAL,
            preserve_thickness=True,
            fill_hatching=True,
            hatch_spacing=hatch_spacing,
            cross_hatch=cross_hatch,
        )
