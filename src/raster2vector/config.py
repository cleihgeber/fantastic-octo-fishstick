"""Configuration settings for raster to vector conversion."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict


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


class PaperSize(Enum):
    """Standard paper sizes with dimensions in mm (width x height in portrait)."""

    # ISO A Series
    A0 = "a0"
    A1 = "a1"
    A2 = "a2"
    A3 = "a3"
    A4 = "a4"
    A5 = "a5"
    A6 = "a6"

    # ISO B Series
    B0 = "b0"
    B1 = "b1"
    B2 = "b2"
    B3 = "b3"
    B4 = "b4"
    B5 = "b5"

    # US Sizes
    LETTER = "letter"
    LEGAL = "legal"
    TABLOID = "tabloid"

    # Other common sizes
    POSTCARD = "postcard"
    SQUARE_100 = "square100"
    SQUARE_150 = "square150"
    SQUARE_200 = "square200"

    # Custom (use with custom dimensions)
    CUSTOM = "custom"


# Paper dimensions in mm (width, height) in portrait orientation
PAPER_DIMENSIONS: Dict[PaperSize, Tuple[float, float]] = {
    # ISO A Series (based on √2 ratio)
    PaperSize.A0: (841, 1189),
    PaperSize.A1: (594, 841),
    PaperSize.A2: (420, 594),
    PaperSize.A3: (297, 420),
    PaperSize.A4: (210, 297),
    PaperSize.A5: (148, 210),
    PaperSize.A6: (105, 148),

    # ISO B Series
    PaperSize.B0: (1000, 1414),
    PaperSize.B1: (707, 1000),
    PaperSize.B2: (500, 707),
    PaperSize.B3: (353, 500),
    PaperSize.B4: (250, 353),
    PaperSize.B5: (176, 250),

    # US Sizes (converted to mm)
    PaperSize.LETTER: (216, 279),  # 8.5" x 11"
    PaperSize.LEGAL: (216, 356),   # 8.5" x 14"
    PaperSize.TABLOID: (279, 432), # 11" x 17"

    # Other common sizes
    PaperSize.POSTCARD: (100, 148),
    PaperSize.SQUARE_100: (100, 100),
    PaperSize.SQUARE_150: (150, 150),
    PaperSize.SQUARE_200: (200, 200),

    # Custom placeholder
    PaperSize.CUSTOM: (210, 297),
}


def get_paper_size(
    paper: PaperSize,
    landscape: bool = False,
    custom_width: Optional[float] = None,
    custom_height: Optional[float] = None,
) -> Tuple[float, float]:
    """Get paper dimensions in mm.

    Args:
        paper: Paper size preset
        landscape: If True, swap width and height
        custom_width: Custom width for CUSTOM paper size
        custom_height: Custom height for CUSTOM paper size

    Returns:
        Tuple of (width_mm, height_mm)
    """
    if paper == PaperSize.CUSTOM:
        if custom_width is None or custom_height is None:
            raise ValueError("Custom paper size requires custom_width and custom_height")
        width, height = custom_width, custom_height
    else:
        width, height = PAPER_DIMENSIONS[paper]

    if landscape:
        return (height, width)
    return (width, height)


def paper_size_from_string(name: str) -> PaperSize:
    """Convert string to PaperSize enum.

    Args:
        name: Paper size name (case-insensitive)

    Returns:
        PaperSize enum value

    Raises:
        ValueError: If name is not recognized
    """
    name_lower = name.lower().strip()

    # Direct enum value match
    for paper in PaperSize:
        if paper.value == name_lower:
            return paper

    # Common aliases
    aliases = {
        "a0": PaperSize.A0,
        "a1": PaperSize.A1,
        "a2": PaperSize.A2,
        "a3": PaperSize.A3,
        "a4": PaperSize.A4,
        "a5": PaperSize.A5,
        "a6": PaperSize.A6,
        "b0": PaperSize.B0,
        "b1": PaperSize.B1,
        "b2": PaperSize.B2,
        "b3": PaperSize.B3,
        "b4": PaperSize.B4,
        "b5": PaperSize.B5,
        "letter": PaperSize.LETTER,
        "us-letter": PaperSize.LETTER,
        "usletter": PaperSize.LETTER,
        "legal": PaperSize.LEGAL,
        "us-legal": PaperSize.LEGAL,
        "tabloid": PaperSize.TABLOID,
        "ledger": PaperSize.TABLOID,
        "postcard": PaperSize.POSTCARD,
        "100x100": PaperSize.SQUARE_100,
        "150x150": PaperSize.SQUARE_150,
        "200x200": PaperSize.SQUARE_200,
        "square100": PaperSize.SQUARE_100,
        "square150": PaperSize.SQUARE_150,
        "square200": PaperSize.SQUARE_200,
        "custom": PaperSize.CUSTOM,
    }

    if name_lower in aliases:
        return aliases[name_lower]

    raise ValueError(
        f"Unknown paper size: '{name}'. "
        f"Valid sizes: {', '.join(p.value for p in PaperSize if p != PaperSize.CUSTOM)}"
    )


def list_paper_sizes() -> Dict[str, Tuple[float, float]]:
    """Get all available paper sizes and their dimensions.

    Returns:
        Dict mapping paper name to (width, height) in mm
    """
    return {
        paper.value: dims
        for paper, dims in PAPER_DIMENSIONS.items()
        if paper != PaperSize.CUSTOM
    }


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
    def for_paper_size(
        cls,
        paper: PaperSize = PaperSize.A4,
        landscape: bool = False,
        custom_width: Optional[float] = None,
        custom_height: Optional[float] = None,
        **kwargs,
    ) -> "ConversionConfig":
        """Create config with a specific paper size.

        Args:
            paper: Paper size preset (A4, A3, Letter, etc.)
            landscape: If True, use landscape orientation
            custom_width: Width in mm for custom paper size
            custom_height: Height in mm for custom paper size
            **kwargs: Additional config options to override

        Returns:
            ConversionConfig with specified paper size

        Example:
            # A3 landscape
            config = ConversionConfig.for_paper_size(PaperSize.A3, landscape=True)

            # US Letter
            config = ConversionConfig.for_paper_size(PaperSize.LETTER)

            # Custom size
            config = ConversionConfig.for_paper_size(
                PaperSize.CUSTOM,
                custom_width=300,
                custom_height=200
            )
        """
        width, height = get_paper_size(paper, landscape, custom_width, custom_height)

        defaults = {
            "output_width": width,
            "output_height": height,
            "stroke_width": 0.4,
            "line_sorting": True,
            "sketch_style": SketchStyle.NATURAL,
            "curve_fitting": True,
        }
        defaults.update(kwargs)

        return cls(**defaults)

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
