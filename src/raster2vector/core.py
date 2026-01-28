"""Core conversion functionality."""

import time
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Callable
from dataclasses import dataclass

from .config import ConversionConfig


# Progress callback type: (step_name, step_number, total_steps, percent_complete)
ProgressCallback = Callable[[str, int, int, float], None]
from .preprocessing import preprocess
from .skeleton import extract_skeleton, clean_skeleton, analyze_skeleton
from .tracing import trace_paths, merge_nearby_endpoints, sort_paths_for_plotting, filter_paths, Path as TracedPath
from .curves import process_all_paths, ProcessedPath
from .svg_output import create_svg, save_svg, SVGDocument, paths_to_gcode
from .stroke_width import expand_paths_for_thickness


@dataclass
class ConversionResult:
    """Result of raster to vector conversion.

    Attributes:
        svg: The SVG document
        paths: List of processed paths
        stats: Conversion statistics
    """

    svg: SVGDocument
    paths: List[ProcessedPath]
    stats: dict


class RasterToVectorConverter:
    """Main converter class for raster to vector conversion.

    This class provides a configurable pipeline for converting
    raster line drawings to vector format suitable for pen plotters.

    Example:
        converter = RasterToVectorConverter()
        result = converter.convert("input.png", "output.svg")

        # Or with custom configuration
        config = ConversionConfig.for_pen_plotter(width_mm=200, height_mm=200)
        config.sketch_style = SketchStyle.SKETCHY
        converter = RasterToVectorConverter(config)
        result = converter.convert("input.png", "output.svg")
    """

    def __init__(self, config: Optional[ConversionConfig] = None):
        """Initialize the converter.

        Args:
            config: Conversion configuration (uses defaults if not provided)
        """
        self.config = config or ConversionConfig()

    def convert(
        self,
        input_path: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ConversionResult:
        """Convert a raster image to vector.

        Args:
            input_path: Input image path or numpy array
            output_path: Output SVG path (optional - if None, SVG is not saved)
            seed: Random seed for reproducible sketchy effects
            progress_callback: Optional callback for progress updates
                              (step_name, step_number, total_steps, percent)

        Returns:
            ConversionResult with SVG, paths, and statistics
        """
        total_steps = 10

        def report_progress(step_name: str, step_num: int):
            if progress_callback:
                percent = (step_num / total_steps) * 100
                progress_callback(step_name, step_num, total_steps, percent)

        # Step 1: Preprocess image
        report_progress("Preprocessing image", 1)
        binary, original_size = preprocess(input_path, self.config)
        source_width, source_height = original_size

        # Step 2: Extract skeleton
        report_progress("Extracting skeleton", 2)
        skeleton = extract_skeleton(binary)
        skeleton = clean_skeleton(
            skeleton,
            remove_small=self.config.remove_small_objects,
            prune_spur_length=self.config.min_path_length,
        )

        # Step 3: Analyze skeleton
        report_progress("Analyzing skeleton", 3)
        analysis = analyze_skeleton(skeleton)

        # Step 4: Trace paths
        report_progress("Tracing paths", 4)
        paths = trace_paths(skeleton, min_path_length=self.config.min_path_length)

        # Step 5: Merge nearby endpoints if configured
        report_progress("Merging endpoints", 5)
        if self.config.merge_nearby_endpoints:
            paths = merge_nearby_endpoints(paths, self.config.merge_distance)

        # Step 6: Filter short paths
        paths = filter_paths(paths, min_length=self.config.min_path_length)

        # Step 7: Expand paths for thickness preservation (if enabled)
        report_progress("Processing thickness", 6)
        thickness_expanded = False
        if self.config.preserve_thickness or self.config.fill_hatching:
            paths = expand_paths_for_thickness(
                paths,
                binary,
                stroke_spacing=self.config.stroke_spacing,
                min_width_for_multi=self.config.min_width_for_multi,
                include_hatching=self.config.fill_hatching,
                hatch_angle=self.config.hatch_angle,
                hatch_spacing=self.config.hatch_spacing,
                min_hatch_area=self.config.min_hatch_area,
            )
            thickness_expanded = True

            # Add cross-hatching if enabled
            if self.config.cross_hatch and self.config.fill_hatching:
                cross_paths = expand_paths_for_thickness(
                    [],  # No base paths, just hatching
                    binary,
                    stroke_spacing=self.config.stroke_spacing,
                    min_width_for_multi=self.config.min_width_for_multi,
                    include_hatching=True,
                    hatch_angle=self.config.hatch_angle + 90,  # Perpendicular
                    hatch_spacing=self.config.hatch_spacing,
                    min_hatch_area=self.config.min_hatch_area,
                )
                paths.extend(cross_paths)

        # Step 8: Sort paths for efficient plotting
        report_progress("Sorting paths", 7)
        if self.config.line_sorting:
            paths = sort_paths_for_plotting(paths)

        # Step 9: Process paths (simplify, smooth, add effects, fit curves)
        report_progress("Applying effects", 8)
        processed_paths = process_all_paths(paths, self.config, seed)

        # Step 10: Determine output dimensions
        report_progress("Creating SVG", 9)
        if self.config.output_width and self.config.output_height:
            out_width = self.config.output_width
            out_height = self.config.output_height
        else:
            # Use source dimensions as pixels
            out_width = source_width
            out_height = source_height

        # Step 11: Create SVG
        svg_doc = create_svg(
            processed_paths,
            width=out_width,
            height=out_height,
            config=self.config,
            units="mm" if self.config.output_width else "px",
            source_width=source_width,
            source_height=source_height,
        )

        # Step 12: Save if output path provided
        report_progress("Finalizing", 10)
        if output_path:
            save_svg(svg_doc, output_path)

        # Collect statistics
        stats = {
            "source_width": source_width,
            "source_height": source_height,
            "skeleton_pixels": analysis.pixel_count,
            "endpoints": len(analysis.endpoints),
            "junctions": len(analysis.junctions),
            "paths_traced": len(paths),
            "paths_output": len(processed_paths),
            "total_path_length": svg_doc.total_path_length,
            "thickness_preserved": thickness_expanded,
            "fill_hatching": self.config.fill_hatching,
        }

        return ConversionResult(svg=svg_doc, paths=processed_paths, stats=stats)

    def convert_to_gcode(
        self,
        input_path: Union[str, Path, np.ndarray],
        output_path: Union[str, Path],
        work_width: float = 200,
        work_height: float = 200,
        feed_rate: float = 1000,
        seed: Optional[int] = None,
    ) -> str:
        """Convert a raster image directly to G-code.

        Args:
            input_path: Input image path or numpy array
            output_path: Output G-code file path
            work_width: Work area width in mm
            work_height: Work area height in mm
            feed_rate: Movement speed in mm/min
            seed: Random seed for reproducible effects

        Returns:
            G-code as string
        """
        # Use same pipeline but output G-code
        self.config.output_width = work_width
        self.config.output_height = work_height

        result = self.convert(input_path, seed=seed)

        gcode = paths_to_gcode(
            result.paths,
            width=work_width,
            height=work_height,
            feed_rate=feed_rate,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(gcode)

        return gcode


def convert(
    input_path: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[ConversionConfig] = None,
    seed: Optional[int] = None,
) -> ConversionResult:
    """Convenience function for quick conversion.

    Args:
        input_path: Input image path or numpy array
        output_path: Output SVG path (optional)
        config: Conversion configuration (optional)
        seed: Random seed for reproducible effects

    Returns:
        ConversionResult with SVG, paths, and statistics

    Example:
        # Quick conversion with defaults
        result = convert("drawing.png", "output.svg")

        # With custom settings
        from raster2vector import ConversionConfig, SketchStyle
        config = ConversionConfig(sketch_style=SketchStyle.SKETCHY)
        result = convert("drawing.png", "output.svg", config=config)
    """
    converter = RasterToVectorConverter(config)
    return converter.convert(input_path, output_path, seed)
