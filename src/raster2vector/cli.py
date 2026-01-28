"""Command-line interface for raster2vector."""

import click
from pathlib import Path
import sys

from .config import ConversionConfig, SketchStyle, ThresholdMethod
from .core import RasterToVectorConverter


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file", type=click.Path(path_type=Path), required=False)
@click.option(
    "--style",
    "-s",
    type=click.Choice(["clean", "natural", "sketchy", "rough"]),
    default="natural",
    help="Drawing style preset (default: natural)",
)
@click.option(
    "--width",
    "-w",
    type=float,
    default=None,
    help="Output width in mm (default: use source dimensions in px)",
)
@click.option(
    "--height",
    "-h",
    type=float,
    default=None,
    help="Output height in mm (default: use source dimensions in px)",
)
@click.option(
    "--stroke-width",
    type=float,
    default=0.5,
    help="Stroke width in output SVG (default: 0.5)",
)
@click.option(
    "--stroke-color",
    type=str,
    default="black",
    help="Stroke color (default: black)",
)
@click.option(
    "--threshold",
    "-t",
    type=click.Choice(["otsu", "adaptive", "fixed"]),
    default="otsu",
    help="Thresholding method (default: otsu)",
)
@click.option(
    "--threshold-value",
    type=int,
    default=128,
    help="Fixed threshold value 0-255 (default: 128)",
)
@click.option(
    "--invert/--no-invert",
    default=False,
    help="Invert image (for white lines on dark background)",
)
@click.option(
    "--denoise/--no-denoise",
    default=True,
    help="Apply denoising (default: yes)",
)
@click.option(
    "--min-path-length",
    type=int,
    default=5,
    help="Minimum path length in pixels (default: 5)",
)
@click.option(
    "--simplify",
    type=float,
    default=1.0,
    help="Path simplification tolerance (default: 1.0, 0=none)",
)
@click.option(
    "--no-curves",
    is_flag=True,
    default=False,
    help="Output polylines instead of Bezier curves",
)
@click.option(
    "--no-sort",
    is_flag=True,
    default=False,
    help="Don't sort paths to minimize pen travel",
)
@click.option(
    "--wobble",
    type=float,
    default=None,
    help="Override wobble amplitude (0-5, default: based on style)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible output",
)
@click.option(
    "--gcode",
    is_flag=True,
    default=False,
    help="Output G-code instead of SVG",
)
@click.option(
    "--feed-rate",
    type=float,
    default=1000,
    help="G-code feed rate in mm/min (default: 1000)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print conversion statistics",
)
@click.version_option(package_name="raster2vector")
def main(
    input_file: Path,
    output_file: Path,
    style: str,
    width: float,
    height: float,
    stroke_width: float,
    stroke_color: str,
    threshold: str,
    threshold_value: int,
    invert: bool,
    denoise: bool,
    min_path_length: int,
    simplify: float,
    no_curves: bool,
    no_sort: bool,
    wobble: float,
    seed: int,
    gcode: bool,
    feed_rate: float,
    verbose: bool,
):
    """Convert raster line drawings to vectors for pen plotters.

    INPUT_FILE: Path to input image (PNG, JPG, etc.)

    OUTPUT_FILE: Path for output file (default: input_name.svg or .gcode)

    Examples:

        # Basic conversion with natural style
        raster2vector drawing.png output.svg

        # Sketchy style for hand-drawn look
        raster2vector -s sketchy drawing.png

        # Specify output size for pen plotter
        raster2vector -w 200 -h 200 drawing.png output.svg

        # Output G-code for CNC plotter
        raster2vector --gcode -w 100 -h 100 drawing.png output.gcode

        # Clean output (no hand-drawn effects)
        raster2vector -s clean drawing.png

    """
    # Determine output file
    if output_file is None:
        suffix = ".gcode" if gcode else ".svg"
        output_file = input_file.with_suffix(suffix)

    # Build configuration
    style_map = {
        "clean": SketchStyle.CLEAN,
        "natural": SketchStyle.NATURAL,
        "sketchy": SketchStyle.SKETCHY,
        "rough": SketchStyle.ROUGH,
    }

    threshold_map = {
        "otsu": ThresholdMethod.OTSU,
        "adaptive": ThresholdMethod.ADAPTIVE,
        "fixed": ThresholdMethod.FIXED,
    }

    config = ConversionConfig(
        sketch_style=style_map[style],
        output_width=width,
        output_height=height,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        threshold_method=threshold_map[threshold],
        threshold_value=threshold_value,
        invert=invert,
        denoise=denoise,
        min_path_length=min_path_length,
        simplify_tolerance=simplify,
        curve_fitting=not no_curves,
        line_sorting=not no_sort,
    )

    # Override wobble if specified
    if wobble is not None:
        config.wobble_amplitude = wobble

    # Create converter and run
    converter = RasterToVectorConverter(config)

    try:
        if gcode:
            if width is None or height is None:
                click.echo("Error: --width and --height are required for G-code output", err=True)
                sys.exit(1)

            click.echo(f"Converting {input_file} to G-code...")
            converter.convert_to_gcode(
                input_file,
                output_file,
                work_width=width,
                work_height=height,
                feed_rate=feed_rate,
                seed=seed,
            )
            click.echo(f"Saved G-code to {output_file}")
        else:
            click.echo(f"Converting {input_file}...")
            result = converter.convert(input_file, output_file, seed=seed)

            click.echo(f"Saved SVG to {output_file}")

            if verbose:
                click.echo("\nConversion Statistics:")
                click.echo(f"  Source size: {result.stats['source_width']}x{result.stats['source_height']} px")
                click.echo(f"  Skeleton pixels: {result.stats['skeleton_pixels']}")
                click.echo(f"  Endpoints: {result.stats['endpoints']}")
                click.echo(f"  Junctions: {result.stats['junctions']}")
                click.echo(f"  Paths output: {result.stats['paths_output']}")
                click.echo(f"  Total path length: {result.stats['total_path_length']:.1f} units")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during conversion: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
