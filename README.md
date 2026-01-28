# Raster2Vector

Convert raster line drawings to vectors for pen plotters.

Raster2Vector takes line drawings in raster format (PNG, JPG, etc.) and converts them to single-line vector paths suitable for pen plotters, laser cutters, and other CNC machines. Unlike typical vectorization tools that create outlines, this tool extracts the **centerlines** of strokes, producing paths that can be drawn with a single pen stroke.

## Features

- **Centerline extraction**: Produces single-line paths, not outlines
- **Sketchy effects**: Optional hand-drawn look with natural wobble and variation
- **Bezier curve fitting**: Smooth curves that plot beautifully
- **Path optimization**: Sorts paths to minimize pen travel distance
- **Multiple output formats**: SVG for plotters, G-code for CNC machines
- **Configurable**: Fine-tune every aspect of the conversion

## Installation

```bash
pip install -e .
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Command Line

```bash
# Basic conversion
raster2vector drawing.png output.svg

# With sketchy hand-drawn effect
raster2vector --style sketchy drawing.png output.svg

# Specify output size for pen plotter (in mm)
raster2vector --width 200 --height 200 drawing.png output.svg

# Clean output (no hand-drawn effects)
raster2vector --style clean drawing.png output.svg

# Output G-code for CNC plotter
raster2vector --gcode --width 100 --height 100 drawing.png output.gcode
```

### Python API

```python
from raster2vector import convert, ConversionConfig, SketchStyle

# Simple conversion
result = convert("drawing.png", "output.svg")

# With custom configuration
config = ConversionConfig(
    sketch_style=SketchStyle.SKETCHY,
    output_width=200,  # mm
    output_height=200,  # mm
    stroke_width=0.4,
)
result = convert("drawing.png", "output.svg", config=config)

# Access conversion statistics
print(f"Generated {result.stats['paths_output']} paths")
print(f"Total path length: {result.stats['total_path_length']:.1f} units")
```

## Style Presets

| Style | Description |
|-------|-------------|
| `clean` | No hand-drawn effects, smooth curves |
| `natural` | Subtle variation, like careful hand drawing (default) |
| `sketchy` | More variation, like quick sketches |
| `rough` | Maximum variation, very hand-drawn look |

## How It Works

1. **Preprocessing**: Load image, convert to grayscale, apply thresholding
2. **Skeletonization**: Extract single-pixel-wide centerlines using morphological thinning
3. **Path Tracing**: Convert skeleton pixels to ordered polyline paths
4. **Curve Fitting**: Fit smooth Bezier curves to the paths
5. **Sketch Effects**: Add optional wobble and variation for hand-drawn look
6. **Optimization**: Sort paths to minimize pen travel distance
7. **Output**: Generate SVG or G-code

## CLI Options

```
Usage: raster2vector [OPTIONS] INPUT_FILE [OUTPUT_FILE]

Options:
  -s, --style [clean|natural|sketchy|rough]
                                  Drawing style preset (default: natural)
  -w, --width FLOAT               Output width in mm
  -h, --height FLOAT              Output height in mm
  --stroke-width FLOAT            Stroke width in output SVG (default: 0.5)
  --stroke-color TEXT             Stroke color (default: black)
  -t, --threshold [otsu|adaptive|fixed]
                                  Thresholding method (default: otsu)
  --threshold-value INTEGER       Fixed threshold value 0-255 (default: 128)
  --invert / --no-invert          Invert image (for white on dark)
  --denoise / --no-denoise        Apply denoising (default: yes)
  --min-path-length INTEGER       Minimum path length in pixels (default: 5)
  --simplify FLOAT                Path simplification tolerance (default: 1.0)
  --no-curves                     Output polylines instead of Bezier curves
  --no-sort                       Don't sort paths to minimize pen travel
  --wobble FLOAT                  Override wobble amplitude (0-5)
  --seed INTEGER                  Random seed for reproducible output
  --gcode                         Output G-code instead of SVG
  --feed-rate FLOAT               G-code feed rate in mm/min (default: 1000)
  -v, --verbose                   Print conversion statistics
  --version                       Show the version and exit.
  --help                          Show this message and exit.
```

## Configuration Options

### ConversionConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sketch_style` | SketchStyle | NATURAL | Style preset |
| `threshold_method` | ThresholdMethod | OTSU | Binarization method |
| `threshold_value` | int | 128 | Fixed threshold value |
| `invert` | bool | False | Invert image colors |
| `denoise` | bool | True | Apply denoising |
| `min_path_length` | int | 5 | Minimum path points |
| `wobble_amplitude` | float | 0.5 | Hand-drawn wobble amount |
| `wobble_frequency` | float | 0.1 | Wobble variation frequency |
| `corner_rounding` | float | 0.3 | Round sharp corners |
| `line_variation` | float | 0.2 | Random position variation |
| `simplify_tolerance` | float | 1.0 | Path simplification |
| `curve_fitting` | bool | True | Fit Bezier curves |
| `output_width` | float | None | Output width in mm |
| `output_height` | float | None | Output height in mm |
| `stroke_width` | float | 0.5 | SVG stroke width |
| `stroke_color` | str | "black" | SVG stroke color |
| `line_sorting` | bool | True | Optimize path order |
| `merge_nearby_endpoints` | bool | True | Connect nearby paths |

## Tips for Best Results

1. **Input images**: Use high-contrast line drawings with clear strokes
2. **Threshold method**: Use `otsu` for most images, `adaptive` for uneven lighting
3. **Invert option**: Use `--invert` for white lines on dark backgrounds
4. **Simplify**: Increase `--simplify` value to reduce path complexity
5. **Style**: Start with `natural`, adjust wobble/variation if needed

## License

MIT License
