"""Web-based GUI using Gradio for raster2vector."""

import tempfile
import time
import threading
from pathlib import Path
from typing import Optional, Tuple, Generator
import numpy as np

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from .config import (
    ConversionConfig,
    SketchStyle,
    ThresholdMethod,
    PaperSize,
    PAPER_DIMENSIONS,
    get_paper_size,
)
from .core import RasterToVectorConverter


# Paper size options for web UI dropdown
PAPER_CHOICES = [
    "Custom",
    "A0 (841×1189 mm)",
    "A1 (594×841 mm)",
    "A2 (420×594 mm)",
    "A3 (297×420 mm)",
    "A4 (210×297 mm)",
    "A5 (148×210 mm)",
    "A6 (105×148 mm)",
    "Letter (216×279 mm)",
    "Legal (216×356 mm)",
    "Tabloid (279×432 mm)",
    "Postcard (100×148 mm)",
    "Square 100mm",
    "Square 150mm",
    "Square 200mm",
]

PAPER_SIZE_MAP = {
    "A0 (841×1189 mm)": PaperSize.A0,
    "A1 (594×841 mm)": PaperSize.A1,
    "A2 (420×594 mm)": PaperSize.A2,
    "A3 (297×420 mm)": PaperSize.A3,
    "A4 (210×297 mm)": PaperSize.A4,
    "A5 (148×210 mm)": PaperSize.A5,
    "A6 (105×148 mm)": PaperSize.A6,
    "Letter (216×279 mm)": PaperSize.LETTER,
    "Legal (216×356 mm)": PaperSize.LEGAL,
    "Tabloid (279×432 mm)": PaperSize.TABLOID,
    "Postcard (100×148 mm)": PaperSize.POSTCARD,
    "Square 100mm": PaperSize.SQUARE_100,
    "Square 150mm": PaperSize.SQUARE_150,
    "Square 200mm": PaperSize.SQUARE_200,
}


def check_gradio():
    """Check if Gradio is available."""
    if not GRADIO_AVAILABLE:
        raise ImportError(
            "Gradio is not installed. Install it with: pip install gradio\n"
            "Or install raster2vector with web UI support: pip install raster2vector[web]"
        )


def convert_image(
    image: np.ndarray,
    style: str,
    paper_size: str,
    landscape: bool,
    width_mm: str,
    height_mm: str,
    stroke_width: float,
    threshold_method: str,
    invert: bool,
    denoise: bool,
    simplify: float,
    curve_fitting: bool,
    line_sorting: bool,
    wobble: float,
    preserve_thickness: bool,
    stroke_spacing: float,
    min_thickness: float,
    hatching: bool,
    hatch_angle: float,
    hatch_spacing: float,
    cross_hatch: bool,
    seed: str,
) -> Generator[Tuple[Optional[str], str], None, None]:
    """Convert an image to SVG with progress updates.

    Yields:
        Tuples of (SVG file path or None, status message)
    """
    if image is None:
        yield None, "Please upload an image first."
        return

    # Step names for display
    STEP_NAMES = {
        1: "Preprocessing image",
        2: "Extracting skeleton",
        3: "Analyzing skeleton",
        4: "Tracing paths",
        5: "Merging endpoints",
        6: "Processing thickness",
        7: "Sorting paths",
        8: "Applying effects",
        9: "Creating SVG",
        10: "Finalizing",
    }

    # Estimated relative weights for each step (will be updated with actual times)
    # These are initial guesses - heavier steps get higher weights
    STEP_WEIGHTS = {
        1: 1.0,   # Preprocessing
        2: 2.0,   # Skeleton extraction
        3: 0.5,   # Analyzing
        4: 1.5,   # Tracing
        5: 2.0,   # Merging (can be slow)
        6: 1.5,   # Thickness
        7: 2.0,   # Sorting (can be slow)
        8: 1.5,   # Effects
        9: 1.0,   # SVG creation
        10: 0.5,  # Finalizing
    }

    # Progress tracking state
    progress_state = {
        "step_name": "Starting...",
        "step_num": 0,
        "prev_step_num": 0,
        "total_steps": 10,
        "percent": 0.0,
        "start_time": time.time(),
        "step_start_time": time.time(),
        "step_times": {},  # step_num -> duration
        "completed_steps": [],
    }

    def format_time(seconds: float) -> str:
        """Format seconds into human readable string."""
        if seconds < 0.1:
            return "<0.1s"
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"

    def estimate_step_remaining(step_num: int, step_elapsed: float) -> float:
        """Estimate remaining time for current step."""
        # If we have historical data for this step type, use it
        if step_num in progress_state["step_times"]:
            historical = progress_state["step_times"][step_num]
            # Estimate based on historical, but don't go below current elapsed
            return max(0, historical - step_elapsed)

        # Otherwise, use average of completed steps scaled by weight
        completed = progress_state["completed_steps"]
        if completed:
            # Calculate average time per weight unit
            total_time = sum(progress_state["step_times"].get(s, 0) for s in completed)
            total_weight = sum(STEP_WEIGHTS.get(s, 1.0) for s in completed)
            if total_weight > 0:
                time_per_weight = total_time / total_weight
                current_weight = STEP_WEIGHTS.get(step_num, 1.0)
                estimated_duration = time_per_weight * current_weight
                return max(0, estimated_duration - step_elapsed)

        # No data yet - assume current step will take 2x elapsed so far
        return step_elapsed

    def estimate_remaining_steps(current_step: int) -> float:
        """Estimate total time for remaining steps after current."""
        remaining_steps = [s for s in range(current_step + 1, 11)]
        if not remaining_steps:
            return 0.0

        completed = progress_state["completed_steps"]
        if completed:
            # Use actual timing data to estimate
            total_time = sum(progress_state["step_times"].get(s, 0) for s in completed)
            total_weight = sum(STEP_WEIGHTS.get(s, 1.0) for s in completed)
            if total_weight > 0:
                time_per_weight = total_time / total_weight
                remaining_weight = sum(STEP_WEIGHTS.get(s, 1.0) for s in remaining_steps)
                return time_per_weight * remaining_weight

        # No completed steps yet - can't estimate
        return 0.0

    def build_progress_message() -> str:
        """Build the progress status message."""
        now = time.time()
        elapsed = now - progress_state["start_time"]
        step_elapsed = now - progress_state["step_start_time"]
        step_name = progress_state["step_name"]
        step_num = progress_state["step_num"]
        total = progress_state["total_steps"]

        # Calculate overall progress based on completed steps and current step progress
        completed_weight = sum(STEP_WEIGHTS.get(s, 1.0) for s in progress_state["completed_steps"])
        total_weight = sum(STEP_WEIGHTS.get(s, 1.0) for s in range(1, total + 1))

        # Estimate progress within current step (assume linear within step)
        current_step_weight = STEP_WEIGHTS.get(step_num, 1.0)
        step_remaining = estimate_step_remaining(step_num, step_elapsed)
        step_total_estimate = step_elapsed + step_remaining
        if step_total_estimate > 0:
            step_progress = min(0.95, step_elapsed / step_total_estimate)  # Cap at 95% until done
        else:
            step_progress = 0.5

        current_partial_weight = current_step_weight * step_progress
        percent = ((completed_weight + current_partial_weight) / total_weight) * 100 if total_weight > 0 else 0

        # Create progress bar
        bar_width = 20
        filled = int(bar_width * percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        msg = f"Converting...\n\n"
        msg += f"[{bar}] {percent:.0f}%\n\n"

        # Current step info
        msg += f"Step {step_num}/{total}: {step_name}\n"
        msg += f"  Step time: {format_time(step_elapsed)}"

        # Estimate for current step
        if step_remaining > 0:
            msg += f" (est. {format_time(step_remaining)} remaining)\n"
        else:
            msg += "\n"

        msg += f"\nTotal elapsed: {format_time(elapsed)}\n"

        # Overall time estimate
        if step_num > 0:
            remaining_current = step_remaining
            remaining_future = estimate_remaining_steps(step_num)
            total_remaining = remaining_current + remaining_future

            if total_remaining > 0:
                msg += f"Estimated remaining: {format_time(total_remaining)}\n"

        # Show completed step times
        if progress_state["completed_steps"]:
            msg += f"\nCompleted steps:\n"
            for s in progress_state["completed_steps"][-5:]:  # Show last 5
                step_time = progress_state["step_times"].get(s, 0)
                msg += f"  {s}. {STEP_NAMES.get(s, 'Step')}: {format_time(step_time)}\n"

        return msg

    def progress_callback(step_name: str, step_num: int, total_steps: int, percent: float):
        """Callback to receive progress updates from converter."""
        now = time.time()

        # Record time for previous step if it changed
        prev_step = progress_state["prev_step_num"]
        if prev_step > 0 and step_num != prev_step:
            step_duration = now - progress_state["step_start_time"]
            progress_state["step_times"][prev_step] = step_duration
            progress_state["completed_steps"].append(prev_step)
            progress_state["step_start_time"] = now

        progress_state["step_name"] = step_name
        progress_state["step_num"] = step_num
        progress_state["prev_step_num"] = step_num
        progress_state["total_steps"] = total_steps
        progress_state["percent"] = percent

    try:
        # Handle paper size
        if paper_size != "Custom" and paper_size in PAPER_SIZE_MAP:
            paper = PAPER_SIZE_MAP[paper_size]
            output_width, output_height = get_paper_size(paper, landscape=landscape)
        else:
            # Parse custom dimensions
            output_width = float(width_mm) if width_mm.strip() else None
            output_height = float(height_mm) if height_mm.strip() else None
            if landscape and output_width and output_height:
                output_width, output_height = output_height, output_width

        # Parse seed
        seed_value = int(seed) if seed.strip() else None

        # Map style
        style_map = {
            "Clean": SketchStyle.CLEAN,
            "Natural": SketchStyle.NATURAL,
            "Sketchy": SketchStyle.SKETCHY,
            "Rough": SketchStyle.ROUGH,
        }

        threshold_map = {
            "Otsu (automatic)": ThresholdMethod.OTSU,
            "Adaptive": ThresholdMethod.ADAPTIVE,
            "Fixed": ThresholdMethod.FIXED,
        }

        # Build config
        config = ConversionConfig(
            sketch_style=style_map.get(style, SketchStyle.NATURAL),
            output_width=output_width,
            output_height=output_height,
            stroke_width=stroke_width,
            threshold_method=threshold_map.get(threshold_method, ThresholdMethod.OTSU),
            invert=invert,
            denoise=denoise,
            simplify_tolerance=simplify,
            curve_fitting=curve_fitting,
            line_sorting=line_sorting,
            wobble_amplitude=wobble,
            preserve_thickness=preserve_thickness,
            stroke_spacing=stroke_spacing,
            min_width_for_multi=min_thickness,
            fill_hatching=hatching,
            hatch_angle=hatch_angle,
            hatch_spacing=hatch_spacing,
            cross_hatch=cross_hatch,
        )

        # Yield initial progress
        yield None, build_progress_message()

        # Run conversion in a thread so we can yield progress updates
        result_holder = {"result": None, "error": None}

        def run_conversion():
            try:
                converter = RasterToVectorConverter(config)
                result_holder["result"] = converter.convert(
                    image,
                    seed=seed_value,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                result_holder["error"] = e

        thread = threading.Thread(target=run_conversion)
        thread.start()

        # Yield progress updates while conversion runs
        last_message = ""
        while thread.is_alive():
            thread.join(timeout=0.1)  # Check every 100ms
            current_message = build_progress_message()
            if current_message != last_message:
                last_message = current_message
                yield None, current_message

        # Yield one final progress update before showing results
        if progress_state["percent"] > 0:
            yield None, build_progress_message()

        # Check for errors
        if result_holder["error"]:
            raise result_holder["error"]

        result = result_holder["result"]

        # Save to temp file
        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / "output.svg"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.svg.content)

        # Build final stats message
        elapsed = time.time() - progress_state["start_time"]
        stats = result.stats
        msg = (
            f"Conversion successful!\n\n"
            f"Completed in {format_time(elapsed)}\n\n"
            f"Source: {stats['source_width']}x{stats['source_height']} px\n"
            f"Paths: {stats['paths_output']}\n"
            f"Total length: {stats['total_path_length']:.0f} units\n"
        )
        if stats.get('thickness_preserved'):
            msg += "Thickness preservation: enabled\n"
        if stats.get('fill_hatching'):
            msg += "Fill hatching: enabled\n"

        yield str(output_path), msg

    except Exception as e:
        yield None, f"Error: {str(e)}"


def get_logo_path() -> Optional[str]:
    """Get the path to the logo file if it exists."""
    import os
    # Check for logo in assets folder
    module_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(module_dir, "assets", "logo.png")
    if os.path.exists(logo_path):
        return logo_path
    return None


def create_web_ui() -> "gr.Blocks":
    """Create the Gradio web interface.

    Returns:
        Gradio Blocks app
    """
    check_gradio()

    with gr.Blocks(
        title="Raster2Vector - Pen Plotter Converter",
        theme=gr.themes.Soft(),
    ) as app:
        # Header with logo
        with gr.Row():
            with gr.Column(scale=1):
                logo_path = get_logo_path()
                if logo_path:
                    gr.Image(
                        value=logo_path,
                        show_label=False,
                        show_download_button=False,
                        show_share_button=False,
                        container=False,
                        height=150,
                    )
            with gr.Column(scale=3):
                gr.Markdown(
                    """
                    # Raster2Vector
                    ### Convert line drawings to vectors for pen plotters

                    Upload a raster image (PNG, JPG) and convert it to single-line SVG paths
                    suitable for pen plotters, laser cutters, and CNC machines.
                    """
                )

        with gr.Row():
            # Left column - Input and settings
            with gr.Column(scale=1):
                # Image input
                input_image = gr.Image(
                    label="Input Image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )

                # Presets
                with gr.Accordion("Quick Presets", open=True):
                    preset_btns = gr.Radio(
                        choices=["Default", "Pen Plotter", "Laser Cutter", "Thick Lines", "Hatching"],
                        value="Default",
                        label="Preset",
                    )

                # Basic settings
                with gr.Accordion("Basic Settings", open=True):
                    style = gr.Dropdown(
                        choices=["Clean", "Natural", "Sketchy", "Rough"],
                        value="Natural",
                        label="Style",
                    )

                    paper_size = gr.Dropdown(
                        choices=PAPER_CHOICES,
                        value="Custom",
                        label="Paper Size",
                    )

                    with gr.Row():
                        landscape = gr.Checkbox(
                            label="Landscape",
                            value=False,
                        )

                    with gr.Row():
                        width_mm = gr.Textbox(
                            label="Width (mm)",
                            placeholder="auto",
                            value="",
                        )
                        height_mm = gr.Textbox(
                            label="Height (mm)",
                            placeholder="auto",
                            value="",
                        )

                    stroke_width = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        label="Stroke Width",
                    )

                # Preprocessing
                with gr.Accordion("Preprocessing", open=False):
                    threshold_method = gr.Dropdown(
                        choices=["Otsu (automatic)", "Adaptive", "Fixed"],
                        value="Otsu (automatic)",
                        label="Threshold Method",
                    )
                    invert = gr.Checkbox(label="Invert (for white on dark)", value=False)
                    denoise = gr.Checkbox(label="Denoise", value=True)

                # Style settings
                with gr.Accordion("Style & Effects", open=False):
                    simplify = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=1.0,
                        step=0.1,
                        label="Simplification",
                    )
                    wobble = gr.Slider(
                        minimum=0,
                        maximum=2,
                        value=0.3,
                        step=0.05,
                        label="Wobble (hand-drawn effect)",
                    )
                    curve_fitting = gr.Checkbox(label="Fit Bezier curves", value=True)
                    line_sorting = gr.Checkbox(label="Optimize path order", value=True)

                # Thickness & Hatching
                with gr.Accordion("Thickness & Hatching", open=False):
                    preserve_thickness = gr.Checkbox(
                        label="Preserve stroke thickness",
                        value=False,
                    )
                    stroke_spacing = gr.Slider(
                        minimum=0.5,
                        maximum=5,
                        value=1.5,
                        step=0.1,
                        label="Stroke Spacing",
                    )
                    min_thickness = gr.Slider(
                        minimum=2,
                        maximum=10,
                        value=4.0,
                        step=0.5,
                        label="Min Thickness for Multi-stroke",
                    )

                    gr.Markdown("---")

                    hatching = gr.Checkbox(
                        label="Fill hatching for solid areas",
                        value=False,
                    )
                    hatch_angle = gr.Slider(
                        minimum=0,
                        maximum=90,
                        value=45,
                        step=5,
                        label="Hatch Angle",
                    )
                    hatch_spacing = gr.Slider(
                        minimum=0.5,
                        maximum=5,
                        value=2.0,
                        step=0.1,
                        label="Hatch Spacing",
                    )
                    cross_hatch = gr.Checkbox(label="Cross-hatch", value=False)

                # Seed
                with gr.Accordion("Reproducibility", open=False):
                    seed = gr.Textbox(
                        label="Random Seed (optional)",
                        placeholder="Leave empty for random",
                        value="",
                    )

                # Convert button
                convert_btn = gr.Button("Convert to SVG", variant="primary", size="lg")

            # Right column - Output
            with gr.Column(scale=1):
                output_file = gr.File(label="Download SVG")
                status_text = gr.Textbox(
                    label="Status",
                    lines=8,
                    interactive=False,
                )

                gr.Markdown(
                    """
                    ### Tips
                    - Use **high-contrast** line drawings for best results
                    - Enable **Invert** for white lines on dark backgrounds
                    - Use **Adaptive** threshold for uneven lighting
                    - Enable **Thickness preservation** to maintain visual weight
                    - Use **Hatching** to fill solid areas with pen strokes
                    """
                )

        # Preset handlers
        def apply_preset(preset):
            # Returns: style, paper_size, landscape, width_mm, height_mm, stroke_width,
            #          preserve_thickness, stroke_spacing, min_thickness,
            #          hatching, hatch_angle, hatch_spacing, cross_hatch
            if preset == "Default":
                return "Natural", "Custom", False, "", "", 0.5, False, 1.5, 4.0, False, 45, 2.0, False
            elif preset == "Pen Plotter":
                return "Natural", "A4 (210×297 mm)", False, "", "", 0.4, False, 1.5, 4.0, False, 45, 2.0, False
            elif preset == "Laser Cutter":
                return "Clean", "Custom", False, "300", "200", 0.1, False, 1.5, 4.0, False, 45, 2.0, False
            elif preset == "Thick Lines":
                return "Natural", "Custom", False, "", "", 0.5, True, 1.5, 4.0, False, 45, 2.0, False
            elif preset == "Hatching":
                return "Natural", "Custom", False, "", "", 0.5, True, 1.5, 4.0, True, 45, 2.0, False
            return "Natural", "Custom", False, "", "", 0.5, False, 1.5, 4.0, False, 45, 2.0, False

        preset_btns.change(
            fn=apply_preset,
            inputs=[preset_btns],
            outputs=[
                style, paper_size, landscape, width_mm, height_mm, stroke_width,
                preserve_thickness, stroke_spacing, min_thickness,
                hatching, hatch_angle, hatch_spacing, cross_hatch
            ],
        )

        # Convert handler
        convert_btn.click(
            fn=convert_image,
            inputs=[
                input_image,
                style,
                paper_size,
                landscape,
                width_mm,
                height_mm,
                stroke_width,
                threshold_method,
                invert,
                denoise,
                simplify,
                curve_fitting,
                line_sorting,
                wobble,
                preserve_thickness,
                stroke_spacing,
                min_thickness,
                hatching,
                hatch_angle,
                hatch_spacing,
                cross_hatch,
                seed,
            ],
            outputs=[output_file, status_text],
        )

    return app


def launch_web_ui(
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
):
    """Launch the web UI.

    Args:
        share: Create a public link
        server_name: Server hostname (0.0.0.0 for network access)
        server_port: Server port
    """
    check_gradio()
    app = create_web_ui()
    app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
    )


def main():
    """Entry point for web UI."""
    launch_web_ui()


if __name__ == "__main__":
    main()
