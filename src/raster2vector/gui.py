"""Graphical user interface for raster2vector."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import threading
import io
import tempfile
import os
from typing import Optional

from .config import (
    ConversionConfig,
    SketchStyle,
    ThresholdMethod,
    PaperSize,
    PAPER_DIMENSIONS,
    get_paper_size,
)
from .core import RasterToVectorConverter


# Paper size display names for GUI
PAPER_SIZE_OPTIONS = [
    ("Custom", None),
    ("A0 (841x1189 mm)", PaperSize.A0),
    ("A1 (594x841 mm)", PaperSize.A1),
    ("A2 (420x594 mm)", PaperSize.A2),
    ("A3 (297x420 mm)", PaperSize.A3),
    ("A4 (210x297 mm)", PaperSize.A4),
    ("A5 (148x210 mm)", PaperSize.A5),
    ("A6 (105x148 mm)", PaperSize.A6),
    ("Letter (216x279 mm)", PaperSize.LETTER),
    ("Legal (216x356 mm)", PaperSize.LEGAL),
    ("Tabloid (279x432 mm)", PaperSize.TABLOID),
    ("Postcard (100x148 mm)", PaperSize.POSTCARD),
    ("Square 100mm", PaperSize.SQUARE_100),
    ("Square 150mm", PaperSize.SQUARE_150),
    ("Square 200mm", PaperSize.SQUARE_200),
]


class RasterToVectorGUI:
    """Main GUI application for raster to vector conversion."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Raster2Vector - Pen Plotter Converter")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

        # State
        self.input_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.input_image: Optional[Image.Image] = None
        self.output_svg: Optional[str] = None
        self.is_converting = False

        # Variables for controls
        self._init_variables()

        # Build UI
        self._build_ui()

        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _init_variables(self):
        """Initialize tkinter variables for all controls."""
        # Style
        self.style_var = tk.StringVar(value="natural")

        # Paper size
        self.paper_size_var = tk.StringVar(value="Custom")
        self.landscape_var = tk.BooleanVar(value=False)

        # Output dimensions
        self.width_var = tk.StringVar(value="")
        self.height_var = tk.StringVar(value="")
        self.stroke_width_var = tk.StringVar(value="0.5")
        self.stroke_color_var = tk.StringVar(value="black")

        # Preprocessing
        self.threshold_var = tk.StringVar(value="otsu")
        self.threshold_value_var = tk.IntVar(value=128)
        self.invert_var = tk.BooleanVar(value=False)
        self.denoise_var = tk.BooleanVar(value=True)

        # Path processing
        self.min_path_length_var = tk.IntVar(value=5)
        self.simplify_var = tk.DoubleVar(value=1.0)
        self.curve_fitting_var = tk.BooleanVar(value=True)
        self.line_sorting_var = tk.BooleanVar(value=True)

        # Sketchy effects
        self.wobble_var = tk.DoubleVar(value=0.3)
        self.line_variation_var = tk.DoubleVar(value=0.15)
        self.corner_rounding_var = tk.DoubleVar(value=0.2)

        # Thickness preservation
        self.preserve_thickness_var = tk.BooleanVar(value=False)
        self.stroke_spacing_var = tk.DoubleVar(value=1.5)
        self.min_thickness_var = tk.DoubleVar(value=4.0)

        # Hatching
        self.hatching_var = tk.BooleanVar(value=False)
        self.hatch_angle_var = tk.DoubleVar(value=45.0)
        self.hatch_spacing_var = tk.DoubleVar(value=2.0)
        self.cross_hatch_var = tk.BooleanVar(value=False)

        # Seed
        self.use_seed_var = tk.BooleanVar(value=False)
        self.seed_var = tk.IntVar(value=42)

    def _build_ui(self):
        """Build the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Top toolbar
        self._build_toolbar(main_frame)

        # Left panel - Controls
        self._build_controls_panel(main_frame)

        # Center panel - Image preview
        self._build_preview_panel(main_frame)

        # Bottom - Status bar
        self._build_status_bar(main_frame)

    def _build_toolbar(self, parent):
        """Build the top toolbar."""
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        # Open button
        self.open_btn = ttk.Button(toolbar, text="Open Image", command=self._open_file)
        self.open_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Convert button
        self.convert_btn = ttk.Button(
            toolbar, text="Convert", command=self._convert, state=tk.DISABLED
        )
        self.convert_btn.pack(side=tk.LEFT, padx=5)

        # Save button
        self.save_btn = ttk.Button(
            toolbar, text="Save SVG", command=self._save_file, state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Presets dropdown
        ttk.Label(toolbar, text="Preset:").pack(side=tk.LEFT, padx=(0, 5))
        preset_combo = ttk.Combobox(
            toolbar,
            values=["Default", "Pen Plotter (A4)", "Laser Cutter", "Thick Lines", "Hatching"],
            state="readonly",
            width=15,
        )
        preset_combo.set("Default")
        preset_combo.pack(side=tk.LEFT, padx=(0, 10))
        preset_combo.bind("<<ComboboxSelected>>", self._apply_preset)
        self.preset_combo = preset_combo

        # Progress bar
        self.progress = ttk.Progressbar(toolbar, mode="indeterminate", length=150)
        self.progress.pack(side=tk.RIGHT, padx=(10, 0))

    def _build_controls_panel(self, parent):
        """Build the left controls panel with tabs."""
        # Controls frame with fixed width
        controls_frame = ttk.LabelFrame(parent, text="Settings", padding="5")
        controls_frame.grid(row=1, column=0, sticky="ns", padx=(0, 10))

        # Notebook for tabs
        notebook = ttk.Notebook(controls_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Basic settings
        basic_tab = ttk.Frame(notebook, padding="10")
        notebook.add(basic_tab, text="Basic")
        self._build_basic_tab(basic_tab)

        # Tab 2: Style settings
        style_tab = ttk.Frame(notebook, padding="10")
        notebook.add(style_tab, text="Style")
        self._build_style_tab(style_tab)

        # Tab 3: Thickness & Hatching
        thickness_tab = ttk.Frame(notebook, padding="10")
        notebook.add(thickness_tab, text="Thickness")
        self._build_thickness_tab(thickness_tab)

        # Tab 4: Advanced
        advanced_tab = ttk.Frame(notebook, padding="10")
        notebook.add(advanced_tab, text="Advanced")
        self._build_advanced_tab(advanced_tab)

    def _build_basic_tab(self, parent):
        """Build basic settings tab."""
        row = 0

        # Style preset
        ttk.Label(parent, text="Style:").grid(row=row, column=0, sticky="w", pady=2)
        style_combo = ttk.Combobox(
            parent,
            textvariable=self.style_var,
            values=["clean", "natural", "sketchy", "rough"],
            state="readonly",
            width=12,
        )
        style_combo.grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        # Paper size section
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10
        )
        row += 1

        ttk.Label(parent, text="Paper Size:").grid(row=row, column=0, sticky="w", pady=2)
        paper_combo = ttk.Combobox(
            parent,
            textvariable=self.paper_size_var,
            values=[opt[0] for opt in PAPER_SIZE_OPTIONS],
            state="readonly",
            width=18,
        )
        paper_combo.grid(row=row, column=1, sticky="w", pady=2)
        paper_combo.bind("<<ComboboxSelected>>", self._on_paper_size_changed)
        self.paper_combo = paper_combo
        row += 1

        # Landscape checkbox
        ttk.Checkbutton(
            parent,
            text="Landscape orientation",
            variable=self.landscape_var,
            command=self._on_orientation_changed,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        # Custom dimensions
        ttk.Label(parent, text="Custom Size (mm):").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 2)
        )
        row += 1

        size_frame = ttk.Frame(parent)
        size_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(size_frame, text="W:").pack(side=tk.LEFT)
        self.width_entry = ttk.Entry(size_frame, textvariable=self.width_var, width=6)
        self.width_entry.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(size_frame, text="H:").pack(side=tk.LEFT)
        self.height_entry = ttk.Entry(size_frame, textvariable=self.height_var, width=6)
        self.height_entry.pack(side=tk.LEFT)
        row += 1

        # Stroke settings
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10
        )
        row += 1

        ttk.Label(parent, text="Stroke Width:").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=self.stroke_width_var, width=8).grid(
            row=row, column=1, sticky="w", pady=2
        )
        row += 1

        ttk.Label(parent, text="Stroke Color:").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=self.stroke_color_var, width=8).grid(
            row=row, column=1, sticky="w", pady=2
        )
        row += 1

        # Preprocessing
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10
        )
        row += 1

        ttk.Label(parent, text="Threshold:").grid(row=row, column=0, sticky="w", pady=2)
        threshold_combo = ttk.Combobox(
            parent,
            textvariable=self.threshold_var,
            values=["otsu", "adaptive", "fixed"],
            state="readonly",
            width=12,
        )
        threshold_combo.grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        ttk.Checkbutton(parent, text="Invert (white on dark)", variable=self.invert_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        row += 1

        ttk.Checkbutton(parent, text="Denoise", variable=self.denoise_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )

    def _build_style_tab(self, parent):
        """Build style/sketch settings tab."""
        row = 0

        # Wobble
        ttk.Label(parent, text="Wobble Amount:").grid(row=row, column=0, sticky="w", pady=2)
        wobble_scale = ttk.Scale(
            parent, from_=0, to=2, variable=self.wobble_var, orient=tk.HORIZONTAL, length=120
        )
        wobble_scale.grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        # Line variation
        ttk.Label(parent, text="Line Variation:").grid(row=row, column=0, sticky="w", pady=2)
        variation_scale = ttk.Scale(
            parent, from_=0, to=1, variable=self.line_variation_var, orient=tk.HORIZONTAL, length=120
        )
        variation_scale.grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        # Corner rounding
        ttk.Label(parent, text="Corner Rounding:").grid(row=row, column=0, sticky="w", pady=2)
        corner_scale = ttk.Scale(
            parent, from_=0, to=1, variable=self.corner_rounding_var, orient=tk.HORIZONTAL, length=120
        )
        corner_scale.grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        # Separator
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10
        )
        row += 1

        # Path simplification
        ttk.Label(parent, text="Simplification:").grid(row=row, column=0, sticky="w", pady=2)
        simplify_scale = ttk.Scale(
            parent, from_=0, to=5, variable=self.simplify_var, orient=tk.HORIZONTAL, length=120
        )
        simplify_scale.grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        # Min path length
        ttk.Label(parent, text="Min Path Length:").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(parent, from_=1, to=50, textvariable=self.min_path_length_var, width=8).grid(
            row=row, column=1, sticky="w", pady=2
        )
        row += 1

        # Separator
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10
        )
        row += 1

        # Checkboxes
        ttk.Checkbutton(parent, text="Fit Bezier curves", variable=self.curve_fitting_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        row += 1

        ttk.Checkbutton(parent, text="Sort paths (reduce travel)", variable=self.line_sorting_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        row += 1

        # Seed
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10
        )
        row += 1

        seed_frame = ttk.Frame(parent)
        seed_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(seed_frame, text="Use seed:", variable=self.use_seed_var).pack(side=tk.LEFT)
        ttk.Spinbox(seed_frame, from_=0, to=99999, textvariable=self.seed_var, width=8).pack(
            side=tk.LEFT, padx=(5, 0)
        )

    def _build_thickness_tab(self, parent):
        """Build thickness and hatching settings tab."""
        row = 0

        # Thickness preservation section
        ttk.Label(parent, text="Thickness Preservation", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5)
        )
        row += 1

        ttk.Checkbutton(
            parent, text="Preserve stroke thickness", variable=self.preserve_thickness_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        ttk.Label(parent, text="Stroke Spacing:").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Scale(
            parent, from_=0.5, to=5, variable=self.stroke_spacing_var, orient=tk.HORIZONTAL, length=120
        ).grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        ttk.Label(parent, text="Min Thickness:").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Scale(
            parent, from_=2, to=10, variable=self.min_thickness_var, orient=tk.HORIZONTAL, length=120
        ).grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        # Separator
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=15
        )
        row += 1

        # Hatching section
        ttk.Label(parent, text="Fill Hatching", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5)
        )
        row += 1

        ttk.Checkbutton(parent, text="Enable hatching for fills", variable=self.hatching_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        row += 1

        ttk.Label(parent, text="Hatch Angle:").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Scale(
            parent, from_=0, to=90, variable=self.hatch_angle_var, orient=tk.HORIZONTAL, length=120
        ).grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        ttk.Label(parent, text="Hatch Spacing:").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Scale(
            parent, from_=0.5, to=5, variable=self.hatch_spacing_var, orient=tk.HORIZONTAL, length=120
        ).grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        ttk.Checkbutton(parent, text="Cross-hatch", variable=self.cross_hatch_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )

    def _build_advanced_tab(self, parent):
        """Build advanced settings tab."""
        row = 0

        ttk.Label(parent, text="Threshold Value (fixed):").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Scale(
            parent, from_=0, to=255, variable=self.threshold_value_var, orient=tk.HORIZONTAL, length=120
        ).grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        # Info text
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=15
        )
        row += 1

        info_text = """Tips for best results:

• Use high-contrast images
• For white on dark, enable Invert
• Use 'adaptive' threshold for
  uneven lighting
• Increase simplification for
  cleaner output
• Use thickness preservation
  to maintain visual weight"""

        ttk.Label(parent, text=info_text, justify=tk.LEFT, foreground="gray").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=5
        )

    def _build_preview_panel(self, parent):
        """Build the center preview panel."""
        preview_frame = ttk.LabelFrame(parent, text="Preview", padding="5")
        preview_frame.grid(row=1, column=1, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        # Input preview
        input_frame = ttk.LabelFrame(preview_frame, text="Input", padding="5")
        input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)

        self.input_canvas = tk.Canvas(input_frame, bg="#f0f0f0", highlightthickness=1)
        self.input_canvas.grid(row=0, column=0, sticky="nsew")
        self.input_canvas.bind("<Configure>", lambda e: self._update_input_preview())

        # Drop hint
        self.input_canvas.create_text(
            150, 150, text="Drop image here\nor click 'Open Image'", fill="gray", tags="hint"
        )

        # Output preview
        output_frame = ttk.LabelFrame(preview_frame, text="Output", padding="5")
        output_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.output_canvas = tk.Canvas(output_frame, bg="#ffffff", highlightthickness=1)
        self.output_canvas.grid(row=0, column=0, sticky="nsew")
        self.output_canvas.bind("<Configure>", lambda e: self._update_output_preview())

        # Output hint
        self.output_canvas.create_text(
            150, 150, text="Output preview\nwill appear here", fill="gray", tags="hint"
        )

        # Stats panel below previews
        stats_frame = ttk.Frame(preview_frame)
        stats_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        self.stats_label = ttk.Label(stats_frame, text="No conversion yet", foreground="gray")
        self.stats_label.pack(side=tk.LEFT)

    def _build_status_bar(self, parent):
        """Build the bottom status bar."""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)

        # File info on right
        self.file_label = ttk.Label(status_frame, text="", foreground="gray")
        self.file_label.pack(side=tk.RIGHT)

    def _on_paper_size_changed(self, event=None):
        """Handle paper size selection change."""
        selected = self.paper_size_var.get()

        # Find the PaperSize enum for selected option
        paper_size = None
        for name, size in PAPER_SIZE_OPTIONS:
            if name == selected:
                paper_size = size
                break

        if paper_size is None:
            # Custom selected - enable manual entry
            self.width_entry.config(state=tk.NORMAL)
            self.height_entry.config(state=tk.NORMAL)
        else:
            # Preset selected - fill in dimensions and disable entry
            width, height = get_paper_size(paper_size, self.landscape_var.get())
            self.width_var.set(str(int(width)))
            self.height_var.set(str(int(height)))
            self.width_entry.config(state=tk.DISABLED)
            self.height_entry.config(state=tk.DISABLED)

    def _on_orientation_changed(self):
        """Handle landscape/portrait toggle."""
        selected = self.paper_size_var.get()

        # Find the PaperSize enum for selected option
        paper_size = None
        for name, size in PAPER_SIZE_OPTIONS:
            if name == selected:
                paper_size = size
                break

        if paper_size is not None:
            # Update dimensions for new orientation
            width, height = get_paper_size(paper_size, self.landscape_var.get())
            self.width_var.set(str(int(width)))
            self.height_var.set(str(int(height)))
        elif self.width_var.get() and self.height_var.get():
            # Custom size - swap width and height
            w, h = self.width_var.get(), self.height_var.get()
            self.width_var.set(h)
            self.height_var.set(w)

    def _apply_preset(self, event=None):
        """Apply a configuration preset."""
        preset = self.preset_combo.get()

        if preset == "Default":
            self.style_var.set("natural")
            self.preserve_thickness_var.set(False)
            self.hatching_var.set(False)
            self.paper_size_var.set("Custom")
            self.width_var.set("")
            self.height_var.set("")
            self.width_entry.config(state=tk.NORMAL)
            self.height_entry.config(state=tk.NORMAL)
        elif preset == "Pen Plotter (A4)":
            self.style_var.set("natural")
            self.paper_size_var.set("A4 (210x297 mm)")
            self._on_paper_size_changed()
            self.stroke_width_var.set("0.4")
            self.line_sorting_var.set(True)
        elif preset == "Laser Cutter":
            self.style_var.set("clean")
            self.paper_size_var.set("Custom")
            self.width_var.set("300")
            self.height_var.set("200")
            self.width_entry.config(state=tk.NORMAL)
            self.height_entry.config(state=tk.NORMAL)
            self.stroke_width_var.set("0.1")
        elif preset == "Thick Lines":
            self.style_var.set("natural")
            self.preserve_thickness_var.set(True)
            self.stroke_spacing_var.set(1.5)
        elif preset == "Hatching":
            self.style_var.set("natural")
            self.hatching_var.set(True)
            self.hatch_spacing_var.set(2.0)

    def _open_file(self):
        """Open an image file."""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*"),
        ]

        path = filedialog.askopenfilename(title="Open Image", filetypes=filetypes)

        if path:
            self.input_path = Path(path)
            self._load_input_image()

    def _load_input_image(self):
        """Load and display the input image."""
        try:
            self.input_image = Image.open(self.input_path)
            self._update_input_preview()

            # Update UI state
            self.convert_btn.config(state=tk.NORMAL)
            self.file_label.config(
                text=f"{self.input_path.name} ({self.input_image.width}x{self.input_image.height})"
            )
            self.status_label.config(text=f"Loaded: {self.input_path.name}")

            # Clear output
            self.output_svg = None
            self._clear_output_preview()
            self.save_btn.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def _update_input_preview(self):
        """Update the input image preview."""
        if self.input_image is None:
            return

        canvas = self.input_canvas
        canvas.delete("all")

        # Get canvas size
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()

        if cw < 10 or ch < 10:
            return

        # Scale image to fit
        img = self.input_image.copy()
        img.thumbnail((cw - 20, ch - 20), Image.Resampling.LANCZOS)

        # Convert to PhotoImage
        self._input_photo = ImageTk.PhotoImage(img)

        # Center on canvas
        x = cw // 2
        y = ch // 2
        canvas.create_image(x, y, image=self._input_photo, anchor=tk.CENTER)

    def _update_output_preview(self):
        """Update the output SVG preview."""
        if self.output_svg is None:
            return

        canvas = self.output_canvas
        canvas.delete("all")

        # Get canvas size
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()

        if cw < 10 or ch < 10:
            return

        try:
            # Try to render SVG preview using cairosvg if available
            self._render_svg_preview(canvas, cw, ch)
        except Exception:
            # Fallback: just show success message
            canvas.create_text(
                cw // 2, ch // 2,
                text="SVG generated successfully!\n\nClick 'Save SVG' to save.",
                fill="green",
                justify=tk.CENTER
            )

    def _render_svg_preview(self, canvas, cw, ch):
        """Try to render SVG preview."""
        try:
            import cairosvg

            # Convert SVG to PNG
            png_data = cairosvg.svg2png(
                bytestring=self.output_svg.encode(),
                output_width=cw - 20,
                output_height=ch - 20,
            )

            # Load as PIL image
            img = Image.open(io.BytesIO(png_data))
            self._output_photo = ImageTk.PhotoImage(img)

            x = cw // 2
            y = ch // 2
            canvas.create_image(x, y, image=self._output_photo, anchor=tk.CENTER)

        except ImportError:
            # cairosvg not available, show placeholder
            canvas.create_text(
                cw // 2, ch // 2,
                text="SVG generated!\n\n(Install cairosvg for preview)\n\nClick 'Save SVG' to save.",
                fill="green",
                justify=tk.CENTER
            )

    def _clear_output_preview(self):
        """Clear the output preview."""
        self.output_canvas.delete("all")
        cw = self.output_canvas.winfo_width()
        ch = self.output_canvas.winfo_height()
        self.output_canvas.create_text(
            cw // 2, ch // 2,
            text="Output preview\nwill appear here",
            fill="gray",
            tags="hint"
        )
        self.stats_label.config(text="No conversion yet", foreground="gray")

    def _get_config(self) -> ConversionConfig:
        """Build ConversionConfig from current UI settings."""
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

        # Parse dimensions
        width = None
        height = None
        try:
            if self.width_var.get().strip():
                width = float(self.width_var.get())
            if self.height_var.get().strip():
                height = float(self.height_var.get())
        except ValueError:
            pass

        return ConversionConfig(
            sketch_style=style_map[self.style_var.get()],
            output_width=width,
            output_height=height,
            stroke_width=float(self.stroke_width_var.get() or 0.5),
            stroke_color=self.stroke_color_var.get() or "black",
            threshold_method=threshold_map[self.threshold_var.get()],
            threshold_value=self.threshold_value_var.get(),
            invert=self.invert_var.get(),
            denoise=self.denoise_var.get(),
            min_path_length=self.min_path_length_var.get(),
            simplify_tolerance=self.simplify_var.get(),
            curve_fitting=self.curve_fitting_var.get(),
            line_sorting=self.line_sorting_var.get(),
            wobble_amplitude=self.wobble_var.get(),
            line_variation=self.line_variation_var.get(),
            corner_rounding=self.corner_rounding_var.get(),
            preserve_thickness=self.preserve_thickness_var.get(),
            stroke_spacing=self.stroke_spacing_var.get(),
            min_width_for_multi=self.min_thickness_var.get(),
            fill_hatching=self.hatching_var.get(),
            hatch_angle=self.hatch_angle_var.get(),
            hatch_spacing=self.hatch_spacing_var.get(),
            cross_hatch=self.cross_hatch_var.get(),
        )

    def _convert(self):
        """Start the conversion process."""
        if self.is_converting or self.input_path is None:
            return

        self.is_converting = True
        self.convert_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Converting...")
        self.progress.start()

        # Run conversion in background thread
        thread = threading.Thread(target=self._do_convert, daemon=True)
        thread.start()

    def _do_convert(self):
        """Perform conversion in background thread."""
        try:
            config = self._get_config()
            converter = RasterToVectorConverter(config)

            seed = self.seed_var.get() if self.use_seed_var.get() else None
            result = converter.convert(self.input_path, seed=seed)

            self.output_svg = result.svg.content

            # Update UI from main thread
            self.root.after(0, lambda: self._conversion_done(result.stats))

        except Exception as e:
            self.root.after(0, lambda: self._conversion_error(str(e)))

    def _conversion_done(self, stats: dict):
        """Handle successful conversion."""
        self.is_converting = False
        self.convert_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.progress.stop()

        self.status_label.config(text="Conversion complete!")

        # Update stats
        stats_text = (
            f"Paths: {stats['paths_output']} | "
            f"Length: {stats['total_path_length']:.0f} units"
        )
        if stats.get('thickness_preserved'):
            stats_text += " | Thickness: ON"
        if stats.get('fill_hatching'):
            stats_text += " | Hatching: ON"

        self.stats_label.config(text=stats_text, foreground="black")

        # Update preview
        self._update_output_preview()

    def _conversion_error(self, error: str):
        """Handle conversion error."""
        self.is_converting = False
        self.convert_btn.config(state=tk.NORMAL)
        self.progress.stop()

        self.status_label.config(text="Conversion failed!")
        messagebox.showerror("Conversion Error", f"Failed to convert image:\n{error}")

    def _save_file(self):
        """Save the output SVG file."""
        if self.output_svg is None:
            return

        # Suggest filename based on input
        if self.input_path:
            initial_name = self.input_path.stem + ".svg"
        else:
            initial_name = "output.svg"

        path = filedialog.asksaveasfilename(
            title="Save SVG",
            defaultextension=".svg",
            initialfile=initial_name,
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
        )

        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.output_svg)

                self.output_path = Path(path)
                self.status_label.config(text=f"Saved: {Path(path).name}")
                messagebox.showinfo("Success", f"SVG saved to:\n{path}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Entry point for GUI application."""
    app = RasterToVectorGUI()
    app.run()


if __name__ == "__main__":
    main()
