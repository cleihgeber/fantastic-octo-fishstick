"""Tests for raster2vector core functionality."""

import numpy as np
import pytest

from raster2vector.config import ConversionConfig, SketchStyle, ThresholdMethod
from raster2vector.preprocessing import to_grayscale, binarize
from raster2vector.skeleton import extract_skeleton, analyze_skeleton
from raster2vector.tracing import trace_paths, Path
from raster2vector.curves import simplify_path, add_wobble, fit_bezier_to_points


class TestConfig:
    """Tests for ConversionConfig."""

    def test_default_config(self):
        config = ConversionConfig()
        assert config.sketch_style == SketchStyle.NATURAL
        assert config.threshold_method == ThresholdMethod.OTSU

    def test_pen_plotter_preset(self):
        config = ConversionConfig.for_pen_plotter(200, 200)
        assert config.output_width == 200
        assert config.output_height == 200
        assert config.line_sorting is True

    def test_style_presets(self):
        clean = ConversionConfig(sketch_style=SketchStyle.CLEAN)
        assert clean.wobble_amplitude == 0.0

        rough = ConversionConfig(sketch_style=SketchStyle.ROUGH)
        assert rough.wobble_amplitude > 0


class TestPreprocessing:
    """Tests for image preprocessing."""

    def test_to_grayscale_already_gray(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        result = to_grayscale(gray)
        assert result.shape == (100, 100)

    def test_to_grayscale_bgr(self):
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = to_grayscale(bgr)
        assert result.shape == (100, 100)

    def test_binarize_otsu(self):
        # Create gradient image
        gray = np.tile(np.arange(256, dtype=np.uint8), (100, 1))
        binary = binarize(gray, ThresholdMethod.OTSU)
        assert set(np.unique(binary)).issubset({0, 255})


class TestSkeleton:
    """Tests for skeletonization."""

    def test_extract_skeleton_line(self):
        # Create thick horizontal line
        image = np.zeros((50, 100), dtype=np.uint8)
        image[20:30, 10:90] = 255  # White line

        skeleton = extract_skeleton(image)

        # Should have skeleton pixels
        assert skeleton.max() > 0

        # Skeleton should be thinner than original
        assert np.sum(skeleton > 0) < np.sum(image > 0)

    def test_analyze_skeleton(self):
        # Create simple cross shape
        skeleton = np.zeros((50, 50), dtype=np.uint8)
        skeleton[25, 10:40] = 255  # Horizontal
        skeleton[10:40, 25] = 255  # Vertical

        analysis = analyze_skeleton(skeleton)

        assert analysis.pixel_count > 0
        assert len(analysis.endpoints) == 4  # Four endpoints
        assert len(analysis.junctions) == 1  # One junction at center


class TestTracing:
    """Tests for path tracing."""

    def test_trace_simple_line(self):
        # Create horizontal skeleton line
        skeleton = np.zeros((50, 100), dtype=np.uint8)
        skeleton[25, 10:90] = 255

        paths = trace_paths(skeleton, min_path_length=3)

        assert len(paths) >= 1
        assert all(isinstance(p, Path) for p in paths)

    def test_path_length(self):
        points = [(0, 0), (3, 0), (3, 4)]  # 3-4-5 triangle partial
        path = Path(points=points)
        assert path.length == 7.0  # 3 + 4


class TestCurves:
    """Tests for curve processing."""

    def test_simplify_path(self):
        # Create path with many points on a line
        points = [(float(i), 0.0) for i in range(100)]
        simplified = simplify_path(points, tolerance=1.0)

        # Should reduce to just 2 points for a straight line
        assert len(simplified) == 2
        assert simplified[0] == (0.0, 0.0)
        assert simplified[-1] == (99.0, 0.0)

    def test_add_wobble_preserves_endpoints(self):
        points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        wobbled = add_wobble(points, amplitude=0.5, seed=42)

        assert len(wobbled) == len(points)
        assert wobbled[0] == points[0]  # First point preserved
        assert wobbled[-1] == points[-1]  # Last point preserved

    def test_fit_bezier_to_points(self):
        points = [(0, 0), (1, 1), (2, 0), (3, 1)]
        segments = fit_bezier_to_points(points)

        assert len(segments) > 0
        assert segments[0].p0 == (0, 0)  # Starts at first point


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
