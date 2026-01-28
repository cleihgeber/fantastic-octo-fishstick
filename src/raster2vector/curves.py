"""Curve fitting and sketchy effect generation."""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

from .tracing import Path
from .config import ConversionConfig, SketchStyle


@dataclass
class BezierSegment:
    """A cubic Bezier curve segment.

    Defined by 4 control points: start, control1, control2, end
    """

    p0: Tuple[float, float]  # Start point
    p1: Tuple[float, float]  # Control point 1
    p2: Tuple[float, float]  # Control point 2
    p3: Tuple[float, float]  # End point


@dataclass
class ProcessedPath:
    """A path that has been processed for output.

    Can contain either simple points or Bezier segments.
    """

    points: Optional[List[Tuple[float, float]]] = None
    bezier_segments: Optional[List[BezierSegment]] = None
    is_closed: bool = False

    @property
    def uses_bezier(self) -> bool:
        return self.bezier_segments is not None and len(self.bezier_segments) > 0


def simplify_path(points: List[Tuple[float, float]], tolerance: float = 1.0) -> List[Tuple[float, float]]:
    """Simplify a path using the Ramer-Douglas-Peucker algorithm.

    This reduces the number of points while preserving the overall shape.

    Args:
        points: List of (x, y) coordinates
        tolerance: Maximum deviation from the original path

    Returns:
        Simplified list of points
    """
    if len(points) <= 2:
        return points

    # Find point with maximum distance from line between first and last
    start = np.array(points[0])
    end = np.array(points[-1])

    # Line vector
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-10:
        return [points[0], points[-1]]

    line_unit = line_vec / line_len

    max_dist = 0.0
    max_idx = 0

    for i in range(1, len(points) - 1):
        point = np.array(points[i])
        # Vector from start to point
        point_vec = point - start
        # Project onto line
        proj_length = np.dot(point_vec, line_unit)
        proj_length = max(0, min(line_len, proj_length))
        proj_point = start + proj_length * line_unit
        # Distance from point to projection
        dist = np.linalg.norm(point - proj_point)

        if dist > max_dist:
            max_dist = dist
            max_idx = i

    # If max distance exceeds tolerance, recursively simplify
    if max_dist > tolerance:
        left = simplify_path(points[: max_idx + 1], tolerance)
        right = simplify_path(points[max_idx:], tolerance)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def smooth_path(
    points: List[Tuple[float, float]], window_size: int = 5
) -> List[Tuple[float, float]]:
    """Smooth a path using Gaussian filtering.

    Args:
        points: List of (x, y) coordinates
        window_size: Size of smoothing window (higher = smoother)

    Returns:
        Smoothed path
    """
    if len(points) < window_size:
        return points

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    sigma = window_size / 4.0
    x_smooth = gaussian_filter1d(x, sigma, mode="nearest")
    y_smooth = gaussian_filter1d(y, sigma, mode="nearest")

    # Preserve endpoints exactly
    x_smooth[0] = x[0]
    x_smooth[-1] = x[-1]
    y_smooth[0] = y[0]
    y_smooth[-1] = y[-1]

    return list(zip(x_smooth, y_smooth))


def add_wobble(
    points: List[Tuple[float, float]],
    amplitude: float = 0.5,
    frequency: float = 0.1,
    seed: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """Add natural-looking wobble to simulate hand-drawn lines.

    The wobble is applied perpendicular to the line direction,
    creating the effect of hand tremor.

    Args:
        points: List of (x, y) coordinates
        amplitude: Maximum wobble distance in pixels
        frequency: How often the wobble changes (0-1)
        seed: Random seed for reproducibility

    Returns:
        Path with added wobble
    """
    if len(points) < 3 or amplitude < 0.01:
        return points

    rng = np.random.default_rng(seed)
    result = [points[0]]  # Keep first point exact

    # Generate smooth noise
    n_points = len(points)
    noise_samples = int(n_points * frequency) + 2
    raw_noise = rng.standard_normal(noise_samples)
    # Interpolate to full length
    noise_x = np.linspace(0, 1, noise_samples)
    interp = interpolate.interp1d(noise_x, raw_noise, kind="cubic")
    full_x = np.linspace(0, 1, n_points)
    noise = interp(full_x) * amplitude

    for i in range(1, len(points) - 1):
        # Calculate perpendicular direction
        prev_pt = np.array(points[i - 1])
        curr_pt = np.array(points[i])
        next_pt = np.array(points[i + 1])

        # Direction vector (average of incoming and outgoing)
        direction = next_pt - prev_pt
        length = np.linalg.norm(direction)

        if length < 1e-10:
            result.append(points[i])
            continue

        # Perpendicular vector
        perp = np.array([-direction[1], direction[0]]) / length

        # Apply wobble perpendicular to path
        offset = perp * noise[i]
        new_point = curr_pt + offset
        result.append((new_point[0], new_point[1]))

    result.append(points[-1])  # Keep last point exact
    return result


def add_line_variation(
    points: List[Tuple[float, float]], variation: float = 0.2, seed: Optional[int] = None
) -> List[Tuple[float, float]]:
    """Add subtle variation to line positions for natural look.

    Unlike wobble (which is perpendicular), this adds random
    displacement in all directions for a more organic feel.

    Args:
        points: List of (x, y) coordinates
        variation: Maximum displacement in pixels
        seed: Random seed for reproducibility

    Returns:
        Path with added variation
    """
    if len(points) < 3 or variation < 0.01:
        return points

    rng = np.random.default_rng(seed)
    result = [points[0]]

    for i in range(1, len(points) - 1):
        x, y = points[i]
        # Random displacement that varies with position along path
        # Less at corners (detected by direction change)
        dx = rng.uniform(-variation, variation)
        dy = rng.uniform(-variation, variation)
        result.append((x + dx, y + dy))

    result.append(points[-1])
    return result


def round_corners(
    points: List[Tuple[float, float]], rounding: float = 0.3
) -> List[Tuple[float, float]]:
    """Slightly round sharp corners for more natural look.

    Args:
        points: List of (x, y) coordinates
        rounding: Amount of rounding (0-1)

    Returns:
        Path with rounded corners
    """
    if len(points) < 3 or rounding < 0.01:
        return points

    result = [points[0]]

    for i in range(1, len(points) - 1):
        prev_pt = np.array(points[i - 1])
        curr_pt = np.array(points[i])
        next_pt = np.array(points[i + 1])

        # Vectors to neighboring points
        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt

        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 < 1e-10 or len2 < 1e-10:
            result.append(points[i])
            continue

        # Calculate angle at corner
        v1_norm = v1 / len1
        v2_norm = v2 / len2
        cos_angle = np.dot(v1_norm, v2_norm)

        # Only round sharp corners (angle < 120 degrees)
        if cos_angle > -0.5:  # cos(120°) ≈ -0.5
            result.append(points[i])
            continue

        # Calculate rounding offset
        offset = min(len1, len2) * rounding * 0.5

        # Add two points to round the corner
        p1 = curr_pt + v1_norm * offset
        p2 = curr_pt + v2_norm * offset

        result.append((p1[0], p1[1]))
        result.append((p2[0], p2[1]))

    result.append(points[-1])
    return result


def fit_bezier_to_points(
    points: List[Tuple[float, float]], segments_per_path: int = 0
) -> List[BezierSegment]:
    """Fit cubic Bezier curves to a series of points.

    Uses the Catmull-Rom to Bezier conversion for smooth curves
    that pass through all control points.

    Args:
        points: List of (x, y) coordinates
        segments_per_path: Target number of segments (0 = auto)

    Returns:
        List of BezierSegment objects
    """
    if len(points) < 2:
        return []

    if len(points) == 2:
        # Straight line as degenerate Bezier
        p0, p1 = points
        return [
            BezierSegment(
                p0=p0,
                p1=(p0[0] + (p1[0] - p0[0]) / 3, p0[1] + (p1[1] - p0[1]) / 3),
                p2=(p0[0] + 2 * (p1[0] - p0[0]) / 3, p0[1] + 2 * (p1[1] - p0[1]) / 3),
                p3=p1,
            )
        ]

    # Convert points to numpy array
    pts = np.array(points)

    # Extend endpoints for Catmull-Rom
    extended = np.vstack(
        [2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]]
    )

    segments = []

    for i in range(1, len(extended) - 2):
        p0 = extended[i]
        p1 = extended[i + 1]
        pm1 = extended[i - 1]
        p2 = extended[i + 2]

        # Catmull-Rom to Bezier conversion
        # Bezier control points from Catmull-Rom
        bp0 = p0
        bp1 = p0 + (p1 - pm1) / 6
        bp2 = p1 - (p2 - p0) / 6
        bp3 = p1

        segments.append(
            BezierSegment(
                p0=(float(bp0[0]), float(bp0[1])),
                p1=(float(bp1[0]), float(bp1[1])),
                p2=(float(bp2[0]), float(bp2[1])),
                p3=(float(bp3[0]), float(bp3[1])),
            )
        )

    return segments


def process_path(path: Path, config: ConversionConfig, seed: Optional[int] = None) -> ProcessedPath:
    """Process a single path with all transformations.

    Args:
        path: Input path
        config: Conversion configuration
        seed: Random seed for reproducibility

    Returns:
        Processed path ready for output
    """
    points = list(path.points)

    # Step 1: Simplify to reduce points while keeping shape
    if config.simplify_tolerance > 0:
        points = simplify_path(points, config.simplify_tolerance)

    # Step 2: Smooth the path
    if config.smoothing_window > 0:
        points = smooth_path(points, config.smoothing_window)

    # Step 3: Round sharp corners
    if config.corner_rounding > 0:
        points = round_corners(points, config.corner_rounding)

    # Step 4: Add hand-drawn effects (only for non-clean styles)
    if config.sketch_style != SketchStyle.CLEAN:
        if config.wobble_amplitude > 0:
            points = add_wobble(points, config.wobble_amplitude, config.wobble_frequency, seed)

        if config.line_variation > 0:
            next_seed = seed + 1000 if seed else None
            points = add_line_variation(points, config.line_variation, next_seed)

    # Step 5: Fit Bezier curves if requested
    if config.curve_fitting and len(points) >= 3:
        bezier_segments = fit_bezier_to_points(points)
        return ProcessedPath(bezier_segments=bezier_segments, is_closed=path.is_closed)
    else:
        return ProcessedPath(points=points, is_closed=path.is_closed)


def process_all_paths(
    paths: List[Path], config: ConversionConfig, seed: Optional[int] = None
) -> List[ProcessedPath]:
    """Process all paths with configuration settings.

    Args:
        paths: List of input paths
        config: Conversion configuration
        seed: Random seed for reproducibility

    Returns:
        List of processed paths
    """
    processed = []

    for i, path in enumerate(paths):
        path_seed = seed + i if seed else None
        processed_path = process_path(path, config, path_seed)
        processed.append(processed_path)

    return processed
