"""Stroke width analysis and multi-stroke generation for preserving visual mass."""

import numpy as np
import cv2
from scipy import ndimage
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .tracing import Path


@dataclass
class StrokeWidthInfo:
    """Information about stroke width along a path.

    Attributes:
        path: The original path
        widths: Width value at each point in the path
        mean_width: Average width along the path
        max_width: Maximum width along the path
    """

    path: Path
    widths: List[float]
    mean_width: float
    max_width: float


def compute_distance_transform(binary: np.ndarray) -> np.ndarray:
    """Compute distance transform of binary image.

    The distance transform gives the distance from each foreground pixel
    to the nearest background pixel, effectively encoding stroke width.

    Args:
        binary: Binary image (black lines on white background expected)

    Returns:
        Distance transform image (float)
    """
    # Ensure we have white foreground (lines) on black background
    if np.mean(binary) > 127:
        # White background - invert so lines are white
        work = cv2.bitwise_not(binary)
    else:
        work = binary.copy()

    # Compute distance transform
    dist = cv2.distanceTransform(work, cv2.DIST_L2, 5)

    return dist


def analyze_stroke_widths(
    paths: List[Path], distance_transform: np.ndarray
) -> List[StrokeWidthInfo]:
    """Analyze stroke width along each path using distance transform.

    Args:
        paths: List of traced paths
        distance_transform: Distance transform of the original binary image

    Returns:
        List of StrokeWidthInfo for each path
    """
    results = []
    h, w = distance_transform.shape

    for path in paths:
        widths = []

        for x, y in path.points:
            # Convert to integer coordinates
            ix, iy = int(round(x)), int(round(y))

            # Clamp to image bounds
            ix = max(0, min(w - 1, ix))
            iy = max(0, min(h - 1, iy))

            # Distance transform value is radius, width is diameter
            width = distance_transform[iy, ix] * 2
            widths.append(width)

        if widths:
            mean_width = np.mean(widths)
            max_width = np.max(widths)
        else:
            mean_width = 1.0
            max_width = 1.0

        results.append(
            StrokeWidthInfo(
                path=path,
                widths=widths,
                mean_width=mean_width,
                max_width=max_width,
            )
        )

    return results


def generate_parallel_strokes(
    path: Path,
    widths: List[float],
    stroke_spacing: float = 1.0,
    min_width_for_multi: float = 3.0,
) -> List[Path]:
    """Generate parallel strokes to fill thick line areas.

    For areas where the original stroke is thick, this generates multiple
    parallel offset paths to preserve the visual mass.

    Args:
        path: Original centerline path
        widths: Width at each point along the path
        stroke_spacing: Spacing between parallel strokes (in pixels)
        min_width_for_multi: Minimum width before generating multiple strokes

    Returns:
        List of paths (including original and offsets)
    """
    if len(path.points) < 2:
        return [path]

    points = np.array(path.points)
    widths_arr = np.array(widths)

    # Calculate average width for this path
    avg_width = np.mean(widths_arr)

    # If thin enough, just return the original
    if avg_width < min_width_for_multi:
        return [path]

    # Calculate how many strokes we need
    # Each stroke covers stroke_spacing, we want to fill the width
    num_strokes = max(1, int(avg_width / stroke_spacing))

    if num_strokes <= 1:
        return [path]

    # Generate offset paths
    result_paths = []

    # Calculate offsets (centered around the centerline)
    total_width = (num_strokes - 1) * stroke_spacing
    offsets = np.linspace(-total_width / 2, total_width / 2, num_strokes)

    for offset in offsets:
        if abs(offset) < 0.01:
            # This is the centerline
            result_paths.append(path)
        else:
            # Generate offset path
            offset_points = _offset_path(points, widths_arr, offset, min_width_for_multi)
            if offset_points is not None and len(offset_points) >= 2:
                result_paths.append(Path(points=offset_points, is_closed=path.is_closed))

    return result_paths if result_paths else [path]


def _offset_path(
    points: np.ndarray,
    widths: np.ndarray,
    offset: float,
    min_width: float,
) -> Optional[List[Tuple[float, float]]]:
    """Create an offset version of a path.

    The offset is applied perpendicular to the path direction.
    Offset is only applied where the original stroke is wide enough.

    Args:
        points: Nx2 array of path points
        widths: Width at each point
        offset: Offset distance (positive = right side, negative = left)
        min_width: Minimum width to apply offset

    Returns:
        List of offset points, or None if path degenerates
    """
    n = len(points)
    if n < 2:
        return None

    offset_points = []

    for i in range(n):
        # Calculate tangent direction
        if i == 0:
            tangent = points[1] - points[0]
        elif i == n - 1:
            tangent = points[n - 1] - points[n - 2]
        else:
            tangent = points[i + 1] - points[i - 1]

        tangent_len = np.linalg.norm(tangent)
        if tangent_len < 1e-10:
            offset_points.append((float(points[i][0]), float(points[i][1])))
            continue

        tangent = tangent / tangent_len

        # Perpendicular (90 degrees counterclockwise)
        perp = np.array([-tangent[1], tangent[0]])

        # Scale offset by local width ratio
        # Only offset if the stroke is wide enough at this point
        local_width = widths[i] if i < len(widths) else widths[-1]

        if local_width >= min_width:
            # Scale offset based on local width vs desired offset
            # This makes offset follow the stroke boundary
            effective_offset = min(abs(offset), local_width / 2 - 0.5)
            if offset < 0:
                effective_offset = -effective_offset
        else:
            # Stroke too thin here, minimal offset
            effective_offset = offset * 0.1

        new_point = points[i] + perp * effective_offset
        offset_points.append((float(new_point[0]), float(new_point[1])))

    return offset_points


def generate_fill_hatching(
    binary: np.ndarray,
    angle: float = 45.0,
    spacing: float = 2.0,
    min_area: int = 100,
) -> List[Path]:
    """Generate hatching lines to fill solid/dense areas.

    This detects areas that are solid fills (not just thick strokes)
    and generates hatching lines to represent them with single strokes.

    Args:
        binary: Binary image (black fills on white background)
        angle: Hatching angle in degrees
        spacing: Spacing between hatch lines in pixels
        min_area: Minimum area to consider for hatching

    Returns:
        List of hatching paths
    """
    # Find filled regions (not just lines)
    # Invert if needed so fills are white
    if np.mean(binary) > 127:
        work = cv2.bitwise_not(binary)
    else:
        work = binary.copy()

    # Find contours of filled regions
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hatch_paths = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Check if this is a filled region (not just a thick line)
        # by comparing area to perimeter
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1:
            continue

        # Circularity-like measure: filled regions have higher area/perimeter ratio
        compactness = area / (perimeter * perimeter) * 4 * np.pi

        # Only hatch if reasonably compact (filled region, not thin line)
        if compactness < 0.1:
            continue

        # Generate hatching for this region
        paths = _hatch_contour(contour, work.shape, angle, spacing)
        hatch_paths.extend(paths)

    return hatch_paths


def _hatch_contour(
    contour: np.ndarray,
    image_shape: Tuple[int, int],
    angle: float,
    spacing: float,
) -> List[Path]:
    """Generate hatching lines within a contour.

    Args:
        contour: OpenCV contour
        image_shape: Shape of the image (h, w)
        angle: Hatching angle in degrees
        spacing: Spacing between lines

    Returns:
        List of hatching paths
    """
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # Create mask for this contour
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)

    # Calculate line parameters
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Direction perpendicular to hatch lines
    perp = np.array([cos_a, sin_a])

    # Direction along hatch lines
    along = np.array([-sin_a, cos_a])

    # Center of bounding box
    cx, cy = x + w / 2, y + h / 2

    # Maximum distance from center to corners
    max_dist = np.sqrt(w * w + h * h) / 2

    # Generate hatch lines
    paths = []
    num_lines = int(2 * max_dist / spacing) + 1

    for i in range(num_lines):
        # Offset from center along perpendicular direction
        offset = (i - num_lines // 2) * spacing

        # Line passes through this point
        line_center = np.array([cx, cy]) + perp * offset

        # Find line endpoints (extend far enough to cross region)
        p1 = line_center - along * max_dist * 1.5
        p2 = line_center + along * max_dist * 1.5

        # Clip line to mask
        clipped = _clip_line_to_mask(p1, p2, mask)

        for segment in clipped:
            if len(segment) >= 2:
                paths.append(Path(points=segment, is_closed=False))

    return paths


def _clip_line_to_mask(
    p1: np.ndarray, p2: np.ndarray, mask: np.ndarray
) -> List[List[Tuple[float, float]]]:
    """Clip a line segment to a binary mask, returning visible segments.

    Args:
        p1: Line start point
        p2: Line end point
        mask: Binary mask (255 = inside)

    Returns:
        List of line segments (each is a list of points)
    """
    h, w = mask.shape

    # Sample points along the line
    length = np.linalg.norm(p2 - p1)
    if length < 1:
        return []

    num_samples = int(length) + 1
    t = np.linspace(0, 1, num_samples)
    points = np.outer(1 - t, p1) + np.outer(t, p2)

    # Check which points are inside the mask
    segments = []
    current_segment = []

    for i, pt in enumerate(points):
        x, y = int(round(pt[0])), int(round(pt[1]))

        # Check if inside image and mask
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            current_segment.append((float(pt[0]), float(pt[1])))
        else:
            # End current segment if any
            if len(current_segment) >= 2:
                segments.append(current_segment)
            current_segment = []

    # Don't forget last segment
    if len(current_segment) >= 2:
        segments.append(current_segment)

    return segments


def expand_paths_for_thickness(
    paths: List[Path],
    binary: np.ndarray,
    stroke_spacing: float = 1.5,
    min_width_for_multi: float = 4.0,
    include_hatching: bool = True,
    hatch_angle: float = 45.0,
    hatch_spacing: float = 2.0,
    min_hatch_area: int = 200,
) -> List[Path]:
    """Expand thin centerline paths to multiple strokes based on original thickness.

    This is the main function to call for preserving visual mass.

    Args:
        paths: Original centerline paths
        binary: Original binary image (for width analysis)
        stroke_spacing: Spacing between parallel strokes
        min_width_for_multi: Minimum width to generate multiple strokes
        include_hatching: Whether to add hatching for filled areas
        hatch_angle: Angle for hatching lines
        hatch_spacing: Spacing for hatching lines
        min_hatch_area: Minimum area to hatch

    Returns:
        Expanded list of paths
    """
    # Compute distance transform for width analysis
    dist_transform = compute_distance_transform(binary)

    # Analyze width along each path
    width_info = analyze_stroke_widths(paths, dist_transform)

    # Generate parallel strokes for thick lines
    expanded_paths = []

    for info in width_info:
        parallel_paths = generate_parallel_strokes(
            info.path,
            info.widths,
            stroke_spacing=stroke_spacing,
            min_width_for_multi=min_width_for_multi,
        )
        expanded_paths.extend(parallel_paths)

    # Optionally add hatching for filled regions
    if include_hatching:
        hatch_paths = generate_fill_hatching(
            binary,
            angle=hatch_angle,
            spacing=hatch_spacing,
            min_area=min_hatch_area,
        )
        expanded_paths.extend(hatch_paths)

    return expanded_paths
