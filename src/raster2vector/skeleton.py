"""Skeletonization for extracting single-pixel-wide centerlines."""

import numpy as np
from skimage.morphology import skeletonize, thin
from skimage import img_as_bool, img_as_ubyte
import cv2
from typing import Tuple, List, Set
from dataclasses import dataclass


@dataclass
class SkeletonAnalysis:
    """Analysis results from skeleton processing.

    Attributes:
        skeleton: Binary skeleton image (single-pixel wide lines)
        endpoints: List of (y, x) coordinates of line endpoints
        junctions: List of (y, x) coordinates of line junctions
        pixel_count: Total number of skeleton pixels
    """

    skeleton: np.ndarray
    endpoints: List[Tuple[int, int]]
    junctions: List[Tuple[int, int]]
    pixel_count: int


def extract_skeleton(binary: np.ndarray, method: str = "zhang") -> np.ndarray:
    """Extract single-pixel-wide skeleton from binary image.

    Args:
        binary: Binary image (black lines on white background, or vice versa)
        method: Skeletonization method - 'zhang' (default) or 'lee'

    Returns:
        Skeleton image (white skeleton on black background)
    """
    # Ensure we have white lines on black background for skeletonization
    # Check if image has more black or white pixels
    if np.mean(binary) > 127:
        # White background, black lines - invert
        work_image = cv2.bitwise_not(binary)
    else:
        work_image = binary.copy()

    # Convert to boolean
    bool_image = img_as_bool(work_image)

    # Apply skeletonization
    if method == "lee":
        skeleton = skeletonize(bool_image, method="lee")
    else:
        skeleton = skeletonize(bool_image, method="zhang")

    # Convert back to uint8
    return img_as_ubyte(skeleton)


def find_neighbors(skeleton: np.ndarray, y: int, x: int) -> List[Tuple[int, int]]:
    """Find all neighboring skeleton pixels for a given position.

    Args:
        skeleton: Skeleton image
        y, x: Position to check

    Returns:
        List of (y, x) coordinates of neighboring pixels
    """
    neighbors = []
    h, w = skeleton.shape

    # 8-connectivity neighborhood
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] > 0:
                neighbors.append((ny, nx))

    return neighbors


def count_neighbors(skeleton: np.ndarray, y: int, x: int) -> int:
    """Count the number of neighboring skeleton pixels.

    Args:
        skeleton: Skeleton image
        y, x: Position to check

    Returns:
        Number of neighboring pixels (0-8)
    """
    return len(find_neighbors(skeleton, y, x))


def analyze_skeleton(skeleton: np.ndarray) -> SkeletonAnalysis:
    """Analyze skeleton to find endpoints and junctions.

    Endpoints have 1 neighbor, junctions have 3+ neighbors.

    Args:
        skeleton: Skeleton image

    Returns:
        SkeletonAnalysis with endpoints, junctions, and pixel count
    """
    endpoints = []
    junctions = []
    pixel_count = 0

    # Find all skeleton pixels and classify them
    h, w = skeleton.shape
    for y in range(h):
        for x in range(w):
            if skeleton[y, x] > 0:
                pixel_count += 1
                n_neighbors = count_neighbors(skeleton, y, x)

                if n_neighbors == 1:
                    endpoints.append((y, x))
                elif n_neighbors >= 3:
                    junctions.append((y, x))

    return SkeletonAnalysis(
        skeleton=skeleton,
        endpoints=endpoints,
        junctions=junctions,
        pixel_count=pixel_count,
    )


def prune_spurs(skeleton: np.ndarray, min_length: int = 5) -> np.ndarray:
    """Remove short spurs (branches) from skeleton.

    Spurs are short branches that often result from noise or
    imperfect line endings.

    Args:
        skeleton: Skeleton image
        min_length: Minimum branch length to keep

    Returns:
        Pruned skeleton image
    """
    result = skeleton.copy()
    changed = True

    while changed:
        changed = False
        analysis = analyze_skeleton(result)

        for endpoint in analysis.endpoints:
            # Trace from endpoint to see if it's a short spur
            path = _trace_until_junction(result, endpoint, analysis.junctions, max_length=min_length)

            if path is not None and len(path) < min_length:
                # Remove this spur
                for y, x in path[:-1]:  # Keep the junction point
                    result[y, x] = 0
                changed = True

    return result


def _trace_until_junction(
    skeleton: np.ndarray,
    start: Tuple[int, int],
    junctions: List[Tuple[int, int]],
    max_length: int,
) -> List[Tuple[int, int]]:
    """Trace from start point until hitting a junction or exceeding max_length.

    Args:
        skeleton: Skeleton image
        start: Starting point (y, x)
        junctions: List of junction coordinates
        max_length: Maximum length to trace

    Returns:
        Path as list of (y, x) coordinates, or None if not a spur
    """
    junction_set = set(junctions)
    path = [start]
    visited = {start}
    current = start

    while len(path) <= max_length:
        neighbors = find_neighbors(skeleton, current[0], current[1])
        unvisited = [n for n in neighbors if n not in visited]

        if not unvisited:
            # Dead end or only visited neighbors
            return path

        if len(unvisited) > 1:
            # Multiple paths - we hit a junction
            return path

        current = unvisited[0]
        path.append(current)
        visited.add(current)

        if current in junction_set:
            # Reached a junction
            return path

    # Exceeded max length - not a short spur
    return None


def clean_skeleton(
    skeleton: np.ndarray, remove_small: int = 10, prune_spur_length: int = 5
) -> np.ndarray:
    """Clean up skeleton by removing noise and short spurs.

    Args:
        skeleton: Skeleton image
        remove_small: Remove connected components smaller than this
        prune_spur_length: Remove spurs shorter than this

    Returns:
        Cleaned skeleton
    """
    result = skeleton.copy()

    # Remove small disconnected components
    if remove_small > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < remove_small:
                result[labels == i] = 0

    # Prune short spurs
    if prune_spur_length > 0:
        result = prune_spurs(result, prune_spur_length)

    return result
