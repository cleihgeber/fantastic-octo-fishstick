"""Path tracing to convert skeleton pixels to ordered polylines."""

import numpy as np
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import deque

try:
    from scipy.spatial import cKDTree
    KDTREE_AVAILABLE = True
except ImportError:
    KDTREE_AVAILABLE = False

from .skeleton import find_neighbors, analyze_skeleton


@dataclass
class Path:
    """A single traced path/polyline.

    Attributes:
        points: List of (x, y) coordinates (note: x, y not y, x for SVG compatibility)
        is_closed: Whether this path forms a closed loop
    """

    points: List[Tuple[float, float]]
    is_closed: bool = False

    def __len__(self) -> int:
        return len(self.points)

    @property
    def length(self) -> float:
        """Calculate total path length."""
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for i in range(len(self.points) - 1):
            dx = self.points[i + 1][0] - self.points[i][0]
            dy = self.points[i + 1][1] - self.points[i][1]
            total += np.sqrt(dx * dx + dy * dy)
        return total

    def reverse(self) -> "Path":
        """Return a reversed copy of this path."""
        return Path(points=list(reversed(self.points)), is_closed=self.is_closed)


def trace_paths(skeleton: np.ndarray, min_path_length: int = 5) -> List[Path]:
    """Trace all paths from a skeleton image.

    This function converts a skeleton image into a list of ordered paths.
    It handles:
    - Simple lines (endpoint to endpoint)
    - Branching structures (paths split at junctions)
    - Closed loops

    Args:
        skeleton: Skeleton image (white lines on black background)
        min_path_length: Minimum number of points for a valid path

    Returns:
        List of Path objects
    """
    if skeleton.max() == 0:
        return []

    # Analyze skeleton structure
    analysis = analyze_skeleton(skeleton)

    # Create working copy to mark visited pixels
    visited = np.zeros_like(skeleton, dtype=bool)
    paths = []

    # Convert junctions to set for fast lookup
    junction_set = set(analysis.junctions)

    # First, trace from all endpoints
    for endpoint in analysis.endpoints:
        if visited[endpoint[0], endpoint[1]]:
            continue

        path = _trace_from_point(skeleton, endpoint, visited, junction_set)
        if path and len(path.points) >= min_path_length:
            paths.append(path)

    # Then, trace from junctions (for branches not connected to endpoints)
    for junction in analysis.junctions:
        neighbors = find_neighbors(skeleton, junction[0], junction[1])
        for neighbor in neighbors:
            if visited[neighbor[0], neighbor[1]]:
                continue

            path = _trace_from_point(skeleton, neighbor, visited, junction_set, start_junction=junction)
            if path and len(path.points) >= min_path_length:
                paths.append(path)

    # Finally, find any remaining unvisited pixels (closed loops)
    remaining = np.where((skeleton > 0) & (~visited))
    if len(remaining[0]) > 0:
        for y, x in zip(remaining[0], remaining[1]):
            if visited[y, x]:
                continue

            path = _trace_loop(skeleton, (y, x), visited)
            if path and len(path.points) >= min_path_length:
                paths.append(path)

    return paths


def _trace_from_point(
    skeleton: np.ndarray,
    start: Tuple[int, int],
    visited: np.ndarray,
    junctions: Set[Tuple[int, int]],
    start_junction: Optional[Tuple[int, int]] = None,
) -> Optional[Path]:
    """Trace a single path starting from a point.

    Args:
        skeleton: Skeleton image
        start: Starting point (y, x)
        visited: Array tracking visited pixels
        junctions: Set of junction coordinates
        start_junction: If tracing from a junction neighbor, include the junction

    Returns:
        Path object or None if too short
    """
    points = []

    # Add starting junction if provided
    if start_junction is not None:
        points.append((start_junction[1], start_junction[0]))  # Convert to (x, y)

    current = start
    path_visited = set()

    while current is not None:
        y, x = current
        points.append((x, y))  # Convert to (x, y) for SVG
        visited[y, x] = True
        path_visited.add(current)

        # Check if we've reached a junction (and it's not our start)
        if current in junctions and len(points) > 1:
            break

        # Find next unvisited neighbor
        neighbors = find_neighbors(skeleton, y, x)
        next_point = None

        for neighbor in neighbors:
            if neighbor not in path_visited and not visited[neighbor[0], neighbor[1]]:
                next_point = neighbor
                break

        current = next_point

    if len(points) < 2:
        return None

    return Path(points=points)


def _trace_loop(
    skeleton: np.ndarray, start: Tuple[int, int], visited: np.ndarray
) -> Optional[Path]:
    """Trace a closed loop starting from a point.

    Args:
        skeleton: Skeleton image
        start: Starting point (y, x)
        visited: Array tracking visited pixels

    Returns:
        Path object (marked as closed) or None
    """
    points = []
    current = start
    first_point = start

    while current is not None:
        y, x = current
        points.append((x, y))  # Convert to (x, y)
        visited[y, x] = True

        # Find next unvisited neighbor
        neighbors = find_neighbors(skeleton, y, x)
        next_point = None

        for neighbor in neighbors:
            if not visited[neighbor[0], neighbor[1]]:
                next_point = neighbor
                break

        # Check if we can close the loop
        if next_point is None:
            for neighbor in neighbors:
                if neighbor == first_point and len(points) > 2:
                    # Close the loop
                    return Path(points=points, is_closed=True)

        current = next_point

    if len(points) < 3:
        return None

    return Path(points=points, is_closed=len(points) > 2)


def merge_nearby_endpoints(
    paths: List[Path], distance_threshold: float = 3.0, max_iterations: int = 50
) -> List[Path]:
    """Merge paths with nearby endpoints using spatial indexing for speed.

    This helps connect paths that were split due to small gaps
    or noise in the original image.

    Args:
        paths: List of paths
        distance_threshold: Maximum distance to merge
        max_iterations: Maximum merge iterations to prevent infinite loops

    Returns:
        List of merged paths
    """
    if len(paths) <= 1:
        return paths

    # Separate closed and open paths - closed paths don't need merging
    closed_paths = [p for p in paths if p.is_closed]
    open_paths = [p for p in paths if not p.is_closed]

    if len(open_paths) <= 1:
        return paths

    # Use union-find for efficient merging
    n = len(open_paths)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    # Build list of all endpoints with their path index and position (start=0, end=1)
    endpoints = []
    endpoint_info = []  # (path_idx, is_end)

    for i, p in enumerate(open_paths):
        endpoints.append(p.points[0])   # start
        endpoint_info.append((i, False))
        endpoints.append(p.points[-1])  # end
        endpoint_info.append((i, True))

    endpoints = np.array(endpoints)

    # Use KDTree for fast nearest-neighbor queries
    if KDTREE_AVAILABLE and len(endpoints) > 10:
        tree = cKDTree(endpoints)
        pairs = tree.query_pairs(r=distance_threshold)

        # Process pairs - connect paths whose endpoints are close
        merge_instructions = []  # (path_i, path_j, i_is_end, j_is_end)

        for ep_i, ep_j in pairs:
            path_i, is_end_i = endpoint_info[ep_i]
            path_j, is_end_j = endpoint_info[ep_j]

            if path_i != path_j:
                merge_instructions.append((path_i, path_j, is_end_i, is_end_j))

        # Sort by distance (approximate - using endpoint indices as proxy)
        # and process merges using union-find
        for path_i, path_j, is_end_i, is_end_j in merge_instructions:
            union(path_i, path_j)
    else:
        # Fallback for small path counts or no scipy
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                dist = np.sqrt((endpoints[i][0] - endpoints[j][0])**2 +
                              (endpoints[i][1] - endpoints[j][1])**2)
                if dist <= distance_threshold:
                    path_i, _ = endpoint_info[i]
                    path_j, _ = endpoint_info[j]
                    if path_i != path_j:
                        union(path_i, path_j)

    # Group paths by their root in union-find
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Merge paths within each group
    merged_paths = []

    for group_indices in groups.values():
        if len(group_indices) == 1:
            merged_paths.append(open_paths[group_indices[0]])
            continue

        # Get all paths in this group
        group_paths = [list(open_paths[i].points) for i in group_indices]

        # Simple greedy merge: start with first path, extend by finding closest endpoints
        result = group_paths[0]
        remaining = group_paths[1:]

        for _ in range(min(len(remaining) * 2, max_iterations)):
            if not remaining:
                break

            best_idx = None
            best_dist = float('inf')
            best_conn = None

            r_start = result[0]
            r_end = result[-1]

            for idx, path in enumerate(remaining):
                p_start = path[0]
                p_end = path[-1]

                # Check all four connection types
                d1 = (r_end[0] - p_start[0])**2 + (r_end[1] - p_start[1])**2
                if d1 < best_dist:
                    best_dist, best_idx, best_conn = d1, idx, ("end", False)

                d2 = (r_end[0] - p_end[0])**2 + (r_end[1] - p_end[1])**2
                if d2 < best_dist:
                    best_dist, best_idx, best_conn = d2, idx, ("end", True)

                d3 = (r_start[0] - p_start[0])**2 + (r_start[1] - p_start[1])**2
                if d3 < best_dist:
                    best_dist, best_idx, best_conn = d3, idx, ("start", True)

                d4 = (r_start[0] - p_end[0])**2 + (r_start[1] - p_end[1])**2
                if d4 < best_dist:
                    best_dist, best_idx, best_conn = d4, idx, ("start", False)

            if best_idx is not None:
                path = remaining.pop(best_idx)
                connect_at, reverse = best_conn

                if reverse:
                    path = list(reversed(path))

                if connect_at == "end":
                    result.extend(path[1:])
                else:
                    result = path[:-1] + result

        # Add any remaining paths that couldn't be merged
        merged_paths.append(Path(points=result))
        for path in remaining:
            merged_paths.append(Path(points=path))

    return closed_paths + merged_paths


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def sort_paths_for_plotting(paths: List[Path]) -> List[Path]:
    """Sort paths to minimize pen travel distance.

    Uses a greedy nearest-neighbor approach to order paths
    such that the end of one path is close to the start of the next.

    Args:
        paths: List of paths to sort

    Returns:
        Sorted list of paths (may have some paths reversed)
    """
    if len(paths) <= 1:
        return paths

    sorted_paths = []
    remaining = list(range(len(paths)))

    # Start with the path closest to origin
    current_pos = (0.0, 0.0)
    best_idx = 0
    best_dist = float("inf")
    best_reversed = False

    for i in remaining:
        d_start = _distance(current_pos, paths[i].points[0])
        d_end = _distance(current_pos, paths[i].points[-1])

        if d_start < best_dist:
            best_dist = d_start
            best_idx = i
            best_reversed = False
        if d_end < best_dist:
            best_dist = d_end
            best_idx = i
            best_reversed = True

    if best_reversed and not paths[best_idx].is_closed:
        sorted_paths.append(paths[best_idx].reverse())
    else:
        sorted_paths.append(paths[best_idx])
    remaining.remove(best_idx)
    current_pos = sorted_paths[-1].points[-1]

    # Greedily add nearest paths
    while remaining:
        best_idx = remaining[0]
        best_dist = float("inf")
        best_reversed = False

        for i in remaining:
            d_start = _distance(current_pos, paths[i].points[0])
            d_end = _distance(current_pos, paths[i].points[-1])

            if d_start < best_dist:
                best_dist = d_start
                best_idx = i
                best_reversed = False
            if d_end < best_dist and not paths[i].is_closed:
                best_dist = d_end
                best_idx = i
                best_reversed = True

        if best_reversed:
            sorted_paths.append(paths[best_idx].reverse())
        else:
            sorted_paths.append(paths[best_idx])

        remaining.remove(best_idx)
        current_pos = sorted_paths[-1].points[-1]

    return sorted_paths


def filter_paths(paths: List[Path], min_length: int = 5, min_pixel_length: float = 10.0) -> List[Path]:
    """Filter out paths that are too short.

    Args:
        paths: List of paths
        min_length: Minimum number of points
        min_pixel_length: Minimum path length in pixels

    Returns:
        Filtered list of paths
    """
    return [p for p in paths if len(p.points) >= min_length and p.length >= min_pixel_length]
