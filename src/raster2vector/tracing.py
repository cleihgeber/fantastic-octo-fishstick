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

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root: # Path compression
            next_node = parent[i]
            parent[i] = root
            i = next_node
        return root

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    # ... (Keep your existing KDTree endpoint building logic here) ...
    # (The KDTree logic is already efficient)

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
        groups.setdefault(root, []).append(i)

    merged_paths = []
    sq_threshold = distance_threshold ** 2 # Work with squared distance to avoid sqrt

    for group_indices in groups.values():
        if len(group_indices) == 1:
            merged_paths.append(open_paths[group_indices[0]])
            continue

        # Convert to deque for faster popping from both ends
        remaining = deque([list(open_paths[i].points) for i in group_indices])
        result = remaining.popleft()
        
        # We only try to merge paths that are actually within distance
        for _ in range(max_iterations):
            if not remaining:
                break
            
            best_idx = -1
            best_dist = sq_threshold + 1.0 # Only care about points within threshold
            best_conn = None

            r_start, r_end = result[0], result[-1]

            # Only check a subset of remaining if the list is huge to prevent hangs
            search_limit = 500 
            for idx, path in enumerate(remaining):
                if idx > search_limit: break 
                
                p_start, p_end = path[0], path[-1]
                
                # Connection checks
                conns = [
                    ((r_end[0]-p_start[0])**2 + (r_end[1]-p_start[1])**2, ("end", False)),
                    ((r_end[0]-p_end[0])**2 + (r_end[1]-p_end[1])**2, ("end", True)),
                    ((r_start[0]-p_start[0])**2 + (r_start[1]-p_start[1])**2, ("start", True)),
                    ((r_start[0]-p_end[0])**2 + (r_start[1]-p_end[1])**2, ("start", False))
                ]
                
                for d, conn_type in conns:
                    if d < best_dist:
                        best_dist, best_idx, best_conn = d, idx, conn_type
                        if d < 0.01: break # "Close enough" optimization

            if best_idx != -1:
                # We found a match, pop it by index from the deque
                # Note: Deque doesn't support indexed popping efficiently, 
                # but for smaller groups it's fine.
                path = list(remaining)[best_idx]
                del remaining[best_idx]
                
                connect_at, should_reverse = best_conn
                if should_reverse: path.reverse()

                if connect_at == "end":
                    result.extend(path[1:])
                else:
                    result = path[:-1] + result
            else:
                # No more close matches in this pass
                break

        merged_paths.append(Path(points=result))
        # Add whatever is left over that couldn't be merged
        for r in remaining:
            merged_paths.append(Path(points=r))

    return closed_paths + merged_paths


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def sort_paths_for_plotting(paths: List[Path]) -> List[Path]:
    """Sort paths to minimize pen travel distance.

    Uses a greedy nearest-neighbor approach with periodic KDTree rebuilds.

    Args:
        paths: List of paths to sort

    Returns:
        Sorted list of paths (may have some paths reversed)
    """
    if len(paths) <= 1:
        return paths

    n = len(paths)

    # For small numbers of paths, use simple O(n²) approach
    if n <= 50 or not KDTREE_AVAILABLE:
        return _sort_paths_simple(paths)

    # For very large numbers, skip sorting entirely (too slow)
    if n > 50000:
        return paths

    sorted_paths = []
    remaining_indices = set(range(n))
    current_pos = np.array([0.0, 0.0])

    # Rebuild KDTree periodically as paths are consumed
    rebuild_interval = max(100, n // 20)  # Rebuild every 5% of paths

    def build_tree_for_remaining():
        """Build KDTree for remaining (unused) paths only."""
        if not remaining_indices:
            return None, None, None

        # Map from tree index to original path index
        idx_list = list(remaining_indices)
        endpoints = []
        endpoint_to_path = []  # (original_path_idx, is_end)

        for orig_idx in idx_list:
            p = paths[orig_idx]
            endpoints.append(p.points[0])
            endpoint_to_path.append((orig_idx, False))
            endpoints.append(p.points[-1])
            endpoint_to_path.append((orig_idx, True))

        if not endpoints:
            return None, None, None

        return cKDTree(np.array(endpoints)), endpoint_to_path, idx_list

    tree, endpoint_map, idx_list = build_tree_for_remaining()
    paths_since_rebuild = 0

    while remaining_indices:
        # Rebuild tree periodically to maintain efficiency
        if paths_since_rebuild >= rebuild_interval and len(remaining_indices) > 50:
            tree, endpoint_map, idx_list = build_tree_for_remaining()
            paths_since_rebuild = 0

        if tree is None or len(remaining_indices) <= 50:
            # Fall back to simple approach for small remaining sets
            remaining_paths = [paths[i] for i in remaining_indices]
            if remaining_paths:
                sorted_remaining = _sort_paths_simple_from_pos(remaining_paths, current_pos)
                sorted_paths.extend(sorted_remaining)
            break

        # Query nearest neighbor
        _, nearest_idx = tree.query(current_pos, k=1)

        if nearest_idx < len(endpoint_map):
            path_idx, is_end = endpoint_map[nearest_idx]

            if path_idx in remaining_indices:
                if is_end and not paths[path_idx].is_closed:
                    sorted_paths.append(paths[path_idx].reverse())
                else:
                    sorted_paths.append(paths[path_idx])

                current_pos = np.array(sorted_paths[-1].points[-1])
                remaining_indices.remove(path_idx)
                paths_since_rebuild += 1
            else:
                # Path already used (stale tree), force rebuild
                tree, endpoint_map, idx_list = build_tree_for_remaining()
                paths_since_rebuild = 0
        else:
            # Index out of bounds, force rebuild
            tree, endpoint_map, idx_list = build_tree_for_remaining()
            paths_since_rebuild = 0

    return sorted_paths


def _sort_paths_simple_from_pos(paths: List[Path], start_pos: Tuple[float, float]) -> List[Path]:
    """Simple O(n²) path sorting starting from a given position."""
    if len(paths) <= 1:
        return paths

    sorted_paths = []
    remaining = list(range(len(paths)))
    current_pos = start_pos

    while remaining:
        best_idx = remaining[0]
        best_dist = float("inf")
        best_reversed = False
        best_remaining_idx = 0

        for ri, i in enumerate(remaining):
            p = paths[i]
            d_start = (current_pos[0] - p.points[0][0])**2 + (current_pos[1] - p.points[0][1])**2
            d_end = (current_pos[0] - p.points[-1][0])**2 + (current_pos[1] - p.points[-1][1])**2

            if d_start < best_dist:
                best_dist = d_start
                best_idx = i
                best_reversed = False
                best_remaining_idx = ri
            if d_end < best_dist and not p.is_closed:
                best_dist = d_end
                best_idx = i
                best_reversed = True
                best_remaining_idx = ri

        if best_reversed:
            sorted_paths.append(paths[best_idx].reverse())
        else:
            sorted_paths.append(paths[best_idx])

        current_pos = sorted_paths[-1].points[-1]
        remaining.pop(best_remaining_idx)

    return sorted_paths


def _sort_paths_simple(paths: List[Path]) -> List[Path]:
    """Simple O(n²) path sorting for small path counts, starting from origin."""
    return _sort_paths_simple_from_pos(paths, (0.0, 0.0))


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
