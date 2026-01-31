"""
Route generation from OSM way segments.
Connects individual OSM ways into actual runnable loops.
"""

import sqlite3
import math
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import heapq
import numpy as np


class RouteGraph:
    """Graph structure for OSM ways"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.nodes = {}  # node_id -> (lat, lng)
        self.edges = defaultdict(list)  # node_id -> [(next_node_id, way_id, distance)]
        self.ways = {}  # way_id -> way data

    def build_graph(self, center_lat: float, center_lng: float, radius_km: float = 2.0):
        """
        Build routing graph from OSM ways near a location.
        This creates a network where ways are connected at their endpoints.
        """
        print(f"Building routing graph around ({center_lat:.4f}, {center_lng:.4f})")

        # CRITICAL: Reset graph data structures to avoid accumulation across calls
        self.nodes = {}
        self.edges = defaultdict(list)
        self.ways = {}

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Calculate bounding box
        lat_offset = radius_km / 111.32
        lng_offset = radius_km / (111.32 * math.cos(math.radians(center_lat)))

        # Query ways in the area - focus on actual roads for better connectivity
        query = """
            SELECT
                osm_id,
                name,
                highway,
                length_meters,
                geometry_wkt
            FROM ways
            WHERE
                min_lat <= ? AND max_lat >= ?
                AND min_lon <= ? AND max_lon >= ?
                AND length_meters < 5000
                AND highway IN ('residential', 'tertiary', 'unclassified',
                               'secondary', 'primary', 'living_street', 'service')
        """

        cur.execute(query, (
            center_lat + lat_offset,
            center_lat - lat_offset,
            center_lng + lng_offset,
            center_lng - lng_offset
        ))

        ways_data = cur.fetchall()
        print(f"  Found {len(ways_data)} way segments in area")

        # Build graph structure - TWO PHASE APPROACH
        # Phase 1: Create all nodes and store way data
        node_counter = 0
        node_map = {}  # (lat, lng) -> node_id
        way_connections = []  # [(start_node, end_node, osm_id, length)]

        for osm_id, name, highway, length_m, geom_wkt in ways_data:
            # Parse coordinates
            coords = self._parse_wkt(geom_wkt)
            if len(coords) < 2:
                continue

            # Store way data
            self.ways[osm_id] = {
                'osm_id': osm_id,
                'name': name or f"{highway.title()} Segment",
                'highway': highway,
                'length': length_m,
                'coords': coords
            }

            # Get/create start and end nodes
            start_pos = self._round_coord(coords[0])
            end_pos = self._round_coord(coords[-1])

            if start_pos not in node_map:
                node_map[start_pos] = node_counter
                self.nodes[node_counter] = start_pos
                node_counter += 1

            if end_pos not in node_map:
                node_map[end_pos] = node_counter
                self.nodes[node_counter] = end_pos
                node_counter += 1

            start_node = node_map[start_pos]
            end_node = node_map[end_pos]

            # Store for phase 2
            way_connections.append((start_node, end_node, osm_id, length_m))

        # Phase 2: Create edges - ways are bidirectional
        for start_node, end_node, osm_id, length_m in way_connections:
            # Can traverse way in both directions
            self.edges[start_node].append((end_node, osm_id, length_m))
            if start_node != end_node:  # Avoid duplicate for loops
                self.edges[end_node].append((start_node, osm_id, length_m))

        conn.close()

        print(f"  Graph: {len(self.nodes)} nodes, {sum(len(e) for e in self.edges.values())} edges")

        # Find largest connected component
        self._find_largest_component()

        return len(self.nodes) > 0

    def _find_largest_component(self):
        """Find and keep only the largest connected component"""
        if not self.nodes:
            return

        visited = set()
        components = []

        # Find all connected components using BFS
        for start_node in list(self.nodes.keys()):
            if start_node in visited:
                continue

            # BFS from this node
            component = set()
            queue = [start_node]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue

                visited.add(node)
                component.add(node)

                # Add neighbors
                for next_node, _, _ in self.edges[node]:
                    if next_node not in visited:
                        queue.append(next_node)

            components.append(component)

        if not components:
            return

        # Keep only the largest component
        largest = max(components, key=len)
        print(f"  Found {len(components)} components, largest has {len(largest)} nodes")

        # Remove nodes not in largest component
        nodes_to_remove = set(self.nodes.keys()) - largest
        for node in nodes_to_remove:
            if node in self.nodes:
                del self.nodes[node]
            if node in self.edges:
                del self.edges[node]

        # Remove edges that point to removed nodes
        for node in list(self.edges.keys()):
            self.edges[node] = [
                (next_node, way_id, dist)
                for next_node, way_id, dist in self.edges[node]
                if next_node in self.nodes
            ]

        print(f"  Using largest component: {len(self.nodes)} nodes, {sum(len(e) for e in self.edges.values())} edges")

    def _parse_wkt(self, wkt: str) -> List[Tuple[float, float]]:
        """Parse WKT linestring"""
        coords_str = wkt.replace('LINESTRING(', '').replace(')', '')
        coords = []
        for pair in coords_str.split(','):
            lng, lat = map(float, pair.strip().split())
            coords.append((lat, lng))
        return coords

    def _round_coord(self, coord: Tuple[float, float], precision: int = 5) -> Tuple[float, float]:
        """
        Round coordinate for node matching.
        Using 5 decimal places (~1m precision) for accurate OSM way connections.
        """
        return (round(coord[0], precision), round(coord[1], precision))

    def find_nearest_node(self, lat: float, lng: float, min_degree: int = 2) -> Optional[int]:
        """
        Find the nearest well-connected graph node to a location.
        Prefers nodes with at least min_degree connections.
        """
        candidates = []

        for node_id, (node_lat, node_lng) in self.nodes.items():
            degree = len(self.edges[node_id])

            # Skip dead-ends and poorly connected nodes
            if degree < min_degree:
                continue

            dist = self._distance(lat, lng, node_lat, node_lng)
            candidates.append((dist, node_id, degree))

        if not candidates:
            # Fallback: accept any node
            for node_id, (node_lat, node_lng) in self.nodes.items():
                dist = self._distance(lat, lng, node_lat, node_lng)
                candidates.append((dist, node_id, len(self.edges[node_id])))

        if not candidates:
            return None

        # Sort by distance, but prefer higher-degree nodes if distance is similar
        candidates.sort(key=lambda x: (x[0], -x[2]))
        return candidates[0][1]

    def _distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points in meters"""
        dlat = (lat2 - lat1) * 111320
        dlng = (lng2 - lng1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
        return math.sqrt(dlat**2 + dlng**2)


def frechet_distance(route_coords: List[Tuple[float, float]],
                     shape_coords: List[Tuple[float, float]]) -> float:
    """
    Calculate discrete Fréchet distance between route and target shape.
    Lower is better (routes match shape more closely).
    """
    n = len(route_coords)
    m = len(shape_coords)

    # Dynamic programming matrix
    dp = np.full((n, m), -1.0)

    def distance(p1, p2):
        """Euclidean distance between two lat/lng points (approximate)"""
        dlat = (p1[0] - p2[0]) * 111320
        dlng = (p1[1] - p2[1]) * 111320 * math.cos(math.radians((p1[0] + p2[0]) / 2))
        return math.sqrt(dlat**2 + dlng**2)

    def compute(i, j):
        if dp[i][j] > -0.5:
            return dp[i][j]

        dist = distance(route_coords[i], shape_coords[j])

        if i == 0 and j == 0:
            dp[i][j] = dist
        elif i == 0:
            dp[i][j] = max(dist, compute(0, j - 1))
        elif j == 0:
            dp[i][j] = max(dist, compute(i - 1, 0))
        else:
            dp[i][j] = max(dist, min(compute(i - 1, j), compute(i, j - 1), compute(i - 1, j - 1)))

        return dp[i][j]

    return compute(n - 1, m - 1)


def generate_reference_circle(center_lat: float, center_lng: float,
                               radius_m: float, num_points: int = 36) -> List[Tuple[float, float]]:
    """
    Generate a reference circular shape for shape matching.
    Returns list of (lat, lng) tuples forming a circle.
    """
    points = []
    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points

        lat_offset = (radius_m / 111320) * math.cos(angle)
        lng_offset = (radius_m / (111320 * math.cos(math.radians(center_lat)))) * math.sin(angle)

        points.append((center_lat + lat_offset, center_lng + lng_offset))

    return points


def sample_shape_evenly(points: List[Tuple[float, float]], num_samples: int) -> List[Tuple[float, float]]:
    """
    Sample points evenly along a path by distance.

    Args:
        points: List of (x, y) points forming a path
        num_samples: Number of points to sample

    Returns:
        List of evenly spaced points along the path
    """
    if len(points) < 2:
        return points

    # Calculate cumulative distances
    distances = [0]
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        dist = math.sqrt(dx*dx + dy*dy)
        distances.append(distances[-1] + dist)

    total_distance = distances[-1]
    if total_distance == 0:
        return points[:num_samples]

    # Sample at even intervals
    sampled = []
    for i in range(num_samples):
        target_dist = (total_distance * i) / num_samples

        # Find segment containing this distance
        for j in range(len(distances) - 1):
            if distances[j] <= target_dist <= distances[j+1]:
                # Interpolate within this segment
                segment_start = distances[j]
                segment_end = distances[j+1]
                segment_length = segment_end - segment_start

                if segment_length > 0:
                    t = (target_dist - segment_start) / segment_length
                    x = points[j][0] + t * (points[j+1][0] - points[j][0])
                    y = points[j][1] + t * (points[j+1][1] - points[j][1])
                    sampled.append((x, y))
                else:
                    sampled.append(points[j])
                break

    return sampled


def calculate_straightness(points: List[Tuple[float, float]]) -> float:
    """
    Calculate how straight a path is.
    Returns ratio of max perpendicular distance to total path length.
    Lower values = straighter path. < 0.15 is very straight.
    """
    if len(points) < 3:
        return 0.0

    # Line from first to last point
    x1, y1 = points[0]
    x2, y2 = points[-1]

    line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if line_length == 0:
        return 0.0

    # Calculate max perpendicular distance from any point to the line
    max_perp_dist = 0
    for x, y in points[1:-1]:
        # Perpendicular distance from point to line
        # Using formula: |ax + by + c| / sqrt(a^2 + b^2)
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2

        perp_dist = abs(a * x + b * y + c) / math.sqrt(a * a + b * b)
        max_perp_dist = max(max_perp_dist, perp_dist)

    # Return ratio: max deviation / line length
    return max_perp_dist / line_length


def transform_user_shape_to_geo(canvas_shape: List[Dict], start_lat: float,
                                 start_lng: float, target_distance_m: float) -> List[Tuple[float, float]]:
    """
    Convert user's drawn shape from canvas coordinates to geographic coordinates.
    Samples evenly spaced waypoints along the path.

    Args:
        canvas_shape: List of {'x': float, 'y': float} from user's drawing
        start_lat, start_lng: Center point for the shape
        target_distance_m: Desired route distance (used for scaling)

    Returns:
        List of (lat, lng) tuples (evenly sampled waypoints)
    """
    if not canvas_shape or len(canvas_shape) < 3:
        raise ValueError("Shape must have at least 3 points")

    # Extract x, y coordinates
    points = [(p['x'], p['y']) for p in canvas_shape]

    # Detect if path is nearly straight
    straightness = calculate_straightness(points)

    # For straight paths, use fewer waypoints to avoid zigzagging
    # For complex shapes, use more waypoints for better matching
    if straightness < 0.05:
        # Nearly perfect straight line - just start and end
        num_waypoints = 3
        print(f"  Detected very straight path (straightness={straightness:.3f}), using {num_waypoints} waypoints")
    elif straightness < 0.15:
        # Somewhat straight - use minimal waypoints
        num_waypoints = 5
        print(f"  Detected straight path (straightness={straightness:.3f}), using {num_waypoints} waypoints")
    else:
        # Complex shape - use more waypoints for accurate matching
        num_waypoints = min(20, max(12, len(points) // 8))
        print(f"  Complex shape (straightness={straightness:.3f}), using {num_waypoints} waypoints")

    sampled_points = sample_shape_evenly(points, num_waypoints)

    print(f"  Shape transform: {len(points)} raw points → {len(sampled_points)} waypoints")

    # Calculate perimeter of sampled shape
    drawn_perimeter = sum(
        math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        for p1, p2 in zip(sampled_points, sampled_points[1:] + [sampled_points[0]])
    )

    if drawn_perimeter == 0:
        raise ValueError("Shape has zero perimeter")

    # Scale factor: target distance / drawn perimeter / detour factor
    # Detour factor accounts for roads being longer than straight lines
    detour_factor = 1.7
    scale_meters_per_pixel = target_distance_m / drawn_perimeter / detour_factor

    print(f"  Drawn perimeter: {drawn_perimeter:.1f}px, scale: {scale_meters_per_pixel:.2f} m/px")

    # Center the shape at origin
    center_x = sum(p[0] for p in sampled_points) / len(sampled_points)
    center_y = sum(p[1] for p in sampled_points) / len(sampled_points)
    centered = [(x - center_x, y - center_y) for x, y in sampled_points]

    # Convert to lat/lng offsets
    geo_points = []
    for x, y in centered:
        # Canvas: y increases downward, but we want north to be up
        # So flip y axis
        y_flipped = -y

        # Convert pixels to meters, then to lat/lng
        meters_x = x * scale_meters_per_pixel
        meters_y = y_flipped * scale_meters_per_pixel

        lat_offset = meters_y / 111320
        lng_offset = meters_x / (111320 * math.cos(math.radians(start_lat)))

        geo_points.append((start_lat + lat_offset, start_lng + lng_offset))

    return geo_points


class RouteGenerator:
    """Generate actual routes from OSM way segments"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.graph = RouteGraph(db_path)

    def generate_shape_matched_loop(
        self,
        start_lat: float,
        start_lng: float,
        target_shape: List[Tuple[float, float]],
        target_distance_m: float,
        tolerance: float = 0.15,
        num_candidates: int = 3
    ) -> Optional[Dict]:
        """
        Generate routes that follow a target shape.
        Uses the shape waypoints directly to create the route.

        Args:
            start_lat, start_lng: Starting location
            target_shape: List of (lat, lng) tuples defining desired shape
            target_distance_m: Desired distance
            tolerance: Distance tolerance
            num_candidates: Number of route variations to try

        Returns:
            Best matching route or None
        """
        print(f"\nGenerating shape-matched {target_distance_m}m loop")
        print(f"  Target shape has {len(target_shape)} points")

        # Calculate search radius based on shape extent
        lats = [p[0] for p in target_shape]
        lngs = [p[1] for p in target_shape]
        lat_range = max(lats) - min(lats)
        lng_range = max(lngs) - min(lngs)

        # Approximate radius in meters
        shape_extent_m = max(
            lat_range * 111320,
            lng_range * 111320 * math.cos(math.radians(start_lat))
        )
        search_radius = min(max(1.5, shape_extent_m / 1000 * 1.5), 5.0)

        print(f"  Shape extent: {shape_extent_m:.0f}m, search radius: {search_radius:.1f}km")

        # Build graph
        if not self.graph.build_graph(start_lat, start_lng, search_radius):
            print("  Failed to build graph")
            return None

        # Find nearest graph nodes to each shape waypoint
        waypoint_nodes = []
        for wp_lat, wp_lng in target_shape:
            node = self.graph.find_nearest_node(wp_lat, wp_lng, min_degree=1)
            if node is None:
                print(f"  Failed to find node near waypoint ({wp_lat:.5f}, {wp_lng:.5f})")
                return None
            waypoint_nodes.append(node)

        # Check if shape is closed or open
        # If start and end are close, it's a closed loop
        # If far apart, it's an open path (line) - create out-and-back
        start_point = target_shape[0]
        end_point = target_shape[-1]
        distance_to_close = self.graph._distance(
            start_point[0], start_point[1], end_point[0], end_point[1]
        )

        # Calculate average spacing between waypoints to determine threshold
        avg_spacing = sum(
            self.graph._distance(target_shape[i][0], target_shape[i][1],
                               target_shape[i+1][0], target_shape[i+1][1])
            for i in range(len(target_shape) - 1)
        ) / (len(target_shape) - 1)

        is_closed = distance_to_close < avg_spacing * 2  # Close if within 2x average spacing

        if is_closed:
            # Closed loop: connect back to start
            waypoint_nodes.append(waypoint_nodes[0])
            print(f"  Found graph nodes for all {len(target_shape)} waypoints (closed loop)")
        else:
            # Open path: Point A to Point B - NO RETURN!
            print(f"  Found graph nodes for all {len(target_shape)} waypoints (open path: point-to-point)")
            print(f"  Route will start at one end and finish at the other")

        # Connect consecutive waypoints via shortest paths
        all_way_tuples = []
        total_distance = 0

        for i in range(len(waypoint_nodes) - 1):
            from_node = waypoint_nodes[i]
            to_node = waypoint_nodes[i + 1]

            # Find shortest path between waypoints
            path = self._shortest_path(from_node, to_node, set())

            if path is None:
                print(f"  Failed to connect waypoint {i} to {i+1}")
                return None

            all_way_tuples.extend(path['ways'])
            total_distance += path['distance']

        # Check distance tolerance
        min_dist = target_distance_m * (1 - tolerance)
        max_dist = target_distance_m * (1 + tolerance)

        print(f"  Generated route: {total_distance:.0f}m (target: {target_distance_m:.0f}m)")

        if not (min_dist <= total_distance <= max_dist):
            print(f"  ⚠ Distance {total_distance:.0f}m outside tolerance ({min_dist:.0f}-{max_dist:.0f}m)")
            # Don't fail - accept it anyway since shape is more important
            # return None

        # Build route coordinates
        route_coords = self._build_route_coords(all_way_tuples)

        print(f"  ✓ Generated shape-matched route: {total_distance:.0f}m, {len(route_coords)} coords")

        return {
            'coordinates': route_coords,
            'distance_meters': total_distance,
            'way_ids': [w[0] for w in all_way_tuples],
            'num_segments': len(all_way_tuples)
        }

    def generate_circular_loop(
        self,
        start_lat: float,
        start_lng: float,
        target_distance_m: float,
        tolerance: float = 0.15
    ) -> Optional[Dict]:
        """
        Generate a circular loop by creating waypoints around a circle
        and connecting them via shortest paths.
        """
        import math

        print(f"\nGenerating circular {target_distance_m}m loop from ({start_lat:.4f}, {start_lng:.4f})")

        # Calculate radius for target circumference
        # Road networks are ~1.5-2x longer than straight-line circumference
        # Use C = 2πr but divide by detour factor
        detour_factor = 1.7  # Roads add ~70% to straight-line distance
        radius_m = target_distance_m / (2 * math.pi * detour_factor)
        radius_km = radius_m / 1000

        # Build graph with radius 1.5x the loop radius to ensure coverage
        search_radius = min(max(1.5, radius_km * 1.8), 5.0)
        print(f"  Loop radius: {radius_m:.0f}m, search radius: {search_radius:.1f}km")

        if not self.graph.build_graph(start_lat, start_lng, search_radius):
            print("  Failed to build graph")
            return None

        # Generate 8-12 waypoints around a circle (more for longer routes)
        num_waypoints = min(12, max(8, int(target_distance_m / 1000)))
        waypoints = []

        for i in range(num_waypoints):
            angle = (2 * math.pi * i) / num_waypoints

            # Convert radius to lat/lng offsets
            lat_offset = (radius_m / 111320) * math.cos(angle)
            lng_offset = (radius_m / (111320 * math.cos(math.radians(start_lat)))) * math.sin(angle)

            wp_lat = start_lat + lat_offset
            wp_lng = start_lng + lng_offset
            waypoints.append((wp_lat, wp_lng))

        print(f"  Generated {num_waypoints} waypoints around circle")

        # Find nearest graph nodes to each waypoint
        waypoint_nodes = []
        for wp_lat, wp_lng in waypoints:
            node = self.graph.find_nearest_node(wp_lat, wp_lng, min_degree=1)
            if node is None:
                print(f"  Failed to find node near waypoint ({wp_lat:.4f}, {wp_lng:.4f})")
                return None
            waypoint_nodes.append(node)

        # Add start node at the end to complete the loop
        start_node = waypoint_nodes[0]
        waypoint_nodes.append(start_node)

        print(f"  Found graph nodes for all waypoints")

        # Connect consecutive waypoints via shortest paths
        all_way_tuples = []
        total_distance = 0

        for i in range(len(waypoint_nodes) - 1):
            from_node = waypoint_nodes[i]
            to_node = waypoint_nodes[i + 1]

            # Find shortest path between waypoints
            path = self._shortest_path(from_node, to_node, set())

            if path is None:
                print(f"  Failed to connect waypoint {i} to {i+1}")
                return None

            all_way_tuples.extend(path['ways'])
            total_distance += path['distance']

        # Check distance tolerance
        min_dist = target_distance_m * (1 - tolerance)
        max_dist = target_distance_m * (1 + tolerance)

        if not (min_dist <= total_distance <= max_dist):
            print(f"  Distance {total_distance:.0f}m outside tolerance ({min_dist:.0f}-{max_dist:.0f}m)")
            return None

        # Build route coordinates
        route_coords = self._build_route_coords(all_way_tuples)

        print(f"  ✓ Generated circular route: {total_distance:.0f}m")

        return {
            'coordinates': route_coords,
            'distance_meters': total_distance,
            'way_ids': [w[0] for w in all_way_tuples],
            'num_segments': len(all_way_tuples)
        }

    def generate_loop(
        self,
        start_lat: float,
        start_lng: float,
        target_distance_m: float,
        tolerance: float = 0.15
    ) -> Optional[Dict]:
        """
        Generate a loop route starting and ending near the given location.

        Args:
            start_lat, start_lng: Starting point
            target_distance_m: Desired route length in meters
            tolerance: Acceptable distance variation (default 15%)

        Returns:
            Route dict with coordinates and metadata, or None if generation fails
        """
        # For longer routes (>4km), use circular waypoint approach
        # For shorter routes, use random walk
        if target_distance_m > 4000:
            print(f"  Using circular waypoint approach for long route ({target_distance_m/1000:.1f}km)")
            # Try multiple times with slightly different radii
            for attempt in range(5):
                # Vary radius slightly: 0.85, 0.925, 1.0, 1.075, 1.15
                radius_variation = 1.0 + (attempt - 2) * 0.075
                adjusted_distance = target_distance_m * radius_variation

                route = self.generate_circular_loop(
                    start_lat, start_lng, adjusted_distance, tolerance
                )

                if route:
                    return route

            print("  ✗ Circular approach failed, trying random walk fallback")

        # Random walk approach for shorter routes or as fallback
        print(f"\nGenerating {target_distance_m}m loop from ({start_lat:.4f}, {start_lng:.4f})")

        radius_km = min(max(1.5, target_distance_m / 1000 * 0.6), 5.0)
        print(f"  Using search radius: {radius_km:.1f}km")

        if not self.graph.build_graph(start_lat, start_lng, radius_km):
            print("  Failed to build graph")
            return None

        start_node = self.graph.find_nearest_node(start_lat, start_lng)
        if start_node is None:
            print("  No nearby nodes found")
            return None

        print(f"  Starting from node {start_node}")

        max_attempts = 25 if target_distance_m <= 3000 else 15

        for attempt in range(max_attempts):
            route = self._generate_random_walk_loop(
                start_node,
                target_distance_m,
                tolerance
            )

            if route:
                print(f"  ✓ Success on attempt {attempt + 1}")
                return route

        print(f"  ✗ Failed after {max_attempts} attempts")
        return None

    def _generate_random_walk_loop(
        self,
        start_node: int,
        target_distance_m: float,
        tolerance: float
    ) -> Optional[Dict]:
        """
        Generate a loop using random walk that tries to return to start.
        """
        current_node = start_node
        visited_ways = []  # List of (way_id, from_node, to_node) tuples
        total_distance = 0.0
        path_nodes = [start_node]
        max_iterations = 1000  # Prevent infinite loops

        # Phase 1: Random walk until ~45% of target distance
        # Shorter exploration creates more circular routes
        explore_distance = target_distance_m * 0.45

        # Get starting position for distance calculations
        start_pos = self.graph.nodes[start_node]

        iterations = 0
        backtrack_count = 0
        max_backtracks = 50

        while total_distance < explore_distance and iterations < max_iterations:
            iterations += 1

            # Get available edges from current node
            # Don't reuse ways (check just way_id, not direction)
            used_way_ids = [w[0] for w in visited_ways]
            available_edges = [
                (next_node, way_id, dist)
                for next_node, way_id, dist in self.graph.edges[current_node]
                if way_id not in used_way_ids
            ]

            if not available_edges:
                # Dead end - backtrack
                backtrack_count += 1
                if backtrack_count > max_backtracks or len(path_nodes) <= 1:
                    return None  # Too many backtracks or can't backtrack

                # Remove last segment
                if visited_ways:
                    last_way_tuple = visited_ways.pop()
                    last_way_id = last_way_tuple[0]
                    last_dist = self.graph.ways[last_way_id]['length']
                    total_distance -= last_dist

                path_nodes.pop()
                current_node = path_nodes[-1]
                continue

            # Pick next edge: bias toward creating circular routes
            if len(available_edges) > 1:
                scored_edges = []
                current_pos = self.graph.nodes[current_node]
                current_dist_from_start = self.graph._distance(
                    start_pos[0], start_pos[1], current_pos[0], current_pos[1]
                )

                for next_node, way_id, dist in available_edges:
                    next_pos = self.graph.nodes[next_node]
                    next_dist_from_start = self.graph._distance(
                        start_pos[0], start_pos[1], next_pos[0], next_pos[1]
                    )
                    next_degree = len(self.graph.edges[next_node])

                    # Create circular routes by managing distance from start
                    # Early phase: allow exploration
                    # Mid phase: prefer maintaining distance (go sideways)
                    # Late phase: prefer moving back toward start
                    progress = total_distance / explore_distance

                    if progress < 0.3:
                        # Early: free exploration
                        distance_penalty = 0
                    elif progress < 0.7:
                        # Mid: prefer edges that maintain or reduce distance from start
                        # This creates the "curve around" behavior
                        # Less aggressive for longer routes
                        penalty_multiplier = 3 if target_distance_m < 5000 else 2
                        if next_dist_from_start > current_dist_from_start + 100:
                            distance_penalty = (next_dist_from_start - current_dist_from_start) * penalty_multiplier
                        else:
                            distance_penalty = 0
                    else:
                        # Late: prefer edges that move back toward start
                        penalty_multiplier = 5 if target_distance_m < 5000 else 3
                        if next_dist_from_start > current_dist_from_start:
                            distance_penalty = (next_dist_from_start - current_dist_from_start) * penalty_multiplier
                        else:
                            # Bonus for moving closer
                            distance_penalty = -(current_dist_from_start - next_dist_from_start)

                    # Score: segment length + connectivity - distance penalty
                    score = dist * 0.5 + next_degree * 10 - distance_penalty
                    scored_edges.append((score, next_node, way_id, dist))

                scored_edges.sort(reverse=True)
                # Pick from top 3 to add randomness
                top_choices = scored_edges[:min(3, len(scored_edges))]
                _, next_node, way_id, dist = random.choice(top_choices)
            else:
                next_node, way_id, dist = available_edges[0]

            # Store way with direction information
            visited_ways.append((way_id, current_node, next_node))
            path_nodes.append(next_node)
            total_distance += dist
            current_node = next_node

        if iterations >= max_iterations:
            return None

        # Phase 2: Find shortest path back to start
        # Try to avoid reusing ways to create more circular routes
        used_way_ids = set(w[0] for w in visited_ways)
        return_path = self._shortest_path(current_node, start_node, used_way_ids)

        # If that fails, allow reusing ways
        if return_path is None:
            return_path = self._shortest_path(current_node, start_node, set())

        if return_path is None:
            return None

        # Combine paths (both are now tuples with direction)
        all_way_tuples = visited_ways + return_path['ways']
        total_distance += return_path['distance']

        # Check if distance is acceptable
        min_dist = target_distance_m * (1 - tolerance)
        max_dist = target_distance_m * (1 + tolerance)

        if not (min_dist <= total_distance <= max_dist):
            return None

        # Build final route coordinates
        route_coords = self._build_route_coords(all_way_tuples)

        return {
            'coordinates': route_coords,
            'distance_meters': total_distance,
            'way_ids': [w[0] for w in all_way_tuples],  # Extract just the IDs
            'num_segments': len(all_way_tuples)
        }

    def _shortest_path(
        self,
        start_node: int,
        end_node: int,
        excluded_ways: set
    ) -> Optional[Dict]:
        """
        Find shortest path between two nodes using Dijkstra's algorithm.
        Avoids ways in excluded_ways list.
        """
        # Dijkstra's algorithm
        distances = {start_node: 0}
        previous = {}
        previous_ways = {}
        queue = [(0, start_node)]
        visited = set()

        while queue:
            current_dist, current_node = heapq.heappop(queue)

            if current_node == end_node:
                # Reconstruct path with direction information
                path_ways = []
                node = end_node
                while node in previous:
                    prev_node = previous[node]
                    way_id = previous_ways[node]
                    # Store as (way_id, from_node, to_node)
                    path_ways.append((way_id, prev_node, node))
                    node = prev_node
                path_ways.reverse()

                return {
                    'ways': path_ways,
                    'distance': distances[end_node]
                }

            if current_node in visited:
                continue

            visited.add(current_node)

            for next_node, way_id, dist in self.graph.edges[current_node]:
                if way_id in excluded_ways:
                    continue

                new_dist = current_dist + dist

                if next_node not in distances or new_dist < distances[next_node]:
                    distances[next_node] = new_dist
                    previous[next_node] = current_node
                    previous_ways[next_node] = way_id
                    heapq.heappush(queue, (new_dist, next_node))

        return None

    def _build_route_coords(self, way_tuples: List) -> List[Dict]:
        """Build coordinate list from way tuples with direction info"""
        coords = []

        for idx, item in enumerate(way_tuples):
            if isinstance(item, tuple):
                way_id, from_node, to_node = item
            else:
                continue

            if way_id not in self.graph.ways:
                continue

            way_coords = list(self.graph.ways[way_id]['coords'])

            # Get exact node positions (these are the connection points)
            from_pos = self.graph.nodes[from_node]
            to_pos = self.graph.nodes[to_node]

            # Check which direction the way goes
            way_start_rounded = self.graph._round_coord(way_coords[0])
            way_end_rounded = self.graph._round_coord(way_coords[-1])

            # Check if the way actually connects these nodes
            forward_match = (from_pos == way_start_rounded and to_pos == way_end_rounded)
            reverse_match = (from_pos == way_end_rounded and to_pos == way_start_rounded)

            if not forward_match and not reverse_match:
                # This shouldn't happen with the graph reset fix, but skip if it does
                print(f"  ERROR: from_node {from_node} doesn't match way {way_id} endpoints!")
                print(f"    from_pos: {from_pos}")
                print(f"    way_start: {way_start_rounded}, way_end: {way_end_rounded}")
                print(f"  ERROR: Way {way_id} doesn't connect properly!")
                continue

            # Determine if we need to reverse
            if reverse_match:
                way_coords = list(reversed(way_coords))

            # Replace first and last coordinates with exact node positions to ensure continuity
            if len(way_coords) > 0:
                way_coords[0] = from_pos
                way_coords[-1] = to_pos

            # Check continuity with previous segment
            if coords and len(way_coords) > 0:
                prev_lat = coords[-1]['lat']
                prev_lng = coords[-1]['lng']
                curr_lat = way_coords[0][0]
                curr_lng = way_coords[0][1]
                gap = self.graph._distance(prev_lat, prev_lng, curr_lat, curr_lng)

                if gap > 50:  # More than 50 meter gap - something is wrong
                    print(f"  GAP: {gap:.0f}m between way {idx-1} and {idx} (way_id {way_id})")

            # Add coordinates (skip first if it matches last point from previous way)
            start_idx = 0
            if coords and len(way_coords) > 0:
                if abs(coords[-1]['lat'] - way_coords[0][0]) < 0.000001 and \
                   abs(coords[-1]['lng'] - way_coords[0][1]) < 0.000001:
                    start_idx = 1

            for lat, lng in way_coords[start_idx:]:
                coords.append({'lat': lat, 'lng': lng})

        return coords
