# Implementation Plan - Route Matching System

## Implementation Order

### Phase 1: Database Setup (Day 1, ~2 hours)
Set up PostgreSQL with spatial extensions

### Phase 2: Import OSM Data (Day 1, ~1 hour)
Process Greater London PBF → PostgreSQL

### Phase 3: Build Routing Graph (Day 1-2, ~3 hours)
Extract walkable/runnable paths, create routing network

### Phase 4: Backend Algorithm (Day 2-3, ~6 hours)
Implement shape matching, Fréchet distance

### Phase 5: API Integration (Day 3, ~2 hours)
Connect algorithm to Flask endpoint

### Phase 6: Frontend Connection (Day 3, ~1 hour)
Wire up to existing UI

---

## PHASE 1: Database Setup

### 1.1 Install PostgreSQL + PostGIS + pgRouting

```bash
# Update package list
sudo apt update

# Install PostgreSQL 15
sudo apt install -y postgresql-15 postgresql-contrib-15

# Install PostGIS (spatial extension)
sudo apt install -y postgresql-15-postgis-3

# Install pgRouting (routing extension)
sudo apt install -y postgresql-15-pgrouting

# Install OSM import tools
sudo apt install -y osm2pgsql osmium-tool

# Python libraries for backend
pip install psycopg2-binary numpy scipy shapely
```

### 1.2 Create Database

```bash
# Switch to postgres user
sudo -u postgres psql

# In PostgreSQL shell:
CREATE DATABASE gis;
\c gis

-- Enable extensions
CREATE EXTENSION postgis;
CREATE EXTENSION pgrouting;
CREATE EXTENSION hstore;

-- Create user (replace 'youruser' with your username)
CREATE USER youruser WITH PASSWORD 'yourpassword';
GRANT ALL PRIVILEGES ON DATABASE gis TO youruser;
GRANT ALL ON ALL TABLES IN SCHEMA public TO youruser;

\q
```

### 1.3 Verify Setup

```bash
psql -d gis -c "SELECT PostGIS_version();"
psql -d gis -c "SELECT version FROM pgr_version();"
```

Expected output: Version numbers for PostGIS and pgRouting

---

## PHASE 2: Import OSM Data

### 2.1 Download London OSM Data

```bash
cd /home/alex/Documents/projects/ichack26
mkdir -p data
cd data

# Download Greater London (116 MB)
wget https://download.geofabrik.de/europe/great-britain/england/greater-london-latest.osm.pbf
```

### 2.2 Import with osm2pgsql

**IMPORTANT: This is the efficient format - data stays in PostgreSQL, NOT flat files**

```bash
# Import OSM data into PostgreSQL
# This creates tables: planet_osm_point, planet_osm_line, planet_osm_roads, planet_osm_polygon
osm2pgsql \
  --create \
  --database gis \
  --username youruser \
  --host localhost \
  --slim \
  --drop \
  --cache 2000 \
  --number-processes 4 \
  --hstore \
  --style /usr/share/osm2pgsql/default.style \
  greater-london-latest.osm.pbf

# This takes ~5-10 minutes for London
# Output will be in PostgreSQL tables, NOT files
```

**What this creates:**
- `planet_osm_point` - POIs, amenities
- `planet_osm_line` - Roads, paths, rivers (THIS IS WHAT WE NEED)
- `planet_osm_roads` - Major roads only
- `planet_osm_polygon` - Buildings, parks

### 2.3 Verify Import

```bash
psql -d gis -c "SELECT COUNT(*) FROM planet_osm_line WHERE highway IS NOT NULL;"
```

Expected: ~300,000-500,000 roads/paths in Greater London

---

## PHASE 3: Build Routing Graph

### 3.1 Create Custom Routes Table

We'll create a clean table optimized for our use case:

```sql
-- Run this in psql
psql -d gis

CREATE TABLE routes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    geometry GEOMETRY(LINESTRING, 4326),  -- The actual route path
    start_point GEOMETRY(POINT, 4326),    -- Where route starts

    -- Basic metrics
    distance_meters FLOAT,
    elevation_gain FLOAT DEFAULT 0,

    -- Shape signature (for fast filtering)
    bbox BOX2D,                -- Bounding box
    perimeter FLOAT,           -- Total perimeter
    aspect_ratio FLOAT,        -- width/height ratio
    complexity_score FLOAT,    -- How twisty (0-1)
    turning_angles FLOAT[],    -- Array of turn angles

    -- Metadata
    surface_type VARCHAR(50),  -- paved, unpaved, etc
    highway_type VARCHAR(50),  -- footway, path, track, etc
    is_loop BOOLEAN DEFAULT false,
    usage_count INT DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW()
);

-- Create spatial indices (CRITICAL for performance)
CREATE INDEX idx_routes_geom ON routes USING GIST(geometry);
CREATE INDEX idx_routes_start ON routes USING GIST(start_point);
CREATE INDEX idx_routes_distance ON routes(distance_meters);
CREATE INDEX idx_routes_bbox ON routes USING GIST(bbox);

-- Composite index for common queries
CREATE INDEX idx_routes_spatial_distance
ON routes USING GIST(start_point)
INCLUDE (distance_meters, aspect_ratio);
```

### 3.2 Create Routing Network

Extract only walkable/runnable ways:

```sql
-- Create a routing-ready network from OSM data
CREATE TABLE routing_network AS
SELECT
    osm_id,
    highway,
    surface,
    access,
    way as geometry,
    ST_Length(way::geography) as length_meters,
    ST_StartPoint(way) as start_node,
    ST_EndPoint(way) as end_node
FROM planet_osm_line
WHERE
    highway IN (
        'footway', 'path', 'track', 'steps',
        'pedestrian', 'cycleway', 'bridleway',
        'residential', 'living_street', 'service',
        'tertiary', 'unclassified'
    )
    AND (access IS NULL OR access NOT IN ('no', 'private'))
    AND way IS NOT NULL;

-- Add indices
CREATE INDEX idx_routing_geom ON routing_network USING GIST(geometry);
CREATE INDEX idx_routing_start ON routing_network USING GIST(start_node);
CREATE INDEX idx_routing_end ON routing_network USING GIST(end_node);

-- Check what we have
SELECT
    highway,
    COUNT(*) as count,
    AVG(length_meters) as avg_length
FROM routing_network
GROUP BY highway
ORDER BY count DESC;
```

### 3.3 Generate Sample Routes (MVP Approach)

For the hackathon, pre-generate routes from popular locations:

```python
# Save this as generate_routes.py
import psycopg2
from psycopg2.extras import execute_values
import numpy as np

# Connect to database
conn = psycopg2.connect("dbname=gis user=youruser password=yourpassword")
cur = conn.cursor()

# Popular London locations (lat, lng)
popular_locations = [
    (51.5074, -0.1278, 'Trafalgar Square'),
    (51.5194, -0.1270, 'Regent\'s Park'),
    (51.5034, -0.1195, 'St James\'s Park'),
    (51.5138, -0.0983, 'Liverpool Street'),
    (51.4816, -0.1916, 'Hyde Park'),
    # Add more as needed
]

def generate_loop_from_location(lat, lng, target_distance_km):
    """
    Generate a loop route from a location.
    Simple approach: find nearby ways and connect them into a loop.
    """

    # Query nearby ways
    cur.execute("""
        WITH nearby_ways AS (
            SELECT
                osm_id,
                geometry,
                length_meters,
                highway,
                surface
            FROM routing_network
            WHERE ST_DWithin(
                geometry::geography,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                %s  -- search radius
            )
            ORDER BY ST_Distance(
                geometry::geography,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
            )
            LIMIT 100
        )
        SELECT
            ST_AsText(ST_LineMerge(ST_Union(geometry))) as route_geom,
            SUM(length_meters) as total_length,
            STRING_AGG(DISTINCT surface, ',') as surfaces,
            STRING_AGG(DISTINCT highway, ',') as highways
        FROM nearby_ways;
    """, (lng, lat, target_distance_km * 1000, lng, lat))

    result = cur.fetchone()
    return result

# Generate routes for each location
for lat, lng, name in popular_locations:
    print(f"Generating routes for {name}...")

    for distance_km in [1, 2, 3, 5]:  # 1km, 2km, 3km, 5km loops
        route = generate_loop_from_location(lat, lng, distance_km)

        if route and route[0]:
            # Insert into routes table
            cur.execute("""
                INSERT INTO routes (
                    name,
                    geometry,
                    start_point,
                    distance_meters,
                    surface_type,
                    highway_type
                )
                VALUES (
                    %s,
                    ST_GeomFromText(%s, 4326),
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                    %s,
                    %s,
                    %s
                )
            """, (
                f"{name} - {distance_km}km Loop",
                route[0],
                lng, lat,
                route[1],
                route[2],
                route[3]
            ))

            print(f"  ✓ Generated {distance_km}km loop ({route[1]:.0f}m)")

conn.commit()
cur.close()
conn.close()

print("Route generation complete!")
```

Run it:
```bash
python generate_routes.py
```

---

## PHASE 4: Backend Algorithm Implementation

### 4.1 Create Algorithm Module

```bash
mkdir -p /home/alex/Documents/projects/ichack26/api/algorithms
touch /home/alex/Documents/projects/ichack26/api/algorithms/__init__.py
```

### 4.2 Shape Utilities

```python
# Save as api/algorithms/shape_utils.py
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class NormalizedShape:
    points: np.ndarray  # Normalized coordinates
    original_bounds: dict
    perimeter: float
    aspect_ratio: float
    complexity: float
    turning_angles: List[float]

def normalize_shape(points: List[dict]) -> NormalizedShape:
    """
    Normalize a user-drawn shape for comparison.

    Input: [{"x": 100, "y": 150}, {"x": 120, "y": 180}, ...]
    Output: Normalized shape with features
    """
    # Convert to numpy array
    coords = np.array([[p['x'], p['y']] for p in points], dtype=float)

    # Get bounds (for denormalization later)
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)

    # Center at origin
    centered = coords - coords.mean(axis=0)

    # Scale to unit size (max dimension = 1)
    scale = np.abs(centered).max()
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered

    # Interpolate to fixed number of points (100)
    normalized = interpolate_curve(normalized, n_points=100)

    # Calculate features
    perimeter = calculate_perimeter(normalized)
    aspect_ratio = (max_x - min_x) / (max_y - min_y) if (max_y - min_y) > 0 else 1.0
    complexity = calculate_complexity(normalized)
    turning_angles = calculate_turning_angles(normalized)

    return NormalizedShape(
        points=normalized,
        original_bounds={'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y},
        perimeter=perimeter,
        aspect_ratio=aspect_ratio,
        complexity=complexity,
        turning_angles=turning_angles
    )

def interpolate_curve(points: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Interpolate curve to have exactly n_points"""
    if len(points) < 2:
        return points

    # Calculate cumulative distance along curve
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_dist = np.concatenate([[0], np.cumsum(distances)])

    # Interpolate at equal intervals
    total_dist = cumulative_dist[-1]
    target_distances = np.linspace(0, total_dist, n_points)

    interpolated_x = np.interp(target_distances, cumulative_dist, points[:, 0])
    interpolated_y = np.interp(target_distances, cumulative_dist, points[:, 1])

    return np.column_stack([interpolated_x, interpolated_y])

def calculate_perimeter(points: np.ndarray) -> float:
    """Calculate total perimeter of the shape"""
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    return float(np.sum(distances))

def calculate_complexity(points: np.ndarray) -> float:
    """
    Calculate complexity score (0-1).
    0 = straight line, 1 = very twisty
    """
    if len(points) < 3:
        return 0.0

    angles = calculate_turning_angles(points)
    # Complexity is based on variation in turning angles
    angle_std = np.std(angles)
    # Normalize to 0-1 range (std of 60 degrees = complexity of 1)
    complexity = min(angle_std / 60.0, 1.0)
    return float(complexity)

def calculate_turning_angles(points: np.ndarray) -> List[float]:
    """Calculate turning angles at each point"""
    if len(points) < 3:
        return []

    angles = []
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]

        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])

        # Turning angle in degrees
        turn = np.degrees(angle2 - angle1)
        # Normalize to -180 to 180
        turn = ((turn + 180) % 360) - 180
        angles.append(turn)

    return angles

def estimate_distance_from_shape(points: List[dict], pixels_per_unit: float = 100) -> float:
    """
    Estimate real-world distance from drawn shape.
    Assumes user drew roughly to scale on canvas.
    """
    coords = np.array([[p['x'], p['y']] for p in points])
    distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    total_pixels = np.sum(distances)

    # Convert to miles (rough estimate)
    # Assume canvas represents roughly 5 miles max dimension
    estimated_miles = (total_pixels / 500) * 5
    return max(0.5, min(estimated_miles, 20))  # Clamp to 0.5-20 miles
```

### 4.3 Fréchet Distance Implementation

```python
# Save as api/algorithms/frechet.py
import numpy as np
from typing import List, Tuple

def frechet_distance(curve_p: np.ndarray, curve_q: np.ndarray) -> float:
    """
    Calculate discrete Fréchet distance between two curves.

    Args:
        curve_p: numpy array of shape (n, 2)
        curve_q: numpy array of shape (m, 2)

    Returns:
        Fréchet distance (float)
    """
    n = len(curve_p)
    m = len(curve_q)

    # DP table
    ca = np.full((n, m), np.inf)

    # Base case
    ca[0, 0] = euclidean_distance(curve_p[0], curve_q[0])

    # Fill first column
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], euclidean_distance(curve_p[i], curve_q[0]))

    # Fill first row
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], euclidean_distance(curve_p[0], curve_q[j]))

    # Fill rest of table
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]),
                euclidean_distance(curve_p[i], curve_q[j])
            )

    return ca[n-1, m-1]

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points"""
    return float(np.sqrt(np.sum((p1 - p2)**2)))

def fast_frechet_filter(user_shape: np.ndarray,
                        candidates: List[np.ndarray],
                        k: int = 50) -> List[Tuple[int, float]]:
    """
    Fast approximate Fréchet distance for initial filtering.
    Uses sampling to speed up computation.

    Returns: List of (index, distance) tuples for top-k candidates
    """
    # Sample every 5th point for speed
    user_sampled = user_shape[::5]

    scores = []
    for idx, candidate in enumerate(candidates):
        candidate_sampled = candidate[::5]
        dist = frechet_distance(user_sampled, candidate_sampled)
        scores.append((idx, dist))

    # Sort and return top-k
    scores.sort(key=lambda x: x[1])
    return scores[:k]
```

### 4.4 Main Matching Logic

```python
# Save as api/algorithms/route_matcher.py
import psycopg2
import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from .shape_utils import normalize_shape, estimate_distance_from_shape
from .frechet import frechet_distance

class RouteMatcher:
    def __init__(self, db_connection_string: str):
        self.conn = psycopg2.connect(db_connection_string)

    def find_matching_routes(self,
                            user_shape: List[dict],
                            location: dict,
                            n_results: int = 10) -> List[Dict]:
        """
        Main function to find matching routes.

        Args:
            user_shape: [{"x": 100, "y": 150}, ...]
            location: {"lat": 51.5074, "lng": -0.1278}
            n_results: Number of results to return

        Returns:
            List of route dictionaries with match scores
        """
        # STAGE 1: Normalize user shape
        normalized = normalize_shape(user_shape)
        estimated_distance_miles = estimate_distance_from_shape(user_shape)

        print(f"Estimated distance: {estimated_distance_miles:.1f} miles")
        print(f"Aspect ratio: {normalized.aspect_ratio:.2f}")
        print(f"Complexity: {normalized.complexity:.2f}")

        # STAGE 2: Spatial filter
        candidates = self._spatial_filter(
            location['lat'],
            location['lng'],
            estimated_distance_miles
        )
        print(f"Stage 2: {len(candidates)} candidates after spatial filter")

        # STAGE 3: Feature filter
        filtered = self._feature_filter(candidates, normalized)
        print(f"Stage 3: {len(filtered)} candidates after feature filter")

        # STAGE 4: Fréchet distance matching
        scored = self._compute_frechet_scores(normalized.points, filtered)
        print(f"Stage 4: Scored {len(scored)} routes")

        # STAGE 5: Rank and return
        ranked = self._rank_routes(scored, normalized)
        return ranked[:n_results]

    def _spatial_filter(self, lat: float, lng: float, distance_miles: float) -> List[Dict]:
        """Query database for nearby routes"""
        cur = self.conn.cursor()

        # Convert miles to meters
        distance_meters = distance_miles * 1609.34
        radius_meters = 8000  # 5 mile search radius

        cur.execute("""
            SELECT
                id,
                name,
                ST_AsText(geometry) as geometry_wkt,
                distance_meters,
                aspect_ratio,
                complexity_score
            FROM routes
            WHERE ST_DWithin(
                start_point::geography,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                %s
            )
            AND distance_meters BETWEEN %s AND %s
            ORDER BY ST_Distance(
                start_point::geography,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
            )
            LIMIT 500
        """, (
            lng, lat, radius_meters,
            distance_meters * 0.7, distance_meters * 1.3,
            lng, lat
        ))

        results = []
        for row in cur.fetchall():
            results.append({
                'id': row[0],
                'name': row[1],
                'geometry_wkt': row[2],
                'distance_meters': row[3],
                'aspect_ratio': row[4],
                'complexity_score': row[5]
            })

        return results

    def _feature_filter(self, candidates: List[Dict], user_shape) -> List[Dict]:
        """Filter by shape features"""
        filtered = []

        for candidate in candidates:
            # Aspect ratio check (±50%)
            if candidate['aspect_ratio']:
                ratio_diff = abs(candidate['aspect_ratio'] - user_shape.aspect_ratio)
                if ratio_diff > user_shape.aspect_ratio * 0.5:
                    continue

            # Complexity check (±50%)
            if candidate['complexity_score']:
                complexity_diff = abs(candidate['complexity_score'] - user_shape.complexity)
                if complexity_diff > 0.5:
                    continue

            filtered.append(candidate)

        return filtered[:200]  # Limit to 200 for Fréchet

    def _compute_frechet_scores(self, user_points: np.ndarray, candidates: List[Dict]) -> List[Dict]:
        """Compute Fréchet distance for each candidate"""

        def score_route(candidate):
            # Parse WKT geometry and normalize
            route_points = self._parse_wkt_linestring(candidate['geometry_wkt'])
            route_normalized = self._normalize_route_coords(route_points)

            # Compute Fréchet distance
            distance = frechet_distance(user_points, route_normalized)
            candidate['frechet_distance'] = distance
            return candidate

        # Parallel computation
        with ThreadPoolExecutor(max_workers=8) as executor:
            scored = list(executor.map(score_route, candidates))

        return scored

    def _parse_wkt_linestring(self, wkt: str) -> np.ndarray:
        """Parse WKT LINESTRING to numpy array"""
        # Remove "LINESTRING(" and ")"
        coords_str = wkt.replace('LINESTRING(', '').replace(')', '')
        coords = []
        for pair in coords_str.split(','):
            lng, lat = map(float, pair.strip().split())
            coords.append([lng, lat])
        return np.array(coords)

    def _normalize_route_coords(self, coords: np.ndarray) -> np.ndarray:
        """Normalize route coordinates same as user shape"""
        centered = coords - coords.mean(axis=0)
        scale = np.abs(centered).max()
        if scale > 0:
            normalized = centered / scale
        else:
            normalized = centered

        # Interpolate to 100 points
        from .shape_utils import interpolate_curve
        return interpolate_curve(normalized, n_points=100)

    def _rank_routes(self, scored: List[Dict], user_shape) -> List[Dict]:
        """Final ranking with bonus factors"""
        for route in scored:
            # Base score from Fréchet (convert to 0-100 scale)
            base_score = 100 * (1 - min(route['frechet_distance'], 1.0))

            # Apply bonuses (small adjustments)
            # TODO: Add surface_type, popularity bonuses

            route['match_score'] = int(base_score)

        # Sort by match score (higher is better)
        scored.sort(key=lambda x: x['match_score'], reverse=True)

        return scored
```

---

## PHASE 5: API Integration

Update `api/routes.py`:

```python
# Add to api/routes.py
from flask import jsonify, request
from . import api_bp
from .algorithms.route_matcher import RouteMatcher

# Initialize route matcher
matcher = RouteMatcher("dbname=gis user=youruser password=yourpassword")

@api_bp.route('/match-routes', methods=['POST'])
def match_routes():
    """Match routes based on drawn shape and location"""
    data = request.get_json()

    if not data or 'location' not in data or 'shape' not in data:
        return jsonify({'error': 'Missing location or shape data'}), 400

    try:
        # Find matching routes
        matches = matcher.find_matching_routes(
            user_shape=data['shape'],
            location=data['location'],
            n_results=10
        )

        # Format for frontend
        formatted_routes = []
        for match in matches:
            route_coords = matcher._parse_wkt_linestring(match['geometry_wkt'])

            formatted_routes.append({
                'id': match['id'],
                'name': match['name'],
                'distance': round(match['distance_meters'] / 1609.34, 1),  # Convert to miles
                'duration': int(match['distance_meters'] / 1609.34 * 10),  # Rough estimate
                'match_score': match['match_score'],
                'elevation_gain': 0,  # TODO: Calculate from elevation data
                'coordinates': [
                    {'lat': float(coord[1]), 'lng': float(coord[0])}
                    for coord in route_coords
                ]
            })

        return jsonify({
            'routes': formatted_routes,
            'count': len(formatted_routes)
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
```

---

## PHASE 6: Testing

### Test the Complete Pipeline

```python
# test_pipeline.py
import requests
import json

# Test shape (simple square)
test_shape = [
    {"x": 100, "y": 100},
    {"x": 200, "y": 100},
    {"x": 200, "y": 200},
    {"x": 100, "y": 200},
    {"x": 100, "y": 100}
]

# Test location (Trafalgar Square)
test_location = {
    "lat": 51.5074,
    "lng": -0.1278
}

# Send request
response = requests.post('http://localhost:5000/api/match-routes', json={
    'shape': test_shape,
    'location': test_location
})

print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
```

---

## File Structure Summary

```
ichack26/
├── data/
│   └── greater-london-latest.osm.pbf (116MB - downloaded)
│
├── api/
│   ├── __init__.py
│   ├── routes.py (updated with real matching)
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── shape_utils.py (NEW)
│   │   ├── frechet.py (NEW)
│   │   └── route_matcher.py (NEW)
│   └── schemas.py
│
├── static/
│   ├── js/app.js (already working)
│   └── css/style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── generate_routes.py (NEW - for pre-computing routes)
├── test_pipeline.py (NEW - for testing)
└── requirements.txt (update with new dependencies)
```

---

## Updated requirements.txt

```txt
flask
flask-cors
psycopg2-binary
numpy
scipy
shapely
```

---

## Quick Start Commands

```bash
# 1. Install everything
sudo apt update && sudo apt install -y postgresql-15 postgresql-15-postgis-3 postgresql-15-pgrouting osm2pgsql osmium-tool
pip install flask flask-cors psycopg2-binary numpy scipy shapely

# 2. Set up database
sudo -u postgres psql -c "CREATE DATABASE gis;"
sudo -u postgres psql -d gis -c "CREATE EXTENSION postgis; CREATE EXTENSION pgrouting; CREATE EXTENSION hstore;"

# 3. Download and import OSM data
cd data
wget https://download.geofabrik.de/europe/great-britain/england/greater-london-latest.osm.pbf
osm2pgsql --create --database gis --slim --drop --cache 2000 --hstore greater-london-latest.osm.pbf

# 4. Create routes table and generate sample routes
psql -d gis < create_routes_table.sql
python generate_routes.py

# 5. Run the app
python app.py

# 6. Test
python test_pipeline.py
```

---

## Expected Timeline

- **Day 1**: Database setup + OSM import (3-4 hours)
- **Day 2**: Algorithm implementation (6-8 hours)
- **Day 3**: Integration + testing (3-4 hours)
- **Total**: ~15 hours of focused work

## Storage Summary

- OSM PBF file: 116 MB (original)
- PostgreSQL database: ~2-3 GB (imported + indexed)
- **Efficient format: PostgreSQL with PostGIS indices** ← This is your answer
- No need for flat files - database IS the efficient format!
