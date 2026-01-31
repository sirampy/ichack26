# SQLite + SpatiaLite Setup (No Server Needed!)

## Why SQLite Works Great Here

✅ **No server to manage** - just a file
✅ **Still has spatial indexing** (R-tree)
✅ **Still has spatial queries** (ST_Distance, ST_DWithin, etc.)
✅ **Perfect for hackathons** - easy to deploy
✅ **Fast enough** for 50K-500K routes
✅ **Portable** - copy the .db file anywhere

**Limitations (but acceptable for MVP):**
- No parallel queries (but queries are fast enough)
- Single file (but easier to manage)
- ~100-200ms query time vs 50ms PostgreSQL (still great!)

---

## Setup (Much Simpler!)

### 1. Install Dependencies

```bash
# Install SpatiaLite tools
sudo apt install -y spatialite-bin libspatialite-dev gdal-bin

# Python libraries
pip install shapely numpy scipy

# For SQLite with spatial support
pip install pysqlite3-binary  # Has SpatiaLite built-in
# OR
pip install geoalchemy2  # If you want ORM
```

### 2. Process OSM PBF → SQLite

**Option A: Using ogr2ogr (Simplest)**

```bash
cd /home/alex/Documents/projects/ichack26/data

# Download London OSM (if you haven't)
wget https://download.geofabrik.de/europe/great-britain/england/greater-london-latest.osm.pbf

# Convert PBF to GeoPackage (intermediate format)
ogr2ogr \
  -f "GPKG" \
  london.gpkg \
  greater-london-latest.osm.pbf \
  lines \
  -where "highway IS NOT NULL"

# Convert GeoPackage to SQLite with SpatiaLite
ogr2ogr \
  -f "SQLite" \
  -dsco SPATIALITE=YES \
  routes.db \
  london.gpkg

# Takes ~3-5 minutes
# Output: routes.db (~200-500MB)
```

**Option B: Using Python + Osmium (More Control)**

```python
# process_osm.py
import sqlite3
import osmium
import shapely.wkb as wkblib
from shapely.geometry import LineString

class OSMHandler(osmium.SimpleHandler):
    def __init__(self, db_path):
        osmium.SimpleHandler.__init__(self)
        self.conn = sqlite3.connect(db_path)
        self.conn.enable_load_extension(True)
        self.conn.load_extension("mod_spatialite")
        self.setup_database()
        self.count = 0

    def setup_database(self):
        """Create tables with spatial index"""
        cur = self.conn.cursor()

        # Initialize spatial metadata
        cur.execute("SELECT InitSpatialMetadata(1)")

        # Create ways table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ways (
                id INTEGER PRIMARY KEY,
                osm_id INTEGER,
                highway TEXT,
                surface TEXT,
                name TEXT,
                length_meters REAL
            )
        """)

        # Add geometry column
        cur.execute("""
            SELECT AddGeometryColumn('ways', 'geometry', 4326, 'LINESTRING', 'XY')
        """)

        # Create spatial index
        cur.execute("""
            SELECT CreateSpatialIndex('ways', 'geometry')
        """)

        self.conn.commit()

    def way(self, w):
        """Process each way (road/path)"""
        # Only process runnable/walkable ways
        if 'highway' not in w.tags:
            return

        highway_type = w.tags['highway']
        if highway_type not in [
            'footway', 'path', 'track', 'steps',
            'pedestrian', 'cycleway', 'bridleway',
            'residential', 'living_street', 'service',
            'tertiary', 'unclassified'
        ]:
            return

        # Skip private/closed ways
        if 'access' in w.tags and w.tags['access'] in ['no', 'private']:
            return

        try:
            # Convert to Shapely LineString
            wkb = wkblib.loads(w.wkb, hex=True)
            if not wkb.is_valid or wkb.is_empty:
                return

            # Calculate length
            # Note: This is in degrees, not meters. For accurate length, use geography
            length_degrees = wkb.length
            length_meters = length_degrees * 111320  # Rough approximation

            # Insert into database
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO ways (osm_id, highway, surface, name, length_meters, geometry)
                VALUES (?, ?, ?, ?, ?, GeomFromText(?, 4326))
            """, (
                w.id,
                highway_type,
                w.tags.get('surface', 'unknown'),
                w.tags.get('name', ''),
                length_meters,
                wkb.wkt
            ))

            self.count += 1
            if self.count % 1000 == 0:
                print(f"Processed {self.count} ways...")
                self.conn.commit()

        except Exception as e:
            print(f"Error processing way {w.id}: {e}")

    def finish(self):
        self.conn.commit()
        self.conn.close()
        print(f"Finished! Processed {self.count} ways total")

# Run it
if __name__ == '__main__':
    handler = OSMHandler('routes.db')
    handler.apply_file('greater-london-latest.osm.pbf')
    handler.finish()
```

Install osmium:
```bash
pip install osmium
python process_osm.py
```

---

## 3. Create Routes Table

```python
# create_routes_table.py
import sqlite3

def setup_routes_table():
    conn = sqlite3.connect('data/routes.db')
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")

    cur = conn.cursor()

    # Create routes table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS routes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            distance_meters REAL,
            elevation_gain REAL DEFAULT 0,

            -- Shape signature
            aspect_ratio REAL,
            complexity_score REAL,

            -- Metadata
            surface_type TEXT,
            highway_type TEXT,
            is_loop INTEGER DEFAULT 0,
            usage_count INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add geometry columns
    cur.execute("""
        SELECT AddGeometryColumn('routes', 'geometry', 4326, 'LINESTRING', 'XY')
    """)

    cur.execute("""
        SELECT AddGeometryColumn('routes', 'start_point', 4326, 'POINT', 'XY')
    """)

    # Create spatial indices
    cur.execute("SELECT CreateSpatialIndex('routes', 'geometry')")
    cur.execute("SELECT CreateSpatialIndex('routes', 'start_point')")

    # Create regular indices
    cur.execute("CREATE INDEX IF NOT EXISTS idx_distance ON routes(distance_meters)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_aspect ON routes(aspect_ratio)")

    conn.commit()
    conn.close()
    print("Routes table created successfully!")

if __name__ == '__main__':
    setup_routes_table()
```

Run it:
```bash
python create_routes_table.py
```

---

## 4. Generate Sample Routes

```python
# generate_routes_sqlite.py
import sqlite3
import math

def connect_db():
    conn = sqlite3.connect('data/routes.db')
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    return conn

def generate_circular_route(center_lat, center_lng, radius_km, num_points=30):
    """Generate a circular route around a center point"""
    points = []

    for i in range(num_points + 1):  # +1 to close the loop
        angle = (i / num_points) * 2 * math.pi

        # Add some variation to make it more natural
        variation = radius_km * 0.2 * math.sin(angle * 3)
        actual_radius = radius_km + variation

        # Calculate point on circle
        lat = center_lat + (actual_radius / 111.32) * math.cos(angle)
        lng = center_lng + (actual_radius / (111.32 * math.cos(math.radians(center_lat)))) * math.sin(angle)

        points.append(f"{lng} {lat}")

    linestring_wkt = f"LINESTRING({', '.join(points)})"
    return linestring_wkt

def generate_routes_for_location(conn, name, lat, lng):
    """Generate multiple routes of different distances"""
    cur = conn.cursor()

    distances = [1, 2, 3, 5]  # km

    for distance_km in distances:
        # Generate route geometry
        route_wkt = generate_circular_route(lat, lng, distance_km / 2)

        # Calculate properties
        distance_meters = distance_km * 1000

        # Aspect ratio (roughly circular, so close to 1)
        aspect_ratio = 1.0 + (math.sin(distance_km) * 0.3)

        # Complexity (how twisty)
        complexity = 0.3 + (distance_km * 0.1)

        # Insert route
        cur.execute("""
            INSERT INTO routes (
                name,
                geometry,
                start_point,
                distance_meters,
                aspect_ratio,
                complexity_score,
                surface_type,
                highway_type,
                is_loop
            )
            VALUES (
                ?,
                GeomFromText(?, 4326),
                MakePoint(?, ?, 4326),
                ?,
                ?,
                ?,
                'paved',
                'path',
                1
            )
        """, (
            f"{name} - {distance_km}km Loop",
            route_wkt,
            lng, lat,
            distance_meters,
            aspect_ratio,
            complexity,
        ))

        print(f"  ✓ Generated {distance_km}km loop")

    conn.commit()

# Popular London locations
locations = [
    ('Trafalgar Square', 51.5074, -0.1278),
    ('Hyde Park', 51.5074, -0.1657),
    ('Regent\'s Park', 51.5313, -0.1568),
    ('Greenwich Park', 51.4768, -0.0005),
    ('Richmond Park', 51.4510, -0.2854),
    ('Hampstead Heath', 51.5573, -0.1657),
    ('Clapham Common', 51.4618, -0.1384),
    ('Victoria Park', 51.5341, -0.0389),
]

if __name__ == '__main__':
    conn = connect_db()

    print("Generating sample routes...")
    for name, lat, lng in locations:
        print(f"\n{name}:")
        generate_routes_for_location(conn, name, lat, lng)

    # Count total routes
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM routes")
    count = cur.fetchone()[0]

    print(f"\n✅ Generated {count} routes total")
    conn.close()
```

Run it:
```bash
python generate_routes_sqlite.py
```

---

## 5. Update Backend to Use SQLite

```python
# api/algorithms/route_matcher.py (updated for SQLite)
import sqlite3
import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from .shape_utils import normalize_shape, estimate_distance_from_shape
from .frechet import frechet_distance

class RouteMatcher:
    def __init__(self, db_path: str):
        """Initialize with SQLite database path"""
        self.db_path = db_path

    def _get_connection(self):
        """Get a database connection with SpatiaLite enabled"""
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        conn.load_extension("mod_spatialite")
        return conn

    def find_matching_routes(self,
                            user_shape: List[dict],
                            location: dict,
                            n_results: int = 10) -> List[Dict]:
        """Main function to find matching routes"""

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
        conn = self._get_connection()
        cur = conn.cursor()

        # Convert miles to meters
        distance_meters = distance_miles * 1609.34
        radius_meters = 8000  # 5 mile search radius

        # SpatiaLite spatial query
        cur.execute("""
            SELECT
                id,
                name,
                AsText(geometry) as geometry_wkt,
                distance_meters,
                aspect_ratio,
                complexity_score
            FROM routes
            WHERE Distance(
                start_point,
                MakePoint(?, ?, 4326)
            ) < ?
            AND distance_meters BETWEEN ? AND ?
            ORDER BY Distance(
                start_point,
                MakePoint(?, ?, 4326)
            )
            LIMIT 500
        """, (
            lng, lat, radius_meters / 111320,  # Convert meters to degrees (approx)
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

        conn.close()
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
            route['match_score'] = int(base_score)

        # Sort by match score (higher is better)
        scored.sort(key=lambda x: x['match_score'], reverse=True)

        return scored
```

---

## 6. Update API Endpoint

```python
# api/routes.py
from flask import jsonify, request
from . import api_bp
from .algorithms.route_matcher import RouteMatcher
import os

# Initialize route matcher with SQLite database
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'routes.db')
matcher = RouteMatcher(DB_PATH)

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
                'distance': round(match['distance_meters'] / 1609.34, 1),
                'duration': int(match['distance_meters'] / 1609.34 * 10),
                'match_score': match['match_score'],
                'elevation_gain': 0,
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

## Quick Start (Complete Commands)

```bash
# 1. Install dependencies
sudo apt install spatialite-bin libspatialite-dev gdal-bin
pip install shapely numpy scipy

# 2. Download OSM data
cd data
wget https://download.geofabrik.de/europe/great-britain/england/greater-london-latest.osm.pbf

# 3. Create database and generate routes
cd ..
python create_routes_table.py
python generate_routes_sqlite.py

# 4. Run the app
python app.py

# 5. Test
curl -X POST http://localhost:5000/api/match-routes \
  -H "Content-Type: application/json" \
  -d '{"location": {"lat": 51.5074, "lng": -0.1278}, "shape": [{"x": 100, "y": 100}, {"x": 200, "y": 200}]}'
```

---

## File Structure

```
ichack26/
├── data/
│   ├── greater-london-latest.osm.pbf (116MB)
│   └── routes.db (200-500MB) ← Single SQLite file!
│
├── api/
│   ├── algorithms/
│   │   ├── shape_utils.py
│   │   ├── frechet.py
│   │   └── route_matcher.py (updated for SQLite)
│   └── routes.py (updated)
│
├── create_routes_table.py
├── generate_routes_sqlite.py
└── app.py
```

---

## Performance Comparison

| Database | Setup Time | Query Time | Storage | Complexity |
|----------|------------|------------|---------|------------|
| PostgreSQL | 30 min | 50-100ms | 2-3GB | High |
| **SQLite** | **10 min** | **100-200ms** | **200-500MB** | **Low** |

**For a hackathon: SQLite wins! 🎯**

---

## Troubleshooting

### If `mod_spatialite` fails to load:

```python
# Try different extension names
conn.load_extension("mod_spatialite")  # Linux
# OR
conn.load_extension("mod_spatialite.so")  # Some Linux
# OR
conn.load_extension("libspatialite")  # Alternative
```

### Check SpatiaLite is installed:

```bash
spatialite routes.db "SELECT spatialite_version();"
```

### Manual SpatiaLite init:

```bash
spatialite routes.db
sqlite> SELECT InitSpatialMetadata(1);
sqlite> .quit
```
