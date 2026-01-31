# Efficient Route Matching Pipeline - High Level Overview

## 1. Fréchet Distance: The Core Algorithm

### The Intuition ("Dog Walking Distance")

Imagine you're walking a dog along two different paths:

```
You:  A ──→ B ──→ C ──→ D
        ╲   │   ╱   │
         ╲  │  ╱    │  ← Leash stretches as you walk
          ╲ │ ╱     │
Dog:      E ──→ F ──→ G ──→ H
```

**Rules:**
1. Both you and your dog must walk forward (no backtracking)
2. You can each control your speed independently
3. The leash can stretch but both must complete your paths

**Fréchet distance = Minimum leash length needed**

### Why This Matters for Routes

```
Drawn Shape:     ╭──╮
                 │  │    ← User draws this
                 ╰──╯

Route Match:    ╭───╮
                │   │    ← Similar! Low Fréchet distance
                ╰───╯

Bad Match:      ─────     ← Different! High Fréchet distance
```

**Key Properties:**
- ✅ Respects the **flow** of the route (clockwise vs counter-clockwise matters)
- ✅ Captures **shape similarity** (not just endpoints)
- ✅ Works for loops, straight routes, figure-8s, etc.
- ✅ Handles routes of different sizes (with normalization)

### Simple Example

```python
# Two routes as point sequences
drawn_route = [(0,0), (1,0), (1,1), (0,1), (0,0)]  # Square
candidate_1 = [(0,0), (2,0), (2,2), (0,2), (0,0)]  # Bigger square → LOW distance
candidate_2 = [(0,0), (1,0), (0,1), (1,1), (0,0)]  # Different shape → HIGH distance

frechet_distance(drawn_route, candidate_1) = 0.3   # Good match!
frechet_distance(drawn_route, candidate_2) = 0.8   # Bad match
```

---

## 2. OSM Data: From PBF to Routing

### Step 1: OSM PBF Format

**What is OSM PBF?**
- Protocol Buffer Format - binary, compressed
- ~10x smaller than XML (planet: 70GB vs 1.5TB)
- ~5-10x faster to parse
- Contains: nodes (points), ways (roads), relations (areas)

```
OSM Data Structure:
├── Nodes: lat/lng coordinates
│   └── node id=123 lat=51.5074 lng=-0.1278
├── Ways: sequences of nodes (roads, paths)
│   └── way id=456 nodes=[123,124,125] tags={highway=residential}
└── Relations: collections of ways (routes, boundaries)
```

### Step 2: Extract Routing Data

**You DON'T query the raw PBF - you process it into a routing graph**

```
OSM PBF File (70GB)
    ↓
Filter for routing-relevant ways
    ↓
Build graph: nodes → edges
    ↓
Store in PostgreSQL + PostGIS (~20GB indexed)
```

**What to extract:**
```python
# Only keep ways suitable for running/cycling
relevant_tags = {
    'highway': ['footway', 'path', 'track', 'residential',
                'cycleway', 'pedestrian', 'living_street'],
    'access': ['yes', 'permissive', 'public'],
}

# Build graph
graph = {
    nodes: [(lat, lng), ...],
    edges: [(from_node, to_node, distance, surface_type), ...]
}
```

---

## 3. The Complete Pipeline (Step by Step)

### OFFLINE PREPROCESSING (Done Once)

```
┌──────────────────────────────────────────────────────────────┐
│ Step 1: Import OSM Data                                      │
│                                                               │
│ osm.pbf (70GB) → osmium-tool → PostgreSQL + PostGIS (25GB)  │
│                                                               │
│ Creates:                                                      │
│ - nodes table: id, lat, lng                                  │
│ - ways table: id, node_array, tags                          │
│ - edges table: from_node, to_node, cost (for routing)       │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 2: Pre-Generate Routes (Optional but FAST)             │
│                                                               │
│ For each popular location (city centers, parks):             │
│   - Generate loops: 1mi, 2mi, 3mi, 5mi, 10mi                │
│   - Use pgRouting to create actual runnable routes           │
│   - Store ~1000 routes per city                             │
│                                                               │
│ Total: ~50,000 pre-computed routes for UK                    │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 3: Calculate Shape Signatures                          │
│                                                               │
│ For each pre-computed route:                                 │
│   signature = {                                              │
│     bbox: [min_lat, min_lng, max_lat, max_lng],            │
│     distance: 2.3 miles,                                    │
│     aspect_ratio: 1.5,  # width/height                      │
│     turning_angles: [15°, -30°, 45°, ...],                  │
│     complexity: 0.7,  # 0=straight, 1=very twisty           │
│   }                                                          │
│                                                               │
│ Store in routes table with spatial index                     │
└──────────────────────────────────────────────────────────────┘
```

**Result:** Database ready for fast queries

---

### ONLINE REQUEST PROCESSING (Real-Time, <1 second)

```
User draws shape + picks location
         ↓
┌──────────────────────────────────────────────────────────────┐
│ STAGE 1: Normalize User's Shape                             │
│                                                               │
│ Input: [(100, 150), (120, 180), ...]  # Canvas pixels       │
│         ↓                                                    │
│ 1. Center at origin                                         │
│ 2. Scale to unit size (max dimension = 1)                   │
│ 3. Interpolate to 100 points (consistent comparison)        │
│         ↓                                                    │
│ Output: [(0.0, 0.1), (0.2, 0.3), ...]                       │
│                                                               │
│ Also calculate:                                              │
│ - Estimated distance from shape perimeter                    │
│ - Aspect ratio (wide vs tall)                               │
│ - Complexity score                                           │
│                                                               │
│ TIME: ~5ms                                                   │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ STAGE 2: Spatial Filter (Coarse)                            │
│                                                               │
│ SQL Query:                                                   │
│ SELECT * FROM routes                                         │
│ WHERE ST_DWithin(                                            │
│   start_point,                                              │
│   ST_Point(user_lng, user_lat),                            │
│   5000  -- 5km radius                                       │
│ )                                                            │
│ AND distance_meters BETWEEN                                  │
│   (estimated_distance * 0.7) AND                            │
│   (estimated_distance * 1.3)                                │
│                                                               │
│ Result: 5000 candidates (from millions)                     │
│ TIME: ~50ms (thanks to spatial index)                       │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ STAGE 3: Feature Filter (Medium)                            │
│                                                               │
│ For each of 5000 candidates:                                │
│   - Compare aspect ratio (filter ±30%)                      │
│   - Compare complexity score (filter ±40%)                  │
│   - Quick angle histogram match                             │
│                                                               │
│ Result: 200 candidates                                       │
│ TIME: ~100ms                                                 │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ STAGE 4: Fréchet Distance (Fine)                            │
│                                                               │
│ For each of 200 candidates:                                 │
│   1. Normalize candidate route (same as user shape)         │
│   2. Calculate Fréchet distance                             │
│      - Dynamic programming: O(n² log n)                     │
│      - With n=100 points: ~10k operations per route         │
│   3. Store (route, distance) pair                           │
│                                                               │
│ Parallelize: 8 threads × 25 routes each                     │
│                                                               │
│ Result: 200 routes with similarity scores                    │
│ TIME: ~400ms (parallelized)                                 │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ STAGE 5: Ranking & Return                                   │
│                                                               │
│ Sort by similarity score (lower Fréchet = better)           │
│ Apply bonus factors:                                         │
│   - Surface type (paved > unpaved)                          │
│   - Popularity (more used routes slightly boosted)          │
│   - Elevation gain preference                               │
│                                                               │
│ Return top 10-20 routes with metadata                        │
│                                                               │
│ TIME: ~5ms                                                   │
└──────────────────────────────────────────────────────────────┘
         ↓
    Return JSON to frontend
```

**TOTAL TIME: ~560ms** ✨

---

## 4. Fréchet Distance Algorithm (Simplified)

### The Dynamic Programming Approach

```python
def frechet_distance(curve_P, curve_Q):
    """
    Discrete Fréchet distance using DP

    curve_P: [(x1,y1), (x2,y2), ..., (xn,yn)]
    curve_Q: [(x1,y1), (x2,y2), ..., (xm,ym)]
    """
    n = len(curve_P)
    m = len(curve_Q)

    # DP table: ca[i][j] = Fréchet distance for P[0:i] and Q[0:j]
    ca = [[float('inf')] * m for _ in range(n)]

    # Base case
    ca[0][0] = euclidean_distance(curve_P[0], curve_Q[0])

    # Fill first column
    for i in range(1, n):
        ca[i][0] = max(
            ca[i-1][0],
            euclidean_distance(curve_P[i], curve_Q[0])
        )

    # Fill first row
    for j in range(1, m):
        ca[0][j] = max(
            ca[0][j-1],
            euclidean_distance(curve_P[0], curve_Q[j])
        )

    # Fill rest of table
    for i in range(1, n):
        for j in range(1, m):
            ca[i][j] = max(
                min(ca[i-1][j], ca[i][j-1], ca[i-1][j-1]),
                euclidean_distance(curve_P[i], curve_Q[j])
            )

    return ca[n-1][m-1]

def euclidean_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
```

### Visual Example

```
Curve P: A─B─C─D
Curve Q: E─F─G─H─I

DP Table (simplified):
      E    F    G    H    I
  ┌────────────────────────
A │ 0.2  0.3  0.5  0.7  0.9
B │ 0.3  0.2  0.4  0.6  0.8
C │ 0.5  0.4  0.3  0.5  0.7
D │ 0.7  0.6  0.5  0.4  0.6

Final Fréchet distance = ca[D][I] = 0.6
```

The algorithm finds the optimal "walking" path through this table that minimizes the maximum leash length.

---

## 5. Efficiency Optimizations

### Database Level

```sql
-- Spatial index for fast location queries
CREATE INDEX idx_routes_location
ON routes USING GIST(start_point);

-- Index on distance for range queries
CREATE INDEX idx_routes_distance
ON routes(distance_meters);

-- Composite index for combined queries
CREATE INDEX idx_routes_spatial_distance
ON routes USING GIST(start_point)
INCLUDE (distance_meters, aspect_ratio);

-- Partial index for popular routes (faster)
CREATE INDEX idx_popular_routes
ON routes(usage_count)
WHERE usage_count > 10;
```

### Application Level

```python
# 1. Connection pooling
from psycopg2.pool import ThreadedConnectionPool
db_pool = ThreadedConnectionPool(minconn=5, maxconn=20, dsn=DATABASE_URL)

# 2. Parallel Fréchet computation
from concurrent.futures import ThreadPoolExecutor

def compute_all_scores(user_shape, candidates):
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(frechet_distance, user_shape, candidate.geometry)
            for candidate in candidates
        ]
        scores = [f.result() for f in futures]
    return scores

# 3. Caching (Redis)
import redis
cache = redis.Redis()

def get_routes_near(lat, lng, radius):
    cache_key = f"routes:{lat:.4f}:{lng:.4f}:{radius}"

    # Check cache first
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Query database
    routes = db.query_spatial(lat, lng, radius)

    # Cache for 1 hour
    cache.setex(cache_key, 3600, json.dumps(routes))
    return routes

# 4. Early termination
def find_best_matches(user_shape, candidates, threshold=0.3):
    """Stop if we find enough good matches"""
    matches = []
    for candidate in candidates:
        score = frechet_distance(user_shape, candidate)
        if score < threshold:
            matches.append((candidate, score))
            if len(matches) >= 20:  # Stop early
                break
    return sorted(matches, key=lambda x: x[1])
```

### Memory Efficiency

```python
# Don't load all route geometries at once
def process_in_batches(candidates, batch_size=50):
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        # Load geometries only for this batch
        geometries = db.fetch_geometries([c.id for c in batch])
        yield from compute_scores(geometries)

# Use generators instead of lists
def filter_candidates(all_routes):
    """Memory-efficient filtering"""
    for route in all_routes:
        if meets_criteria(route):
            yield route  # Don't store all in memory
```

---

## 6. Tools & Setup

### Recommended Stack

```yaml
Data Storage:
  - PostgreSQL 15+ with PostGIS extension
  - pgRouting extension for route generation

Data Processing:
  - osmium-tool: for processing PBF files
  - osm2pgsql: import OSM into PostgreSQL

Backend:
  - Python 3.11+ with NumPy/SciPy for algorithms
  - FastAPI or Flask for API

Caching:
  - Redis for query caching

Optional:
  - Valhalla: alternative routing engine
  - OSRM: ultra-fast routing (C++)
```

### Quick Setup Commands

```bash
# 1. Install dependencies
sudo apt install postgresql-15-postgis-3 postgresql-15-pgrouting
pip install psycopg2 numpy scipy

# 2. Download OSM data (UK example)
wget https://download.geofabrik.de/europe/great-britain-latest.osm.pbf

# 3. Import to PostgreSQL
osm2pgsql -d gis --create --slim -G \
  --hstore --tag-transform-script openstreetmap-carto.lua \
  great-britain-latest.osm.pbf

# 4. Set up routing graph
psql -d gis -f /usr/share/postgresql/15/contrib/postgis-3.x/routing_setup.sql
```

---

## 7. Expected Performance

### Query Performance Breakdown

```
Total: ~560ms

├─ Spatial filter:     50ms  (1M routes → 5K candidates)
├─ Feature filter:     100ms (5K → 200 candidates)
├─ Fréchet compute:    400ms (200 routes, parallelized)
└─ Ranking:            10ms  (sort & format)
```

### Scaling

| Users | Routes | DB Size | Query Time | Server |
|-------|--------|---------|------------|--------|
| 10 | 50K | 5GB | 500ms | 2 core, 4GB RAM |
| 100 | 200K | 20GB | 600ms | 4 core, 8GB RAM |
| 1000 | 1M | 100GB | 800ms | 8 core, 16GB RAM + Redis |

### Optimization Potential

```
Current:  ~560ms average
With Redis caching: ~200ms (80% cache hit rate)
With pre-computed hashes: ~150ms
With GPU acceleration: ~50ms (overkill for hackathon)
```

---

## 8. Example Request Flow

```
1. User draws shape in browser
   └─> Canvas coordinates: [(100,150), (250,200), ...]

2. Frontend sends to backend:
   POST /api/match-routes
   {
     "location": {"lat": 51.5074, "lng": -0.1278},
     "shape": [{"x": 100, "y": 150}, {"x": 250, "y": 200}, ...]
   }

3. Backend processes:
   a) Normalize shape → [(0,0.1), (0.5,0.2), ...]
   b) Estimate distance → 2.3 miles
   c) Query database → 5000 nearby routes
   d) Feature filter → 200 candidates
   e) Fréchet matching → scores for each
   f) Rank and return top 10

4. Response sent back:
   {
     "routes": [
       {
         "id": "route_123",
         "name": "Thames Path Loop",
         "distance": 2.4,
         "match_score": 92,
         "coordinates": [{"lat": 51.507, "lng": -0.128}, ...]
       },
       ...
     ]
   }

5. Frontend renders routes on map
```

---

## Summary

**The key to efficiency:**

1. **Spatial indexing** eliminates 99.5% of routes immediately (millions → thousands)
2. **Feature filtering** eliminates 96% of remaining routes (thousands → hundreds)
3. **Fréchet distance** only on final candidates (hundreds, parallelized)
4. **Caching** avoids repeated work for popular locations

**Why Fréchet works:**
- It captures the "walking experience" of the route
- Respects direction and flow
- Works on any shape (loops, lines, figure-8s)
- Can be computed efficiently with DP

**Result:** Sub-second matching that feels "right" to users! 🎯
