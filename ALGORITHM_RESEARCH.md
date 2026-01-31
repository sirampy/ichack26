# Route Shape Matching Algorithm Research

## Problem Statement
Given a user-drawn shape and a location, find real-world routes (from OSM data) that are visually/geometrically similar to the drawn shape.

---

## 1. Shape Similarity Algorithms

### Option A: Fréchet Distance ⭐ RECOMMENDED
**What it is:** Measures similarity between curves considering point ordering. Often described as the "dog walking distance" - imagine walking a dog where you both must move forward along your respective paths.

**Pros:**
- Preserves path topology and ordering
- Rotation/translation invariant with preprocessing
- Well-studied with efficient implementations
- Captures the "feel" of following a route

**Cons:**
- O(n² log n) complexity with optimization
- Can be sensitive to point density
- Doesn't consider route feasibility

**Best for:** Matching the overall shape and flow of routes

**Implementation:**
```python
# Discrete Fréchet Distance
def frechet_distance(curve1, curve2):
    # Dynamic programming approach
    # Returns distance metric
```

### Option B: Dynamic Time Warping (DTW)
**What it is:** Finds optimal alignment between two time series by allowing stretching/compression.

**Pros:**
- Handles routes of different lengths naturally
- Can match similar shapes at different scales
- Fast approximations available (FastDTW)

**Cons:**
- Can over-match dissimilar routes
- Computationally expensive O(n²)
- May not respect geometric constraints

**Best for:** When routes might have similar "feel" but different distances

### Option C: Hausdorff Distance
**What it is:** Maximum distance from any point in one shape to nearest point in the other.

**Pros:**
- Simple to compute
- Intuitive interpretation
- Fast: O(n log n) with spatial indexing

**Cons:**
- Very sensitive to outliers
- Doesn't consider path ordering
- Can miss overall shape similarity

**Best for:** Quick coarse filtering before expensive matching

### Option D: Turning Angle Sequence Matching
**What it is:** Compare sequences of turn angles along the route.

**Pros:**
- Rotation/translation invariant by nature
- Captures the "experience" of the route
- Fast to compute
- Good for matching "twisty" vs "straight" routes

**Cons:**
- Loses absolute position information
- Sensitive to sampling rate
- Multiple shapes can have similar angle sequences

**Best for:** Supplementary metric, especially for runners who care about "flat" vs "twisty"

### Option E: Shape Context / Fourier Descriptors
**What it is:** Represent shape in frequency domain or with local context histograms.

**Pros:**
- Rotation/scale invariant
- Compact representation
- Fast matching in feature space

**Cons:**
- Loses some geometric detail
- Complex to implement correctly
- May not capture local features well

**Best for:** Pre-filtering in large databases

---

## 2. OSM Data Strategy

### Option A: Overpass API (Online Querying)

**Architecture:**
```
User Request → Query Overpass → Parse OSM data → Build graph → Match routes
```

**Pros:**
- Always fresh data
- No local storage needed
- Simple deployment
- Good for MVP/prototype

**Cons:**
- Rate limited (typical: 2 requests/sec)
- Network latency (2-10 seconds per query)
- Requires internet connection
- Can't handle high traffic
- May time out on large area queries

**Cost:** Free but rate-limited

**When to use:** Early development, low-traffic apps, small geographic areas

### Option B: Offline OSM Dump ⭐ RECOMMENDED FOR PRODUCTION

**Architecture:**
```
OSM Planet/Extract → PostgreSQL + PostGIS → pgRouting → Query engine
```

**Pros:**
- Fast queries (<100ms)
- No rate limits
- Offline operation
- Full control over data
- Can pre-compute routes

**Cons:**
- Large storage (10GB-1TB depending on area)
- Complex setup (Docker helps)
- Needs periodic updates
- Requires database expertise

**Storage Requirements:**
- UK: ~20GB
- Europe: ~100GB
- Planet: ~1TB

**When to use:** Production, high traffic, need for speed

### Option C: Hybrid Approach

**Architecture:**
```
Local routing graph + Overpass for fresh POI/metadata
```

**Pros:**
- Fast routing on local data
- Fresh POI information
- Smaller storage than full planet

**Cons:**
- Complex to manage
- Still has network dependency
- Cache invalidation challenges

---

## 3. Overall System Architecture

### RECOMMENDED: Multi-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Preprocessing (Offline, One-Time)                  │
│ - Import OSM data (roads, paths, trails)                    │
│ - Filter for runnable/cyclable ways                         │
│ - Build routing graph with pgRouting                        │
│ - Pre-generate popular routes from various start points     │
│ - Calculate shape signatures for each route                 │
│ - Store in PostGIS with spatial indices                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Request Processing (Real-Time)                     │
│ 1. Normalize user's drawn shape                             │
│    - Center, scale, interpolate to N points                 │
│    - Extract features: bounding box, perimeter, complexity  │
│                                                              │
│ 2. Coarse Spatial Filter                                    │
│    - Query routes near user location (R-tree index)         │
│    - Filter by distance range (±50% of drawn distance)      │
│    - Filter by bounding box aspect ratio                    │
│    - Reduces candidates from millions to ~1000              │
│                                                              │
│ 3. Feature-Based Pre-Filter                                 │
│    - Compare coarse shape metrics                           │
│    - Turning angle distribution                             │
│    - Complexity score                                       │
│    - Reduces to ~100 candidates                             │
│                                                              │
│ 4. Fine-Grained Matching                                    │
│    - Compute Fréchet distance for each candidate            │
│    - Score based on similarity                              │
│    - Consider secondary factors (elevation, surface)        │
│                                                              │
│ 5. Ranking & Return                                         │
│    - Sort by combined score                                 │
│    - Return top 10-20 matches                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Algorithm Design

### Shape Normalization
```python
def normalize_shape(points):
    """
    Normalize drawn shape for comparison
    """
    # 1. Convert to numpy array
    points = np.array(points)

    # 2. Center at origin
    centroid = points.mean(axis=0)
    centered = points - centroid

    # 3. Scale to unit size (max dimension = 1)
    scale = np.abs(centered).max()
    normalized = centered / scale

    # 4. Interpolate to fixed number of points (e.g., 100)
    # This ensures consistent comparison
    interpolated = interpolate_curve(normalized, n_points=100)

    # 5. Rotate to canonical orientation (optional)
    # Align longest axis with x-axis

    return interpolated, centroid, scale

def extract_shape_features(points):
    """
    Extract compact shape descriptor
    """
    features = {
        'perimeter': compute_perimeter(points),
        'area': compute_area(points),  # if closed curve
        'compactness': 4 * pi * area / perimeter^2,
        'aspect_ratio': bounding_box_width / height,
        'turning_angles': compute_turning_angles(points),
        'complexity': count_direction_changes(points),
        'fourier_descriptors': compute_fourier_desc(points)[:10]
    }
    return features
```

### Route Generation Strategies

**Strategy 1: Pre-Computed Route Database** (FAST)
```python
# Offline: Pre-generate routes
for start_point in grid_of_popular_locations:
    for distance in [1, 2, 3, 5, 10]:  # miles
        routes = generate_loops(start_point, distance)
        for route in routes:
            signature = extract_shape_features(route)
            store_in_db(route, signature)

# Online: Query pre-computed routes
candidates = db.query(
    near=user_location,
    distance_range=(target_dist * 0.7, target_dist * 1.3)
)
```

**Strategy 2: Dynamic Route Generation** (FLEXIBLE)
```python
def generate_matching_route(start_point, shape, distance):
    """
    Use A* with shape-following heuristic
    """
    # Heuristic: prefer moves that follow the drawn shape
    def heuristic(current, goal, drawn_shape):
        # Distance to nearest point on drawn shape
        shape_dist = min_distance_to_shape(current, drawn_shape)

        # Prefer directions that align with shape
        next_shape_point = get_next_shape_point(current, drawn_shape)
        direction_score = angle_similarity(current, next_shape_point)

        return shape_dist * 0.7 + direction_score * 0.3

    # A* search with custom heuristic
    path = astar(
        start=start_point,
        goal=start_point,  # loop
        heuristic=heuristic,
        target_length=distance
    )
    return path
```

**Strategy 3: Sample & Rank** ⭐ RECOMMENDED HYBRID
```python
def find_matching_routes(location, shape, n_results=10):
    """
    Hybrid approach: pre-computed + sampling
    """
    # Step 1: Quick spatial filter
    candidates = db.query_spatial(
        location=location,
        radius=5_miles,
        distance_range=(shape.length * 0.8, shape.length * 1.2)
    )

    # Step 2: Feature-based filter
    shape_features = extract_features(shape)
    filtered = []
    for route in candidates:
        feature_distance = feature_similarity(
            shape_features,
            route.features
        )
        if feature_distance < THRESHOLD:
            filtered.append(route)

    # Step 3: Expensive shape matching (only on filtered set)
    scored = []
    for route in filtered:
        score = frechet_distance(
            normalize(shape),
            normalize(route.geometry)
        )
        scored.append((route, score))

    # Step 4: Rank and return
    scored.sort(key=lambda x: x[1])  # Lower distance = better match
    return scored[:n_results]
```

---

## 5. Similarity Scoring Function

```python
def compute_similarity_score(drawn_shape, candidate_route):
    """
    Multi-factor similarity score
    """
    # Normalize both shapes
    shape_norm = normalize_shape(drawn_shape)
    route_norm = normalize_shape(candidate_route.geometry)

    # Geometric similarity (0-1, higher is better)
    frechet = frechet_distance(shape_norm, route_norm)
    geometric_score = 1 / (1 + frechet)  # Convert to 0-1 range

    # Turning angle similarity
    shape_angles = compute_turning_angles(shape_norm)
    route_angles = compute_turning_angles(route_norm)
    angle_score = angle_sequence_similarity(shape_angles, route_angles)

    # Distance match (prefer routes close to drawn distance)
    drawn_length = estimate_length(drawn_shape)
    actual_length = candidate_route.distance
    distance_score = 1 - abs(drawn_length - actual_length) / drawn_length

    # Complexity match (simple vs twisty)
    shape_complexity = compute_complexity(drawn_shape)
    route_complexity = candidate_route.complexity
    complexity_score = 1 - abs(shape_complexity - route_complexity)

    # Quality factors (optional bonuses)
    surface_bonus = 1.1 if candidate_route.surface == 'paved' else 1.0
    popularity_bonus = 1 + (candidate_route.usage_count / 1000)

    # Weighted combination
    final_score = (
        geometric_score * 0.50 +      # Most important
        angle_score * 0.25 +
        distance_score * 0.15 +
        complexity_score * 0.10
    ) * surface_bonus * popularity_bonus

    return final_score * 100  # Return as percentage
```

---

## 6. Database Schema

```sql
-- Routes table
CREATE TABLE routes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    geometry GEOMETRY(LINESTRING, 4326),
    start_point GEOMETRY(POINT, 4326),
    distance_meters FLOAT,
    elevation_gain FLOAT,
    surface_type VARCHAR(50),

    -- Shape signature for fast filtering
    perimeter FLOAT,
    area FLOAT,
    compactness FLOAT,
    aspect_ratio FLOAT,
    complexity_score FLOAT,
    turning_angles FLOAT[],
    fourier_descriptors FLOAT[],

    -- Metadata
    usage_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Spatial indices for fast queries
CREATE INDEX idx_routes_geom ON routes USING GIST(geometry);
CREATE INDEX idx_routes_start ON routes USING GIST(start_point);
CREATE INDEX idx_routes_distance ON routes(distance_meters);

-- GiST index for array similarity (requires pg_trgm or similar)
CREATE INDEX idx_routes_angles ON routes USING GIN(turning_angles);
```

---

## 7. Performance Optimizations

### Caching Strategy
```python
# Cache popular queries
@cache(ttl=3600)  # 1 hour
def get_routes_near(lat, lng, radius=5):
    return db.query_spatial(lat, lng, radius)

# Cache shape normalizations
@lru_cache(maxsize=1000)
def normalize_and_extract(shape_hash):
    return normalize_shape(shape), extract_features(shape)
```

### Approximations for Speed
```python
# Use simplified Fréchet for initial pass
def fast_frechet_filter(shape, routes, k=50):
    """
    Use approximate Fréchet to get top-k candidates quickly
    """
    # Sample only every Nth point
    shape_sampled = shape[::5]

    scores = []
    for route in routes:
        route_sampled = route[::5]
        dist = approximate_frechet(shape_sampled, route_sampled)
        scores.append((route, dist))

    # Return top-k for full Fréchet calculation
    scores.sort(key=lambda x: x[1])
    return [route for route, _ in scores[:k]]
```

### Parallel Processing
```python
from multiprocessing import Pool

def score_route(args):
    shape, route = args
    return compute_similarity_score(shape, route)

def score_all_routes_parallel(shape, routes):
    with Pool(processes=8) as pool:
        scores = pool.map(score_route, [(shape, r) for r in routes])
    return scores
```

---

## 8. Alternative: Machine Learning Approach

### Option: Siamese Neural Network

**Architecture:**
```
Input: Two route images (rasterized)
    ↓
Shared CNN Encoder
    ↓
Feature Vectors (embedding space)
    ↓
Distance Metric (L2 or cosine)
    ↓
Similarity Score
```

**Pros:**
- Learns what humans consider "similar"
- Can capture complex patterns
- Fast inference after training

**Cons:**
- Needs labeled training data
- Less interpretable
- Requires significant compute for training
- Harder to debug

**Training Data:**
- Collect pairs of (drawn_shape, selected_route) from users
- Label as similar/dissimilar
- Or use triplet loss: (anchor, positive, negative)

**Use Case:** If you have lots of user interaction data and want to optimize for human perception of similarity

---

## 9. Recommended Implementation Path

### Phase 1: MVP (Week 1-2)
- Use Overpass API for data
- Simple Hausdorff distance for matching
- Pre-generate ~100 popular routes per city
- Basic spatial filtering

**Pros:** Fast to build, validates concept
**Cons:** Limited routes, slow queries

### Phase 2: Production (Week 3-6)
- Set up PostgreSQL + PostGIS
- Import OSM extract for target regions
- Implement Fréchet distance matching
- Add shape feature filtering
- Pre-compute routes in popular areas

**Pros:** Fast, scalable, good results
**Cons:** Infrastructure complexity

### Phase 3: Optimization (Week 7+)
- Add turning angle analysis
- Implement route generation algorithm
- Add caching layer (Redis)
- Parallel score computation
- ML-based ranking (optional)

**Pros:** Best quality matches, handles any location
**Cons:** Complex, resource intensive

---

## 10. Trade-off Summary

| Approach | Speed | Accuracy | Storage | Complexity | Scalability |
|----------|-------|----------|---------|------------|-------------|
| Overpass + Hausdorff | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ |
| Offline + Fréchet | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Pre-computed DB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Dynamic Generation | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| ML Approach | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 11. Final Recommendation

**For a hackathon/MVP:**
Use **Offline OSM + Multi-stage Pipeline + Fréchet Distance**

**Why:**
1. Fast enough for real-time queries (<1 second)
2. Good match quality
3. Reasonable complexity
4. Scalable to production
5. No ML training needed
6. Debuggable and interpretable

**Architecture:**
```
1. Import OSM extract for hack region (UK: ~20GB)
2. Pre-generate 1000 routes around popular locations
3. Index with shape features
4. Use spatial + feature filtering → Fréchet on top 50
5. Return top 10 matches
```

**Expected Performance:**
- Query time: 200-800ms
- Match quality: 75-85% user satisfaction
- Storage: 30GB (data + index)
- Can handle 100+ concurrent users

**Next Steps:**
1. Set up PostGIS database
2. Implement shape normalization
3. Build route pre-generation script
4. Implement Fréchet distance
5. Create API endpoint with filtering pipeline
