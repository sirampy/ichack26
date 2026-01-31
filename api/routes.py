from flask import jsonify, request
from . import api_bp
from .db import RouteDB
from .route_generator import RouteGenerator
import os
import threading

# Initialize database connection
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'routes.db')
db = RouteDB(DB_PATH)
route_generator = RouteGenerator(DB_PATH)


@api_bp.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})


@api_bp.route('/db-stats')
def db_stats():
    """Get database statistics"""
    try:
        stats = db.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/match-routes', methods=['POST'])
def match_routes():
    """
    Match routes based on drawn shape and location.
    POC: Returns real routes from database near the location.

    Expected request body:
    {
        "location": {"lat": 51.5074, "lng": -0.1278},
        "shape": [{"x": 100, "y": 150}, {"x": 120, "y": 180}, ...]
    }

    Returns list of matching routes with coordinates.
    """
    data = request.get_json()

    if not data or 'location' not in data or 'shape' not in data:
        return jsonify({'error': 'Missing location or shape data'}), 400

    location = data['location']
    shape = data['shape']
    desired_distance_miles = data.get('desired_distance_miles', 3.0)

    try:
        # Convert desired distance to meters
        target_distance_m = desired_distance_miles * 1609.34

        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        print(f"\n{'='*60}", flush=True)
        print(f"ROUTE MATCHING REQUEST", flush=True)
        print(f"Location: ({location['lat']:.4f}, {location['lng']:.4f})", flush=True)
        print(f"Desired distance: {desired_distance_miles:.1f} miles ({target_distance_m:.0f}m)", flush=True)
        print(f"{'='*60}\n", flush=True)

        # Generate reference shape for matching
        from .route_generator import generate_reference_circle, transform_user_shape_to_geo

        if shape and len(shape) >= 3:
            # User drew a shape - use it!
            print(f"Using user-drawn shape with {len(shape)} points")
            try:
                reference_shape = transform_user_shape_to_geo(
                    shape, location['lat'], location['lng'], target_distance_m
                )
            except Exception as e:
                print(f"Failed to transform user shape: {e}")
                print("Falling back to circular shape")
                radius_m = target_distance_m / (2 * 3.14159 * 1.7)
                reference_shape = generate_reference_circle(
                    location['lat'], location['lng'], radius_m, num_points=36
                )
        else:
            # No shape or insufficient points - use circular shape
            print("No user shape provided, using circular shape")
            radius_m = target_distance_m / (2 * 3.14159 * 1.7)
            reference_shape = generate_reference_circle(
                location['lat'], location['lng'], radius_m, num_points=36
            )
            print(f"Generated reference circular shape with radius {radius_m:.0f}m")

        # Generate routes using shape matching
        print("Attempting to generate shape-matched routes...")
        generated_routes = []

        # Try to generate 2 routes with slightly different parameters
        for attempt in range(2):
            try:
                # Vary distance slightly for each attempt to get different routes
                distance_variation = target_distance_m * (1.0 + (attempt - 0.5) * 0.1)

                route = route_generator.generate_shape_matched_loop(
                    start_lat=location['lat'],
                    start_lng=location['lng'],
                    target_shape=reference_shape,
                    target_distance_m=distance_variation,
                    tolerance=0.7
                )

                if route:
                    route['attempt'] = attempt
                    generated_routes.append(route)
                    print(f"  ✓ Generated route {len(generated_routes)}: {route['distance_meters']:.0f}m")

                    if len(generated_routes) >= 2:
                        break
            except Exception as e:
                print(f"  ✗ Route generation attempt {attempt + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        db_routes = []

        # Format routes for frontend
        formatted_routes = []

        # Format generated routes
        for i, route in enumerate(generated_routes):
            formatted_routes.append({
                'id': f"generated_{i}",
                'name': f"Generated Loop {i+1}",
                'distance': round(route['distance_meters'] / 1609.34, 2),
                'duration': int(route['distance_meters'] / 1609.34 * 10),
                'match_score': 90 - (i * 5),  # TODO: Real shape matching
                'elevation_gain': 0,
                'coordinates': route['coordinates'],
                'type': 'generated'
            })

        # Format database segment routes (if any)
        for i, route in enumerate(db_routes):
            # Parse WKT geometry
            coords = db.parse_wkt_linestring(route['geometry_wkt'])

            # Convert to frontend format
            formatted_routes.append({
                'id': f"osm_{route['osm_id']}",
                'name': route['name'],
                'distance': round(route['length_meters'] / 1609.34, 2),  # meters to miles
                'duration': int(route['length_meters'] / 1609.34 * 10),  # ~10 min/mile
                'match_score': 85 - (i * 3),  # POC: Fake scores, descending
                'elevation_gain': 0,
                'coordinates': [
                    {'lat': lat, 'lng': lng}
                    for lat, lng in coords
                ],
                'type': 'segment'
            })

        if not formatted_routes:
            return jsonify({
                'routes': [],
                'count': 0,
                'message': 'No routes found near this location'
            })

        return jsonify({
            'routes': formatted_routes,
            'count': len(formatted_routes),
            'source': 'generated' if generated_routes else 'database',
            'search_params': {
                'target_distance_miles': desired_distance_miles
            }
        })

    except Exception as e:
        print(f"Error querying routes: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def estimate_shape_length(shape):
    """Estimate length of drawn shape in pixels"""
    if len(shape) < 2:
        return 0

    total = 0
    for i in range(len(shape) - 1):
        dx = shape[i+1]['x'] - shape[i]['x']
        dy = shape[i+1]['y'] - shape[i]['y']
        total += (dx**2 + dy**2)**0.5

    return total
