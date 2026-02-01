from flask import jsonify, request
from . import api_bp
from .db import RouteDB
from .route_generator import RouteGenerator, transform_user_shape_to_geo, generate_reference_circle
import os
import requests as http_requests

# Initialize database connection
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'routes.db')
db = RouteDB(DB_PATH)
route_generator = RouteGenerator(DB_PATH)

# OSRM API configuration
OSRM_BASE_URL = os.getenv('OSRM_BASE_URL', 'http://localhost:5050')


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
    Match routes based on drawn shape and location using OSRM Map Matching API.

    Expected request body:
    {
        "location": {"lat": 51.5074, "lng": -0.1278},
        "shape": [{"x": 100, "y": 150}, {"x": 120, "y": 180}, ...],
        "desired_distance_miles": 3.0
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

        print(f"\n{'='*60}")
        print(f"ROUTE GENERATION REQUEST (OSRM Route API)")
        print(f"Location: ({location['lat']:.4f}, {location['lng']:.4f})")
        print(f"Desired distance: {desired_distance_miles:.1f} miles ({target_distance_m:.0f}m)")
        print(f"{'='*60}\n")

        # Transform user's drawn shape to geographic coordinates
        if shape and len(shape) >= 3:
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
            print("No user shape provided, using circular shape")
            radius_m = target_distance_m / (2 * 3.14159 * 1.7)
            reference_shape = generate_reference_circle(
                location['lat'], location['lng'], radius_m, num_points=36
            )

        print(f"Generated reference shape with {len(reference_shape)} waypoints")

        # OSRM can handle many more waypoints than GraphHopper's free tier
        # Sample to reasonable number for performance (OSRM recommends < 100 for match)
        max_waypoints = 50
        if len(reference_shape) > max_waypoints:
            indices = [int(i * (len(reference_shape) - 1) / (max_waypoints - 1)) for i in range(max_waypoints)]
            sampled_shape = [reference_shape[i] for i in indices]
            print(f"Sampled {len(sampled_shape)} waypoints from {len(reference_shape)} points")
        else:
            sampled_shape = reference_shape

        # Convert to OSRM format: lng,lat;lng,lat;...
        # OSRM uses semicolon-separated coordinate pairs
        coordinates_str = ';'.join([f"{lng},{lat}" for lat, lng in sampled_shape])

        # Call OSRM Route API (not Match API)
        # Route API finds the best route through waypoints, preserving the shape better
        url = f"{OSRM_BASE_URL}/route/v1/foot/{coordinates_str}"

        params = {
            'overview': 'full',  # Get full route geometry
            'geometries': 'geojson',  # Use GeoJSON format for coordinates
            'steps': 'false',  # We don't need turn-by-turn instructions
            'continue_straight': 'false',  # Allow turns at waypoints
        }

        print(f"Calling OSRM Route API with {len(sampled_shape)} waypoints...")
        print(f"URL: {url}")

        response = http_requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            error_msg = f"OSRM API error: {response.status_code}"
            try:
                error_data = response.json()
                error_msg += f" - {error_data.get('message', 'Unknown error')}"
            except:
                error_msg += f" - {response.text[:200]}"
            print(f"ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 500

        osrm_data = response.json()

        # Check if we got a valid response
        if osrm_data.get('code') != 'Ok':
            error_msg = osrm_data.get('message', 'Route generation failed')
            print(f"OSRM returned error: {error_msg}")
            return jsonify({
                'routes': [],
                'count': 0,
                'message': f'Could not generate route: {error_msg}'
            })

        if 'routes' not in osrm_data or len(osrm_data['routes']) == 0:
            print("No routes returned from OSRM")
            return jsonify({
                'routes': [],
                'count': 0,
                'message': 'Could not generate route. Try a different location or shape.'
            })

        # Extract the route (first route)
        route = osrm_data['routes'][0]
        distance_m = route['distance']
        duration_s = route['duration']

        # Get coordinates from the geometry
        if 'geometry' in route and 'coordinates' in route['geometry']:
            coordinates = route['geometry']['coordinates']
            # OSRM GeoJSON format is [lng, lat]
            route_coords = [
                {'lat': lat, 'lng': lng}
                for lng, lat in coordinates
            ]
        else:
            print("No geometry in OSRM response")
            return jsonify({
                'routes': [],
                'count': 0,
                'message': 'Route generation failed. Please try again.'
            })

        print(f"✓ Successfully generated route: {distance_m:.0f}m, {len(route_coords)} points")

        formatted_route = {
            'id': 'osrm_route',
            'name': 'Your Generated Route',
            'distance': round(distance_m / 1609.34, 2),  # meters to miles
            'duration': int(duration_s / 60),  # seconds to minutes
            'match_score': 100,  # Route API always finds a valid route
            'elevation_gain': 0,
            'coordinates': route_coords,
            'type': 'osrm'
        }

        return jsonify({
            'routes': [formatted_route],
            'count': 1,
            'source': 'osrm',
            'search_params': {
                'target_distance_miles': desired_distance_miles
            }
        })

    except http_requests.exceptions.Timeout:
        print("ERROR: OSRM API request timed out")
        return jsonify({'error': 'Request timed out. Please try again.'}), 504
    except http_requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to OSRM server")
        return jsonify({'error': 'OSRM server is not running. Please start it with: docker-compose up -d'}), 503
    except Exception as e:
        print(f"Error matching route: {e}")
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
