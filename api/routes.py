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

# GraphHopper API configuration
GRAPHHOPPER_KEY = '2b7e595c-0bab-4b84-93ba-7ace57677174'  # Set this in your environment


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
    Match routes based on drawn shape and location using GraphHopper Map Matching API.

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
        print(f"ROUTE MATCHING REQUEST (GraphHopper API)")
        print(f"Location: ({location['lat']:.4f}, {location['lng']:.4f})")
        print(f"Desired distance: {desired_distance_miles:.1f} miles ({target_distance_m:.0f}m)")
        print(f"{'='*60}\n")

        # Check if GraphHopper key is configured
        if not GRAPHHOPPER_KEY:
            return jsonify({
                'error': 'GraphHopper API key not configured. Set GRAPHHOPPER_KEY environment variable.'
            }), 500

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

        # Convert to GraphHopper Routing API format: [longitude, latitude]
        # Note: GraphHopper POST uses [lng, lat] order, not [lat, lng]
        # Free tier allows max 5 waypoints, so sample if we have more
        max_waypoints = 5
        if len(reference_shape) > max_waypoints:
            # Sample evenly spaced points from the reference shape
            indices = [int(i * (len(reference_shape) - 1) / (max_waypoints - 1)) for i in range(max_waypoints)]
            sampled_shape = [reference_shape[i] for i in indices]
            gps_points = [[lng, lat] for lat, lng in sampled_shape]
            print(f"Sampled {len(gps_points)} waypoints from {len(reference_shape)} points (free tier limit)")
        else:
            gps_points = [[lng, lat] for lat, lng in reference_shape]

        # Call GraphHopper Routing API (works with free tier, unlike Map Matching)
        url = "https://graphhopper.com/api/1/route"

        payload = {
            "points": gps_points,
            "profile": "foot",
            "locale": "en",
            "points_encoded": False,
            "instructions": False
        }

        params = {
            "key": GRAPHHOPPER_KEY
        }

        headers = {
            "Content-Type": "application/json"
        }

        print(f"Calling GraphHopper Routing API with {len(gps_points)} waypoints...")
        response = http_requests.post(url, json=payload, params=params, headers=headers, timeout=30)

        if response.status_code != 200:
            error_msg = f"GraphHopper API error: {response.status_code}"
            try:
                error_data = response.json()
                error_msg += f" - {error_data.get('message', 'Unknown error')}"
            except:
                error_msg += f" - {response.text[:200]}"
            print(f"ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 500

        gh_data = response.json()

        # Check if we got a valid response
        if 'paths' not in gh_data or len(gh_data['paths']) == 0:
            print("No paths returned from GraphHopper")
            return jsonify({
                'routes': [],
                'count': 0,
                'message': 'Could not match route to roads. Try a different location or shape.'
            })

        # Extract the matched route
        path = gh_data['paths'][0]
        distance_m = path['distance']
        duration_ms = path['time']

        # Get coordinates from the path
        if 'points' in path:
            coordinates = path['points']['coordinates']
            # GraphHopper returns [lng, lat] format
            route_coords = [
                {'lat': lat, 'lng': lng}
                for lng, lat in coordinates
            ]
        else:
            print("No coordinates in GraphHopper response")
            return jsonify({
                'routes': [],
                'count': 0,
                'message': 'Route matching failed. Please try again.'
            })

        print(f"✓ Successfully matched route: {distance_m:.0f}m, {len(route_coords)} points")

        formatted_route = {
            'id': 'graphhopper_matched',
            'name': 'Your Matched Route',
            'distance': round(distance_m / 1609.34, 2),  # meters to miles
            'duration': int(duration_ms / 60000),  # milliseconds to minutes
            'match_score': 95,
            'elevation_gain': 0,
            'coordinates': route_coords,
            'type': 'graphhopper'
        }

        return jsonify({
            'routes': [formatted_route],
            'count': 1,
            'source': 'graphhopper',
            'search_params': {
                'target_distance_miles': desired_distance_miles
            }
        })

    except http_requests.exceptions.Timeout:
        print("ERROR: GraphHopper API request timed out")
        return jsonify({'error': 'Request timed out. Please try again.'}), 504
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
