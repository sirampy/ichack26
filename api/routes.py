from flask import jsonify, request, Response
from . import api_bp
from .db import RouteDB
from .route_generator import RouteGenerator, transform_user_shape_to_geo, generate_reference_circle
from .image_processing import image_to_line
from .gpx_generator import generate_gpx
import os
import requests as http_requests
import uuid
from datetime import datetime

# Initialize database connection
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'routes.db')
db = RouteDB(DB_PATH)
route_generator = RouteGenerator(DB_PATH)

# OSRM API configuration
OSRM_BASE_URL = os.getenv('OSRM_BASE_URL', 'http://localhost:5050')

# In-memory route storage
PUBLISHED_ROUTES = {}


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

        # Use all waypoints (no downsampling)
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


@api_bp.route('/image-to-line', methods=['POST'])
def image_to_line_endpoint():
    """
    Convert uploaded image to line points.

    Expected: multipart/form-data with 'image' file
    Optional: 'num_points' (default 500)

    Returns: {'points': [{'x': x, 'y': y}, ...], 'image_size': [width, height]}
    """
    try:
        # Check if image file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get num_points parameter
        num_points = int(request.form.get('num_points', 500))
        num_points = max(100, min(2000, num_points))  # Clamp between 100-2000

        print(f"\n{'='*60}")
        print(f"IMAGE TO LINE CONVERSION")
        print(f"Filename: {file.filename}")
        print(f"Points to extract: {num_points}")
        print(f"{'='*60}\n")

        # Read image bytes
        image_bytes = file.read()

        # Convert image to line
        print(f"Processing image...")
        points, img_shape = image_to_line(image_bytes, num_points)

        print(f"✓ Generated {len(points)} points from image")
        print(f"  Image size: {img_shape[1]}x{img_shape[0]}")

        return jsonify({
            'points': points,
            'image_size': [img_shape[1], img_shape[0]],  # [width, height]
            'num_points': len(points)
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/export-gpx', methods=['POST'])
def export_gpx():
    """
    Generate and download GPX file from route data.

    Expected request body:
    {
        "route": {...},  # Route object with coordinates
        "location": {"lat": 51.5074, "lng": -0.1278}
    }

    Returns: GPX file download
    """
    try:
        data = request.get_json()

        if not data or 'route' not in data:
            return jsonify({'error': 'No route data provided'}), 400

        route = data['route']
        location = data.get('location', {'lat': 0, 'lng': 0})

        print(f"\n{'='*60}")
        print(f"GPX EXPORT REQUEST")
        print(f"Route: {route.get('name', 'Unknown')}")
        print(f"Distance: {route.get('distance', 0)} miles")
        print(f"Points: {len(route.get('coordinates', []))}")
        print(f"{'='*60}\n")

        # Generate GPX
        gpx_content = generate_gpx(route, location)

        # Create filename
        route_name = route.get('name', 'route').replace(' ', '_').lower()
        distance = route.get('distance', 0)
        filename = f"{route_name}_{distance}mi.gpx"

        print(f"✓ Generated GPX file: {filename}")

        # Return as downloadable file
        return Response(
            gpx_content,
            mimetype='application/gpx+xml',
            headers={
                'Content-Disposition': f'attachment; filename={filename}'
            }
        )

    except Exception as e:
        print(f"Error generating GPX: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/publish-route', methods=['POST'])
def publish_route():
    """
    Publish a route to the community.

    Expected request body:
    {
        "name": "Route Name",
        "distance": 3.2,
        "duration": 32,
        "coordinates": [...],
        "location": {"lat": 51.5074, "lng": -0.1278},
        "elevation_gain": 45
    }

    Returns: {"id": "route_id", "url": "/route/route_id"}
    """
    try:
        data = request.get_json()

        if not data or 'coordinates' not in data:
            return jsonify({'error': 'Missing route data'}), 400

        # Generate unique ID
        route_id = str(uuid.uuid4())[:8]

        # Store route
        PUBLISHED_ROUTES[route_id] = {
            'id': route_id,
            'name': data.get('name', 'Unnamed Route'),
            'distance': data.get('distance', 0),
            'duration': data.get('duration', 0),
            'coordinates': data['coordinates'],
            'location': data.get('location', {}),
            'elevation_gain': data.get('elevation_gain', 0),
            'created_at': datetime.utcnow().isoformat(),
            'author': 'anonymous'
        }

        print(f"✓ Published route: {route_id} - {PUBLISHED_ROUTES[route_id]['name']}")

        return jsonify({
            'id': route_id,
            'url': f'/route/{route_id}'
        })

    except Exception as e:
        print(f"Error publishing route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/routes', methods=['GET'])
def get_routes():
    """
    Get all published routes.

    Returns: {"routes": [...], "count": N}
    """
    try:
        routes = list(PUBLISHED_ROUTES.values())
        # Sort by created_at, newest first
        routes.sort(key=lambda r: r.get('created_at', ''), reverse=True)

        return jsonify({
            'routes': routes,
            'count': len(routes)
        })

    except Exception as e:
        print(f"Error getting routes: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/routes/<route_id>', methods=['GET'])
def get_route(route_id):
    """
    Get a specific published route by ID.

    Returns: route object or 404
    """
    try:
        route = PUBLISHED_ROUTES.get(route_id)

        if not route:
            return jsonify({'error': 'Route not found'}), 404

        return jsonify(route)

    except Exception as e:
        print(f"Error getting route: {e}")
        return jsonify({'error': str(e)}), 500
