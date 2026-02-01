"""
GPX file generation for route export.
"""

from datetime import datetime
from typing import List, Dict


def generate_gpx(route: Dict, location: Dict) -> str:
    """
    Generate a GPX file from route coordinates.

    Args:
        route: Route dict with 'coordinates', 'name', 'distance', etc.
        location: Starting location dict with 'lat', 'lng'

    Returns:
        GPX XML string
    """
    coordinates = route.get('coordinates', [])
    route_name = route.get('name', 'Running Route')
    distance_miles = route.get('distance', 0)

    # GPX header
    gpx = '''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1"
     creator="Route Matcher - https://github.com/yourusername/route-matcher"
     xmlns="http://www.topografix.com/GPX/1/1"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">
'''

    # Metadata
    now = datetime.utcnow().isoformat() + 'Z'
    gpx += f'''  <metadata>
    <name>{route_name}</name>
    <desc>Generated running route - {distance_miles} miles</desc>
    <time>{now}</time>
  </metadata>
'''

    # Track
    gpx += f'''  <trk>
    <name>{route_name}</name>
    <type>running</type>
    <trkseg>
'''

    # Add all trackpoints
    for coord in coordinates:
        lat = coord['lat']
        lng = coord['lng']
        gpx += f'      <trkpt lat="{lat}" lon="{lng}"></trkpt>\n'

    # Close track
    gpx += '''    </trkseg>
  </trk>
'''

    # Close GPX
    gpx += '</gpx>\n'

    return gpx


def generate_gpx_route(route: Dict, location: Dict) -> str:
    """
    Generate a GPX file with route points (for navigation).
    This creates waypoints instead of a track.

    Args:
        route: Route dict with 'coordinates', 'name', 'distance', etc.
        location: Starting location dict with 'lat', 'lng'

    Returns:
        GPX XML string
    """
    coordinates = route.get('coordinates', [])
    route_name = route.get('name', 'Running Route')
    distance_miles = route.get('distance', 0)

    # GPX header
    gpx = '''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1"
     creator="Route Matcher"
     xmlns="http://www.topografix.com/GPX/1/1"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">
'''

    # Metadata
    now = datetime.utcnow().isoformat() + 'Z'
    gpx += f'''  <metadata>
    <name>{route_name}</name>
    <desc>Running route - {distance_miles} miles</desc>
    <time>{now}</time>
  </metadata>
'''

    # Route
    gpx += f'''  <rte>
    <name>{route_name}</name>
'''

    # Sample route points (don't need all of them, just key waypoints)
    # Take every Nth point to keep file size reasonable
    num_waypoints = min(100, len(coordinates))
    if num_waypoints > 0:
        step = max(1, len(coordinates) // num_waypoints)
        waypoints = coordinates[::step]

        for i, coord in enumerate(waypoints):
            lat = coord['lat']
            lng = coord['lng']
            gpx += f'    <rtept lat="{lat}" lon="{lng}">\n'
            gpx += f'      <name>WP{i+1}</name>\n'
            gpx += f'    </rtept>\n'

    # Close route
    gpx += '''  </rte>
'''

    # Close GPX
    gpx += '</gpx>\n'

    return gpx
