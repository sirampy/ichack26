#!/usr/bin/env python3
"""Test with relaxed parameters"""

from api.route_generator import RouteGenerator

generator = RouteGenerator('data/routes.db')

# Try with very relaxed tolerance
route = generator.generate_loop(
    start_lat=51.5074,
    start_lng=-0.1278,
    target_distance_m=1500,
    tolerance=0.5  # Accept ±50%!
)

if route:
    print("✅ SUCCESS!")
    print(f"Distance: {route['distance_meters']:.0f}m ({route['distance_meters']/1609.34:.2f} mi)")
    print(f"Segments: {route['num_segments']}")
    print(f"Coordinates: {len(route['coordinates'])}")
else:
    print("❌ Still failed even with 50% tolerance")
