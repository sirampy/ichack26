#!/usr/bin/env python3
"""Test route generation"""

from api.route_generator import RouteGenerator

generator = RouteGenerator('data/routes.db')

# Test generating loops of different sizes
test_cases = [
    (51.5074, -0.1278, 1000, "1km loop near Trafalgar Square"),
    (51.5074, -0.1278, 2000, "2km loop near Trafalgar Square"),
    (51.5074, -0.1278, 3000, "3km loop near Trafalgar Square"),
]

for lat, lng, distance_m, description in test_cases:
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"{'='*60}")

    route = generator.generate_loop(lat, lng, distance_m)

    if route:
        distance_km = route['distance_meters'] / 1000
        distance_mi = route['distance_meters'] / 1609.34

        print(f"\n✅ SUCCESS!")
        print(f"  Generated distance: {distance_km:.2f}km ({distance_mi:.2f} miles)")
        print(f"  Number of segments: {route['num_segments']}")
        print(f"  Coordinate points: {len(route['coordinates'])}")
        print(f"  First point: ({route['coordinates'][0]['lat']:.5f}, {route['coordinates'][0]['lng']:.5f})")
        print(f"  Last point: ({route['coordinates'][-1]['lat']:.5f}, {route['coordinates'][-1]['lng']:.5f})")

        # Check if it's actually a loop (start ≈ end)
        start = route['coordinates'][0]
        end = route['coordinates'][-1]
        lat_diff = abs(start['lat'] - end['lat'])
        lng_diff = abs(start['lng'] - end['lng'])

        if lat_diff < 0.001 and lng_diff < 0.001:
            print(f"  ✓ Forms a proper loop (start ≈ end)")
        else:
            print(f"  ⚠ Not a perfect loop (distance: ~{(lat_diff + lng_diff) * 111:.0f}m apart)")
    else:
        print(f"\n❌ FAILED to generate route")
