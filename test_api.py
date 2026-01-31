#!/usr/bin/env python3
"""Test the match-routes API endpoint"""

import requests
import json

# Test with a location in London and a simple shape
test_data = {
    "location": {
        "lat": 51.5074,
        "lng": -0.1278
    },
    "shape": [
        {"x": 100, "y": 100},
        {"x": 200, "y": 150},
        {"x": 250, "y": 200},
        {"x": 200, "y": 250},
        {"x": 100, "y": 200},
        {"x": 50, "y": 150}
    ],
    "desired_distance_miles": 2.0
}

print("Testing /api/match-routes endpoint...")
print(f"Location: {test_data['location']}")
print(f"Distance: {test_data['desired_distance_miles']} miles")
print(f"Shape points: {len(test_data['shape'])}")
print("\nSending request...")

try:
    response = requests.post(
        'http://localhost:5000/api/match-routes',
        json=test_data,
        timeout=30
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Success!")
        print(f"Routes found: {data.get('count', 0)}")
        print(f"Source: {data.get('source', 'unknown')}")

        if data.get('routes'):
            print(f"\nFirst route details:")
            route = data['routes'][0]
            print(f"  Name: {route.get('name')}")
            print(f"  Distance: {route.get('distance')} miles")
            print(f"  Duration: {route.get('duration')} min")
            print(f"  Match score: {route.get('match_score')}%")
            print(f"  Coordinates: {len(route.get('coordinates', []))} points")
        else:
            print("\n⚠ No routes returned")
            print(f"Message: {data.get('message')}")
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)

except requests.exceptions.Timeout:
    print("\n✗ Request timed out after 30 seconds")
except requests.exceptions.ConnectionError:
    print("\n✗ Could not connect to server. Is it running on http://localhost:5000?")
except Exception as e:
    print(f"\n✗ Error: {e}")
