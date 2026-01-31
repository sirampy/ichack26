#!/usr/bin/env python3
"""Test distance filtering"""

import requests
import json

# Test different distances
test_cases = [
    {"distance": 1.0, "location": {"lat": 51.5074, "lng": -0.1278}},
    {"distance": 3.0, "location": {"lat": 51.5074, "lng": -0.1278}},
    {"distance": 5.0, "location": {"lat": 51.5074, "lng": -0.1278}},
    {"distance": 10.0, "location": {"lat": 51.5074, "lng": -0.1278}},
]

shape = [
    {"x": 100, "y": 100},
    {"x": 200, "y": 100},
    {"x": 200, "y": 200},
    {"x": 100, "y": 200}
]

print("Testing distance-based filtering...\n")

for test in test_cases:
    payload = {
        "location": test["location"],
        "shape": shape,
        "desired_distance_miles": test["distance"]
    }

    response = requests.post('http://localhost:5000/api/match-routes', json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f"Target: {test['distance']} miles")
        print(f"  Found: {data['count']} routes")
        print(f"  Range: {data['search_params']['min_distance_miles']}-{data['search_params']['max_distance_miles']} miles")

        if data['routes']:
            distances = [r['distance'] for r in data['routes']]
            print(f"  Actual distances: {min(distances):.1f} - {max(distances):.1f} miles")
        print()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        print()
