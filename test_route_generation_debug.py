#!/usr/bin/env python3
"""Debug route generation"""

from api.route_generator import RouteGenerator
import random

random.seed(42)  # For reproducibility

generator = RouteGenerator('data/routes.db')

# Test just one case with detailed output
lat, lng = 51.5074, -0.1278
distance_m = 1500

print(f"Generating {distance_m}m loop from ({lat:.4f}, {lng:.4f})")
print()

# Build graph
generator.graph.build_graph(lat, lng, 2.0)

# Find start node
start_node = generator.graph.find_nearest_node(lat, lng)
print(f"Start node: {start_node}")
print(f"Start node position: {generator.graph.nodes[start_node]}")
print(f"Edges from start node: {len(generator.graph.edges[start_node])}")
print()

# Try one walk manually with debug output
current_node = start_node
visited_ways = []
total_distance = 0.0
explore_distance = distance_m * 0.6

print(f"Phase 1: Random walk to {explore_distance}m")
for step in range(20):
    available = [
        (next_node, way_id, dist)
        for next_node, way_id, dist in generator.graph.edges[current_node]
        if way_id not in visited_ways
    ]

    if not available:
        print(f"  Step {step}: DEAD END at node {current_node}")
        break

    next_node, way_id, dist = random.choice(available)
    visited_ways.append(way_id)
    total_distance += dist
    current_node = next_node

    print(f"  Step {step}: node {current_node}, +{dist:.0f}m (total: {total_distance:.0f}m)")

    if total_distance >= explore_distance:
        print(f"  Reached target explore distance")
        break

print(f"\nPhase 2: Find path back from node {current_node} to {start_node}")
return_path = generator._shortest_path(current_node, start_node, set())

if return_path:
    print(f"  ✓ Found return path: {return_path['distance']:.0f}m using {len(return_path['ways'])} ways")
    print(f"  Total loop distance: {total_distance + return_path['distance']:.0f}m")
else:
    print(f"  ✗ No return path found")
