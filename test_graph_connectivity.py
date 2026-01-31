#!/usr/bin/env python3
"""Test if the routing graph is connected"""

from api.route_generator import RouteGenerator
import random

generator = RouteGenerator('data/routes.db')
generator.graph.build_graph(51.5074, -0.1278, 2.0)

# Pick 10 random node pairs and see if we can find paths
nodes = list(generator.graph.nodes.keys())
successes = 0
failures = 0

for i in range(10):
    start = random.choice(nodes)
    end = random.choice(nodes)

    if start == end:
        continue

    path = generator._shortest_path(start, end, set())

    if path:
        successes += 1
        print(f"✓ Found path from {start} to {end}: {path['distance']:.0f}m")
    else:
        failures += 1
        print(f"✗ No path from {start} to {end}")

print(f"\nResult: {successes} successes, {failures} failures")

if failures > 5:
    print("\n⚠ Graph appears to be heavily fragmented/disconnected!")
    print("This is why route generation fails - can't find paths back.")
