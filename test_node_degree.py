#!/usr/bin/env python3
"""Check node degree distribution"""

from api.route_generator import RouteGenerator

generator = RouteGenerator('data/routes.db')
generator.graph.build_graph(51.5074, -0.1278, 2.0)

# Count node degrees
degree_counts = {}
for node_id, edges in generator.graph.edges.items():
    degree = len(edges)
    degree_counts[degree] = degree_counts.get(degree, 0) + 1

print("Node degree distribution:")
for degree in sorted(degree_counts.keys()):
    count = degree_counts[degree]
    print(f"  Degree {degree}: {count} nodes")

print(f"\nTotal nodes: {len(generator.graph.nodes)}")
print(f"Nodes with degree 0 (isolated): {degree_counts.get(0, 0)}")
print(f"Nodes with degree 1 (dead ends): {degree_counts.get(1, 0)}")
print(f"Nodes with degree 2+ (connected): {sum(c for d, c in degree_counts.items() if d >= 2)}")

# Find a well-connected node and show its connections
for node_id, edges in generator.graph.edges.items():
    if len(edges) >= 3:
        print(f"\nExample: Node {node_id} at {generator.graph.nodes[node_id]}")
        print(f"  Has {len(edges)} connections:")
        for next_node, way_id, dist in edges[:5]:
            print(f"    → Node {next_node} via way {way_id} ({dist:.0f}m)")
        break
