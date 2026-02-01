"""
Convert an image to a single continuous line drawing.
Uses edge detection + TSP optimization for path planning.
"""

import cv2
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


def image_to_edge_points(image_path, num_points=500, edge_threshold1=50, edge_threshold2=150):
    """
    Extract edge points from an image.

    Args:
        image_path: Path to input image
        num_points: Target number of points to extract
        edge_threshold1, edge_threshold2: Canny edge detection thresholds

    Returns:
        List of (x, y) edge points
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blurred, edge_threshold1, edge_threshold2)

    # Get edge pixel coordinates
    edge_points = np.column_stack(np.where(edges > 0))

    # Sample points if we have too many
    if len(edge_points) > num_points:
        indices = np.random.choice(len(edge_points), num_points, replace=False)
        edge_points = edge_points[indices]

    # Convert from (row, col) to (x, y)
    points = [(int(y), int(x)) for x, y in edge_points]

    return points, img.shape


def connect_points_greedy_nearest_neighbor(points):
    """
    Connect points using greedy nearest neighbor algorithm.
    Fast approximation for TSP - not optimal but very fast.

    Args:
        points: List of (x, y) tuples

    Returns:
        List of points in connected order
    """
    if not points:
        return []

    unvisited = set(range(len(points)))
    path = [0]  # Start at first point
    unvisited.remove(0)

    while unvisited:
        current = path[-1]
        current_point = points[current]

        # Find nearest unvisited point
        nearest = min(unvisited,
                     key=lambda i: (points[i][0] - current_point[0])**2 +
                                   (points[i][1] - current_point[1])**2)

        path.append(nearest)
        unvisited.remove(nearest)

    return [points[i] for i in path]


def optimize_path_2opt(points, max_iterations=100):
    """
    Improve path using 2-opt optimization.
    Swaps edges to reduce total path length.

    Args:
        points: Ordered list of (x, y) tuples
        max_iterations: Maximum number of improvement iterations

    Returns:
        Optimized list of points
    """
    def path_length(pts):
        return sum(((pts[i+1][0] - pts[i][0])**2 + (pts[i+1][1] - pts[i][1])**2)**0.5
                   for i in range(len(pts) - 1))

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(1, len(points) - 2):
            for j in range(i + 1, len(points)):
                if j - i == 1:
                    continue

                # Calculate current distance
                current = (((points[i][0] - points[i-1][0])**2 + (points[i][1] - points[i-1][1])**2)**0.5 +
                          ((points[j][0] - points[j-1][0])**2 + (points[j][1] - points[j-1][1])**2)**0.5)

                # Calculate distance after swap
                new = (((points[j][0] - points[i-1][0])**2 + (points[j][1] - points[i-1][1])**2)**0.5 +
                      ((points[i][0] - points[j-1][0])**2 + (points[i][1] - points[j-1][1])**2)**0.5)

                if new < current:
                    # Reverse the segment between i and j
                    points[i:j] = reversed(points[i:j])
                    improved = True

        if iteration % 10 == 0:
            print(f"  2-opt iteration {iteration}, path length: {path_length(points):.0f}")

    return points


def image_to_continuous_line(image_path, num_points=500, optimize=True):
    """
    Convert image to a single continuous line.

    Args:
        image_path: Path to input image
        num_points: Number of points to extract from edges
        optimize: Whether to optimize the path with 2-opt

    Returns:
        List of (x, y) points forming continuous line
    """
    print(f"Processing {image_path}...")

    # Extract edge points
    print(f"Extracting {num_points} edge points...")
    points, img_shape = image_to_edge_points(image_path, num_points)
    print(f"  Found {len(points)} edge points")

    # Connect points using nearest neighbor
    print("Connecting points with nearest neighbor algorithm...")
    connected_points = connect_points_greedy_nearest_neighbor(points)

    # Optimize path
    if optimize:
        print("Optimizing path with 2-opt...")
        connected_points = optimize_path_2opt(connected_points)

    print(f"✓ Generated continuous line with {len(connected_points)} points")

    return connected_points, img_shape


def save_line_visualization(points, img_shape, output_path):
    """Save visualization of the continuous line."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw the line
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    ax.plot(xs, ys, 'k-', linewidth=0.5)
    ax.set_xlim(0, img_shape[1])
    ax.set_ylim(0, img_shape[0])
    ax.invert_yaxis()  # Flip y-axis to match image coordinates
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_to_line.py <image_path> [num_points]")
        print("Example: python image_to_line.py photo.jpg 1000")
        sys.exit(1)

    image_path = sys.argv[1]
    num_points = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    # Convert image to line
    points, img_shape = image_to_continuous_line(image_path, num_points, optimize=True)

    # Save visualization
    output_path = image_path.rsplit('.', 1)[0] + '_lineart.png'
    save_line_visualization(points, img_shape, output_path)

    # Save points as JSON for use in the app
    import json
    points_output = image_path.rsplit('.', 1)[0] + '_points.json'
    with open(points_output, 'w') as f:
        json.dump({'points': [{'x': x, 'y': y} for x, y in points]}, f)
    print(f"Saved points to {points_output}")
