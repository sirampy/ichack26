"""
Image to line conversion for route generation.
Uses contour-based approach for cleaner, more accurate lines.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple


def simplify_points(points: List[Tuple[int, int]], epsilon: float = 2.0) -> List[Tuple[int, int]]:
    """
    Simplify a path using Douglas-Peucker algorithm.

    Args:
        points: List of (x, y) tuples
        epsilon: Simplification tolerance (higher = more aggressive)

    Returns:
        Simplified list of points
    """
    if len(points) < 3:
        return points

    # Convert to numpy array format for cv2.approxPolyDP
    points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    # Apply Douglas-Peucker algorithm
    simplified = cv2.approxPolyDP(points_array, epsilon, closed=False)

    # Convert back to list of tuples
    return [(int(pt[0][0]), int(pt[0][1])) for pt in simplified]


def extract_contours_from_image(image_bytes: bytes) -> Tuple[List[np.ndarray], Tuple]:
    """
    Extract contours from image using edge detection.

    Args:
        image_bytes: Image data as bytes

    Returns:
        List of contours (each is array of points), image shape
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply strong Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply adaptive thresholding for better edge detection
    # This works better than Canny for noisy images
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, img.shape


def connect_contours_into_line(contours: List[np.ndarray], target_points: int = 500) -> List[Tuple[int, int]]:
    """
    Connect multiple contours into a single continuous line.
    Uses greedy nearest neighbor to connect contour endpoints.

    Args:
        contours: List of contours from cv2.findContours
        target_points: Target number of points in final line

    Returns:
        List of (x, y) tuples forming continuous line
    """
    if not contours:
        return []

    # Sort contours by length (longest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Take top contours that contribute to target points
    selected_contours = []
    total_points = 0

    for contour in contours:
        if total_points >= target_points:
            break
        selected_contours.append(contour)
        total_points += len(contour)

    if not selected_contours:
        return []

    # Convert contours to list of point sequences
    sequences = []
    for contour in selected_contours:
        # Flatten contour array and convert to list of tuples
        points = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
        sequences.append(points)

    # Connect sequences using nearest neighbor
    connected = sequences[0].copy()
    remaining = sequences[1:]

    while remaining:
        last_point = connected[-1]

        # Find closest sequence
        min_dist = float('inf')
        closest_idx = 0
        connect_to_start = True

        for i, seq in enumerate(remaining):
            # Distance to start of sequence
            dist_start = (seq[0][0] - last_point[0])**2 + (seq[0][1] - last_point[1])**2
            if dist_start < min_dist:
                min_dist = dist_start
                closest_idx = i
                connect_to_start = True

            # Distance to end of sequence (connect in reverse)
            dist_end = (seq[-1][0] - last_point[0])**2 + (seq[-1][1] - last_point[1])**2
            if dist_end < min_dist:
                min_dist = dist_end
                closest_idx = i
                connect_to_start = False

        # Add closest sequence
        next_seq = remaining.pop(closest_idx)
        if not connect_to_start:
            next_seq = next_seq[::-1]  # Reverse if connecting to end

        connected.extend(next_seq)

    # Simplify to target number of points
    if len(connected) > target_points:
        # Sample evenly
        indices = np.linspace(0, len(connected) - 1, target_points, dtype=int)
        connected = [connected[i] for i in indices]

    return connected


def image_to_line(image_bytes: bytes, num_points: int = 500) -> Tuple[List[Dict], Tuple]:
    """
    Convert image to a continuous line of points using contour extraction.

    Args:
        image_bytes: Image data as bytes
        num_points: Target number of points for the line

    Returns:
        List of {'x': x, 'y': y} dicts, image shape
    """
    print(f"  Extracting contours from image...")

    # Extract contours
    contours, img_shape = extract_contours_from_image(image_bytes)
    print(f"  Found {len(contours)} contours")

    if not contours:
        raise ValueError("No contours found in image. Try a different image with clearer edges.")

    # Connect contours into a single line
    print(f"  Connecting contours into continuous line...")
    points = connect_contours_into_line(contours, num_points)
    print(f"  Generated {len(points)} points")

    # Simplify the line to reduce noise (minimal simplification)
    print(f"  Simplifying line...")
    simplified = simplify_points(points, epsilon=0.75)
    print(f"  Simplified to {len(simplified)} points")

    # Convert to dict format
    result = [{'x': x, 'y': y} for x, y in simplified]

    return result, img_shape
