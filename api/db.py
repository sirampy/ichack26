"""
Simple database wrapper for querying OSM routes.
POC implementation - just basic spatial queries.
"""

import sqlite3
import os
from typing import List, Dict, Tuple


class RouteDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at {db_path}")

    def get_connection(self):
        """Get a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        try:
            conn.load_extension("mod_spatialite")
        except:
            pass  # SpatiaLite not available, will use basic queries
        return conn

    def find_routes_near(
        self,
        lat: float,
        lng: float,
        radius_km: float = 5.0,
        min_length_m: float = 500,
        max_length_m: float = 10000,
        limit: int = 20
    ) -> List[Dict]:
        """
        Find routes near a location.
        POC: Simple bounding box query without full spatial index.
        """
        conn = self.get_connection()
        cur = conn.cursor()

        # Calculate rough bounding box (1 degree ≈ 111km)
        lat_offset = radius_km / 111.32
        lng_offset = radius_km / (111.32 * abs(lat) / 90)  # Rough cos(lat) adjustment

        min_lat = lat - lat_offset
        max_lat = lat + lat_offset
        min_lng = lng - lng_offset
        max_lng = lng + lng_offset

        # Query ways within bounding box
        query = """
            SELECT
                osm_id,
                name,
                highway,
                surface,
                length_meters,
                geometry_wkt,
                min_lat,
                max_lat,
                min_lon,
                max_lon,
                -- Calculate distance from user location to center of route bbox
                ((min_lat + max_lat) / 2.0 - ?) * ((min_lat + max_lat) / 2.0 - ?) +
                ((min_lon + max_lon) / 2.0 - ?) * ((min_lon + max_lon) / 2.0 - ?) AS dist_sq
            FROM ways
            WHERE
                min_lat <= ? AND max_lat >= ?
                AND min_lon <= ? AND max_lon >= ?
                AND length_meters BETWEEN ? AND ?
                AND highway IN ('footway', 'path', 'cycleway', 'residential', 'tertiary', 'unclassified', 'service')
            ORDER BY
                dist_sq ASC
            LIMIT ?
        """

        cur.execute(query, (
            # Distance calculation parameters (lat, lat, lng, lng)
            lat, lat, lng, lng,
            # Bounding box parameters
            max_lat, min_lat,
            max_lng, min_lng,
            # Length range
            min_length_m, max_length_m,
            # Limit
            limit
        ))

        routes = []
        for row in cur.fetchall():
            routes.append({
                'osm_id': row[0],
                'name': row[1] or f"{row[2].title()} Route",
                'highway': row[2],
                'surface': row[3],
                'length_meters': row[4],
                'geometry_wkt': row[5],
                'bbox': {
                    'min_lat': row[6],
                    'max_lat': row[7],
                    'min_lon': row[8],
                    'max_lon': row[9]
                },
                'distance_sq': row[10]  # For debugging
            })

        conn.close()
        return routes

    def parse_wkt_linestring(self, wkt: str) -> List[Tuple[float, float]]:
        """Parse WKT LINESTRING to list of (lat, lng) tuples"""
        # Remove "LINESTRING(" and ")"
        coords_str = wkt.replace('LINESTRING(', '').replace(')', '')
        coords = []
        for pair in coords_str.split(','):
            lng, lat = map(float, pair.strip().split())
            coords.append((lat, lng))
        return coords

    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = self.get_connection()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM ways")
        total = cur.fetchone()[0]

        cur.execute("SELECT highway, COUNT(*) FROM ways GROUP BY highway ORDER BY COUNT(*) DESC LIMIT 5")
        top_types = cur.fetchall()

        conn.close()

        return {
            'total_routes': total,
            'top_types': [{'type': t[0], 'count': t[1]} for t in top_types]
        }
