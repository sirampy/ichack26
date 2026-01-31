#!/usr/bin/env python3
"""
Transform OSM PBF file into SQLite database with spatial support.
Extracts walkable/runnable ways from OpenStreetMap data.
"""

import sqlite3
import osmium
import sys
from pathlib import Path

class OSMWayHandler(osmium.SimpleHandler):
    def __init__(self, db_path):
        osmium.SimpleHandler.__init__(self)
        self.db_path = db_path
        self.setup_database()
        self.way_count = 0
        self.filtered_count = 0
        self.location_errors = 0

    def setup_database(self):
        """Initialize SQLite database with spatial support"""
        print("Setting up database...")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.enable_load_extension(True)

        try:
            # Try to load SpatiaLite extension
            self.conn.load_extension("mod_spatialite")
            print("✓ Loaded SpatiaLite extension")
            self.has_spatialite = True
        except sqlite3.OperationalError:
            try:
                self.conn.load_extension("mod_spatialite.so")
                print("✓ Loaded SpatiaLite extension (.so)")
                self.has_spatialite = True
            except sqlite3.OperationalError:
                print("⚠ Warning: SpatiaLite not available, using basic SQLite")
                self.has_spatialite = False

        cur = self.conn.cursor()

        if self.has_spatialite:
            # Initialize spatial metadata
            cur.execute("SELECT InitSpatialMetadata(1)")

        # Create ways table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ways (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                osm_id INTEGER UNIQUE,
                highway TEXT,
                surface TEXT,
                name TEXT,
                access TEXT,
                length_meters REAL,
                num_nodes INTEGER,
                min_lat REAL,
                max_lat REAL,
                min_lon REAL,
                max_lon REAL,
                geometry_wkt TEXT
            )
        """)

        if self.has_spatialite:
            # Add geometry column and spatial index
            try:
                cur.execute("""
                    SELECT AddGeometryColumn('ways', 'geometry', 4326, 'LINESTRING', 'XY')
                """)
            except:
                pass  # Column might already exist

            try:
                cur.execute("SELECT CreateSpatialIndex('ways', 'geometry')")
            except:
                pass  # Index might already exist

        # Create regular indices
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ways_highway ON ways(highway)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ways_bbox ON ways(min_lat, min_lon, max_lat, max_lon)")

        self.conn.commit()
        print("✓ Database initialized")

    def way(self, w):
        """Process each OSM way"""
        self.way_count += 1

        if self.way_count % 10000 == 0:
            print(f"  Processed {self.way_count} ways, kept {self.filtered_count}...")

        # Filter: only keep ways with highway tag
        if 'highway' not in w.tags:
            return

        highway_type = w.tags['highway']

        # Filter: only runnable/walkable ways
        allowed_types = {
            'footway', 'path', 'track', 'steps',
            'pedestrian', 'cycleway', 'bridleway',
            'residential', 'living_street', 'service',
            'tertiary', 'tertiary_link', 'unclassified',
            'secondary', 'secondary_link',
            'primary', 'primary_link'
        }

        if highway_type not in allowed_types:
            return

        # Filter: skip private/restricted access
        access = w.tags.get('access', '')
        if access in ['no', 'private']:
            return

        # Must have at least 2 nodes
        if len(w.nodes) < 2:
            return

        try:
            # Extract node coordinates
            # Important: osmium needs location index for coordinates
            nodes = []
            for node in w.nodes:
                try:
                    if node.location.valid():
                        nodes.append((node.lon, node.lat))
                except:
                    self.location_errors += 1
                    pass

            if len(nodes) < 2:
                if self.location_errors < 10:  # Only warn first few times
                    print(f"  Warning: Way {w.id} has insufficient valid nodes ({len(nodes)})")
                return

            # Calculate bounding box
            lons = [n[0] for n in nodes]
            lats = [n[1] for n in nodes]
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)

            # Calculate accurate length using proper geographic distance
            import math
            length_meters = 0
            for i in range(len(nodes) - 1):
                lon1, lat1 = nodes[i]
                lon2, lat2 = nodes[i + 1]

                # Proper distance calculation
                # 1 degree latitude = 111320 meters everywhere
                # 1 degree longitude = 111320 * cos(latitude) meters
                dlat_m = (lat2 - lat1) * 111320
                dlon_m = (lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
                length_meters += math.sqrt(dlat_m**2 + dlon_m**2)

            # Create WKT (Well-Known Text) geometry
            coords_str = ', '.join([f"{lon} {lat}" for lon, lat in nodes])
            geometry_wkt = f"LINESTRING({coords_str})"

            # Insert into database
            cur = self.conn.cursor()

            if self.has_spatialite:
                cur.execute("""
                    INSERT OR REPLACE INTO ways (
                        osm_id, highway, surface, name, access,
                        length_meters, num_nodes,
                        min_lat, max_lat, min_lon, max_lon,
                        geometry_wkt, geometry
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GeomFromText(?, 4326))
                """, (
                    w.id,
                    highway_type,
                    w.tags.get('surface', 'unknown'),
                    w.tags.get('name', ''),
                    access,
                    length_meters,
                    len(nodes),
                    min_lat, max_lat, min_lon, max_lon,
                    geometry_wkt,
                    geometry_wkt
                ))
            else:
                cur.execute("""
                    INSERT OR REPLACE INTO ways (
                        osm_id, highway, surface, name, access,
                        length_meters, num_nodes,
                        min_lat, max_lat, min_lon, max_lon,
                        geometry_wkt
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    w.id,
                    highway_type,
                    w.tags.get('surface', 'unknown'),
                    w.tags.get('name', ''),
                    access,
                    length_meters,
                    len(nodes),
                    min_lat, max_lat, min_lon, max_lon,
                    geometry_wkt
                ))

            self.filtered_count += 1

            # Commit every 1000 ways for performance
            if self.filtered_count % 1000 == 0:
                self.conn.commit()

        except Exception as e:
            print(f"Error processing way {w.id}: {e}")

    def finish(self):
        """Finalize and close database"""
        self.conn.commit()

        # Print statistics
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM ways")
        total = cur.fetchone()[0]

        cur.execute("SELECT highway, COUNT(*) FROM ways GROUP BY highway ORDER BY COUNT(*) DESC")
        stats = cur.fetchall()

        print(f"\n✓ Transformation complete!")
        print(f"  Total OSM ways processed: {self.way_count:,}")
        print(f"  Runnable ways extracted: {total:,}")
        print(f"\n  Breakdown by type:")
        for highway_type, count in stats[:10]:
            print(f"    {highway_type:20s}: {count:,}")

        self.conn.close()


def main():
    pbf_path = Path("/home/alex/Downloads/greater-london-260130.osm.pbf")
    db_path = Path("/home/alex/Documents/projects/ichack26/data/routes.db")

    # Check if PBF file exists
    if not pbf_path.exists():
        print(f"Error: PBF file not found at {pbf_path}")
        sys.exit(1)

    print(f"Starting OSM PBF transformation")
    print(f"Input:  {pbf_path} ({pbf_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Output: {db_path}")
    print()

    # Process the PBF file
    handler = OSMWayHandler(str(db_path))

    print("Reading OSM PBF file (this may take 5-10 minutes)...")
    print("Note: Building location index first (required for node coordinates)...")

    # Apply with location index - THIS IS CRITICAL
    handler.apply_file(str(pbf_path), locations=True)
    handler.finish()

    print(f"\nDatabase created: {db_path}")
    print(f"Size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
