#!/bin/bash

# OSRM Setup Script
# This script downloads OpenStreetMap data and prepares it for OSRM

set -e

REGION=${1:-"greater-london"}
# Geofabrik URL structure (note: great-britain is now united-kingdom)
case "${REGION}" in
    "greater-london")
        OSM_URL="https://download.geofabrik.de/europe/united-kingdom/england/greater-london-latest.osm.pbf"
        ;;
    "england")
        OSM_URL="https://download.geofabrik.de/europe/united-kingdom/england-latest.osm.pbf"
        ;;
    "scotland")
        OSM_URL="https://download.geofabrik.de/europe/united-kingdom/scotland-latest.osm.pbf"
        ;;
    "united-kingdom")
        OSM_URL="https://download.geofabrik.de/europe/united-kingdom-latest.osm.pbf"
        ;;
    *)
        OSM_URL="${REGION}"  # Allow custom URL
        ;;
esac
DATA_DIR="./osrm-data"
MAP_FILE="${DATA_DIR}/map.osm.pbf"

echo "==================================="
echo "OSRM Map Data Setup"
echo "==================================="
echo "Region: ${REGION}"
echo "Data directory: ${DATA_DIR}"
echo ""

# Create data directory
mkdir -p ${DATA_DIR}

# Download map data if not exists
if [ ! -f "${MAP_FILE}" ]; then
    echo "Downloading map data for ${REGION}..."
    curl -L "${OSM_URL}" -o "${MAP_FILE}"
    echo "✓ Download complete"
else
    echo "Map data already exists, skipping download"
fi

echo ""
echo "Processing map data with OSRM..."
echo "(This may take several minutes)"
echo ""

# Extract
echo "1/3 Extracting..."
docker run --rm -v "${PWD}/${DATA_DIR}:/data" osrm/osrm-backend:latest \
    osrm-extract -p /opt/foot.lua /data/map.osm.pbf

# Partition
echo "2/3 Partitioning..."
docker run --rm -v "${PWD}/${DATA_DIR}:/data" osrm/osrm-backend:latest \
    osrm-partition /data/map.osrm

# Customize
echo "3/3 Customizing..."
docker run --rm -v "${PWD}/${DATA_DIR}:/data" osrm/osrm-backend:latest \
    osrm-customize /data/map.osrm

echo ""
echo "==================================="
echo "✓ OSRM setup complete!"
echo "==================================="
echo ""
echo "You can now start the OSRM server with:"
echo "  docker-compose up -d"
echo ""
echo "The OSRM API will be available at:"
echo "  http://localhost:5000"
echo ""
echo "Test it with:"
echo "  curl 'http://localhost:5000/route/v1/driving/-0.1278,51.5074;-0.1377,51.5074?overview=false'"
echo ""
