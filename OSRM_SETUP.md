# OSRM Self-Hosted Setup

This project now uses a self-hosted OSRM (Open Source Routing Machine) instance for route matching instead of GraphHopper's API.

## Why OSRM?

- **No API limits**: Self-hosted means no rate limits or API keys
- **More waypoints**: Can handle 50+ waypoints vs GraphHopper's 5-waypoint free tier limit
- **Better map matching**: OSRM's Match API is specifically designed for GPS trace matching
- **Privacy**: All routing happens locally
- **Free and open source**

## Quick Start

### 1. Download and Process Map Data

First, download OpenStreetMap data for your region and process it for OSRM:

```bash
./setup-osrm.sh
```

By default, this downloads Greater London data. For other regions:

```bash
# For all of Great Britain
./setup-osrm.sh great-britain

# For England
./setup-osrm.sh england

# For Scotland
./setup-osrm.sh scotland
```

**Note**: Processing time depends on region size:
- Greater London: ~2-5 minutes
- England: ~15-30 minutes
- Great Britain: ~30-60 minutes

### 2. Start the OSRM Server

```bash
docker-compose up -d
```

This starts the OSRM server at `http://localhost:5000`

### 3. Verify It's Working

Test the OSRM server:

```bash
curl 'http://localhost:5000/route/v1/foot/-0.1278,51.5074;-0.1377,51.5074?overview=false'
```

You should see a JSON response with routing data.

### 4. Start Your Flask App

The Flask app is configured to use the OSRM server automatically:

```bash
python app.py
```

## Configuration

The OSRM base URL can be configured via environment variable:

```bash
export OSRM_BASE_URL=http://localhost:5000
```

## OSRM Services Used

- **Match API** (`/match/v1/foot/...`): Snaps GPS traces to the road network
  - Used for matching user-drawn shapes to actual routes
  - Returns confidence scores for match quality
  - Can handle complex, noisy GPS traces

## Troubleshooting

### OSRM server won't start

Check if map data is processed:
```bash
ls -lh osrm-data/map.osrm*
```

You should see several files. If not, run `./setup-osrm.sh` again.

### Port 5000 already in use

If you have another service on port 5000, edit `docker-compose.yml`:

```yaml
ports:
  - "5555:5000"  # Use port 5555 instead
```

Then update the OSRM URL:
```bash
export OSRM_BASE_URL=http://localhost:5555
```

### Connection refused errors

Make sure the OSRM container is running:
```bash
docker-compose ps
```

If it's not running:
```bash
docker-compose up -d
```

View logs:
```bash
docker-compose logs -f osrm-backend
```

### Out of memory during processing

If processing fails, your region may be too large. Try a smaller region or increase Docker's memory limit in Docker Desktop settings.

## Stopping the Server

```bash
docker-compose down
```

## Data Storage

Map data is stored in `./osrm-data/` (excluded from git).

File sizes vary by region:
- Greater London: ~200-500 MB
- England: ~2-4 GB
- Great Britain: ~4-8 GB

## Advanced: Custom Profiles

OSRM supports different routing profiles (foot, bike, car). The default setup uses `foot` profile suitable for running routes.

To use a different profile, modify the `setup-osrm.sh` script and replace `/opt/foot.lua` with:
- `/opt/bicycle.lua` for cycling
- `/opt/car.lua` for driving

## API Reference

### Match Service

```
GET /match/v1/foot/{coordinates}?overview=full&geometries=geojson
```

Where `coordinates` is semicolon-separated `lng,lat` pairs:
```
-0.1278,51.5074;-0.1377,51.5074;-0.1478,51.5074
```

Returns matched route with geometry and confidence score.

## Resources

- [OSRM Documentation](http://project-osrm.org/)
- [OSRM API Reference](https://project-osrm.org/docs/v5.24.0/api/)
- [Geofabrik Downloads](https://download.geofabrik.de/) - OSM data source
