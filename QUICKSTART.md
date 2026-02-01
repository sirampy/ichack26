# Quick Start Guide

## Setup OSRM and Run the App

### Step 1: Setup OSRM (one-time setup)

Download and process map data for routing:

```bash
./setup-osrm.sh
```

This will:
- Download Greater London OpenStreetMap data (~150 MB)
- Process it for OSRM routing (~2-5 minutes)
- Create files in `./osrm-data/`

### Step 2: Start OSRM Server

```bash
docker-compose up -d
```

OSRM will be available at `http://localhost:5000`

### Step 3: Start Flask App

```bash
python app.py
```

Flask will run at `http://localhost:8000`

### Step 4: Open in Browser

Navigate to:
```
http://localhost:8000
```

## Ports

- **OSRM Server**: `http://localhost:5000`
- **Flask App**: `http://localhost:8000`

## Stopping Everything

```bash
# Stop OSRM
docker-compose down

# Stop Flask (Ctrl+C in terminal)
```

## Troubleshooting

### "Cannot connect to OSRM server"

Make sure OSRM is running:
```bash
docker-compose ps
```

If not running:
```bash
docker-compose up -d
```

### "Map data not processed"

Run the setup script:
```bash
./setup-osrm.sh
```

### Check OSRM logs

```bash
docker-compose logs -f osrm-backend
```

## What Changed?

✅ **Before**: GraphHopper API (limited to 5 waypoints, requires API key)
✅ **Now**: Self-hosted OSRM (50+ waypoints, no limits, better matching)

The route matching is now faster, more accurate, and has no API limitations!
