// Route detail page

const API_BASE = '/api';

// Load route data
async function loadRoute() {
    try {
        // Fetch route from API
        const response = await fetch(`${API_BASE}/routes/${routeId}`);

        if (!response.ok) {
            throw new Error('Route not found');
        }

        const route = await response.json();

        if (!route) {
            showError('Route not found');
            return;
        }

        // Update page with route data
        document.getElementById('route-name').textContent = route.name;
        document.getElementById('route-distance').textContent = `${route.distance} miles`;
        document.getElementById('route-duration').textContent = `${route.duration} min`;
        document.getElementById('route-elevation').textContent = `${route.elevation_gain} ft`;
        document.getElementById('route-location').textContent = route.location.name;

        // Update page title
        document.title = `${route.name} - Route Matcher`;

        // Initialize map
        initMap(route);

        // Setup download button
        document.getElementById('download-gpx-btn').addEventListener('click', () => {
            downloadGPX(route);
        });

    } catch (error) {
        console.error('Failed to load route:', error);
        showError('Failed to load route');
    }
}

// Initialize map with route
function initMap(route) {
    const map = new maplibregl.Map({
        container: 'route-map',
        style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        center: [route.location.lng, route.location.lat],
        zoom: 12
    });

    map.addControl(new maplibregl.NavigationControl(), 'top-right');

    map.on('load', () => {
        // Add route line
        map.addSource('route', {
            type: 'geojson',
            data: {
                type: 'Feature',
                geometry: {
                    type: 'LineString',
                    coordinates: route.coordinates.map(coord => [coord.lng, coord.lat])
                }
            }
        });

        map.addLayer({
            id: 'route-line',
            type: 'line',
            source: 'route',
            paint: {
                'line-color': '#667eea',
                'line-width': 4,
                'line-opacity': 1
            }
        });

        // Add start marker
        const startCoord = route.coordinates[0];
        const el = document.createElement('div');
        el.style.width = '20px';
        el.style.height = '20px';
        el.style.borderRadius = '50%';
        el.style.background = '#10b981';
        el.style.border = '3px solid white';
        el.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)';

        new maplibregl.Marker({ element: el })
            .setLngLat([startCoord.lng, startCoord.lat])
            .addTo(map);

        // Fit map to route
        const bounds = route.coordinates.reduce((bounds, coord) => {
            return bounds.extend([coord.lng, coord.lat]);
        }, new maplibregl.LngLatBounds(
            [route.coordinates[0].lng, route.coordinates[0].lat],
            [route.coordinates[0].lng, route.coordinates[0].lat]
        ));

        map.fitBounds(bounds, { padding: 50 });
    });
}

// Download GPX file
async function downloadGPX(route) {
    try {
        const response = await fetch(`${API_BASE}/export-gpx`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ route, location: route.location })
        });

        if (!response.ok) {
            throw new Error('Failed to generate GPX');
        }

        // Get the filename from the Content-Disposition header
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'route.gpx';
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }

        // Download the file
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        console.log('GPX file downloaded:', filename);

    } catch (error) {
        console.error('Failed to download GPX:', error);
        alert('Failed to download GPX file');
    }
}

// Show error message
function showError(message) {
    document.getElementById('route-name').textContent = message;
    document.querySelector('.route-stats-grid').style.display = 'none';
    document.querySelector('.route-actions').style.display = 'none';
}

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    loadRoute();
});
