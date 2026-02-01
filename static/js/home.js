// Home page - Community routes feed

const API_BASE = '/api';

// Helper function to display location
function getLocationDisplay(location) {
    if (!location) return 'Unknown location';
    if (location.name) return location.name;
    if (location.lat && location.lng) {
        return `${location.lat.toFixed(4)}, ${location.lng.toFixed(4)}`;
    }
    return 'Unknown location';
}

// Load and display routes
async function loadRoutes() {
    const grid = document.getElementById('routes-grid');

    try {
        // Fetch routes from API
        const response = await fetch(`${API_BASE}/routes`);
        const data = await response.json();
        const routes = data.routes;

        if (routes.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    <h3>No routes yet</h3>
                    <p>Be the first to create and share a route with the community!</p>
                    <a href="/new" class="btn-primary">Create First Route</a>
                </div>
            `;
            return;
        }

        // Clear loading message
        grid.innerHTML = '';

        // Create route cards
        routes.forEach(route => {
            const card = createRouteCard(route);
            grid.appendChild(card);
        });

    } catch (error) {
        console.error('Failed to load routes:', error);
        grid.innerHTML = `
            <div class="empty-state">
                <h3>Failed to load routes</h3>
                <p>Please try again later.</p>
            </div>
        `;
    }
}

// Create a route card element
function createRouteCard(route) {
    const card = document.createElement('div');
    card.className = 'route-card';
    card.onclick = () => window.location.href = `/route/${route.id}`;

    card.innerHTML = `
        <div class="route-card-map" id="map-${route.id}"></div>
        <div class="route-card-content">
            <div class="route-card-header">
                <h3 class="route-card-title">${route.name}</h3>
            </div>
            <div class="route-card-stats">
                <div class="route-stat">
                    📏 <span class="route-stat-value">${route.distance}</span> mi
                </div>
            </div>
            <div class="route-card-footer">
                <span class="route-card-location">📍 ${getLocationDisplay(route.location)}</span>
            </div>
        </div>
    `;

    // Initialize mini map for this route after DOM insertion
    setTimeout(() => initMiniMap(route), 100);

    return card;
}

// Initialize a mini map for a route card
function initMiniMap(route) {
    const mapContainer = document.getElementById(`map-${route.id}`);
    if (!mapContainer) return;

    const map = new maplibregl.Map({
        container: `map-${route.id}`,
        style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        center: [route.location.lng, route.location.lat],
        zoom: 12,
        interactive: false // Disable map interactions
    });

    map.on('load', () => {
        // Add route line
        map.addSource(`route-${route.id}`, {
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
            id: `route-layer-${route.id}`,
            type: 'line',
            source: `route-${route.id}`,
            paint: {
                'line-color': '#667eea',
                'line-width': 3,
                'line-opacity': 0.8
            }
        });

        // Fit map to route
        const bounds = route.coordinates.reduce((bounds, coord) => {
            return bounds.extend([coord.lng, coord.lat]);
        }, new maplibregl.LngLatBounds(
            [route.coordinates[0].lng, route.coordinates[0].lat],
            [route.coordinates[0].lng, route.coordinates[0].lat]
        ));

        map.fitBounds(bounds, { padding: 20 });
    });
}

// Filter and sort functionality
function setupFilters() {
    const distanceFilter = document.getElementById('distance-filter');
    const sortFilter = document.getElementById('sort-filter');

    distanceFilter.addEventListener('change', () => {
        // TODO: Implement filtering
        console.log('Filter by distance:', distanceFilter.value);
    });

    sortFilter.addEventListener('change', () => {
        // TODO: Implement sorting
        console.log('Sort by:', sortFilter.value);
    });
}

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    loadRoutes();
    setupFilters();
});
