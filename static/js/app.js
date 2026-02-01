// API base URL
const API_BASE = '/api';

// State management
const state = {
    currentPanel: 0,
    drawingData: null,
    location: null,
    selectedRoute: null,
    matchedRoutes: [],
    desiredDistance: 3.0  // miles
};

// Map instances
let locationMap = null;
let resultsMap = null;
let exportMap = null;
let locationMarker = null;

// Helper function to make API calls
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Panel Navigation
function scrollToPanel(panelIndex) {
    const container = document.getElementById('panels');
    const panel = container.children[panelIndex];

    // Smooth scroll to panel
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'start' });

    // Update state
    state.currentPanel = panelIndex;

    // Update progress bar
    updateProgressBar(panelIndex);
}

function updateProgressBar(activeIndex) {
    const steps = document.querySelectorAll('.progress-step');
    steps.forEach((step, index) => {
        step.classList.remove('active', 'completed');
        if (index === activeIndex) {
            step.classList.add('active');
        } else if (index < activeIndex) {
            step.classList.add('completed');
        }
    });
}

// Canvas Drawing Setup
let canvas, ctx;
let isDrawing = false;
let currentStroke = [];
let hasCompletedDrawing = false;

function initCanvas() {
    canvas = document.getElementById('drawing-canvas');
    ctx = canvas.getContext('2d');

    // Set canvas size
    const container = canvas.parentElement;
    canvas.width = Math.min(600, container.clientWidth - 40);
    canvas.height = Math.min(600, container.clientHeight - 40);

    // Drawing settings
    ctx.strokeStyle = '#667eea';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Click to start/stop drawing
    canvas.addEventListener('click', handleCanvasClick);
    canvas.addEventListener('mousemove', draw);

    // Touch support
    canvas.addEventListener('touchstart', handleTouchStart);
    canvas.addEventListener('touchmove', handleTouchMove);
    canvas.addEventListener('touchend', handleTouchEnd);

    // Update cursor
    updateCursor();
}

function handleCanvasClick(e) {
    if (hasCompletedDrawing) {
        // If there's already a drawing, reset and start new
        resetDrawing();
        startNewDrawing(e);
    } else if (isDrawing) {
        // Stop current drawing
        stopDrawing();
    } else {
        // Start new drawing
        startNewDrawing(e);
    }
}

function startNewDrawing(e) {
    isDrawing = true;
    hasCompletedDrawing = false;
    const pos = getMousePos(e);
    currentStroke = [pos];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    updateCursor();
}

function draw(e) {
    if (!isDrawing) return;

    const pos = getMousePos(e);
    currentStroke.push(pos);

    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
}

function stopDrawing() {
    if (isDrawing && currentStroke.length > 1) {
        isDrawing = false;
        hasCompletedDrawing = true;
        checkDrawingComplete();
        updateCursor();
    }
}

function resetDrawing() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    currentStroke = [];
    isDrawing = false;
    hasCompletedDrawing = false;
    checkDrawingComplete();
    updateCursor();
}

function generateCircle() {
    // Clear existing drawing
    resetDrawing();

    // Calculate circle parameters
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.35; // 70% of half the canvas

    // Generate circle points
    const numPoints = 36; // Smooth circle
    currentStroke = [];

    for (let i = 0; i <= numPoints; i++) {
        const angle = (2 * Math.PI * i) / numPoints;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        currentStroke.push({ x, y });
    }

    // Draw the circle
    ctx.beginPath();
    ctx.moveTo(currentStroke[0].x, currentStroke[0].y);
    for (let i = 1; i < currentStroke.length; i++) {
        ctx.lineTo(currentStroke[i].x, currentStroke[i].y);
    }
    ctx.stroke();

    // Mark as complete
    hasCompletedDrawing = true;
    checkDrawingComplete();
    updateCursor();

    console.log('Generated circle with', currentStroke.length, 'points');
}

function updateCursor() {
    if (hasCompletedDrawing) {
        canvas.style.cursor = 'pointer';  // Can click to restart
    } else if (isDrawing) {
        canvas.style.cursor = 'crosshair';  // Currently drawing
    } else {
        canvas.style.cursor = 'pointer';  // Can click to start
    }
}

function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

// Touch support functions
function handleTouchStart(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const clickEvent = new MouseEvent('click', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(clickEvent);
}

function handleTouchMove(e) {
    e.preventDefault();
    if (!isDrawing) return;

    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const pos = {
        x: touch.clientX - rect.left,
        y: touch.clientY - rect.top
    };

    currentStroke.push(pos);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
}

function handleTouchEnd(e) {
    e.preventDefault();
    // Touch end doesn't stop drawing - need to tap again to stop
}

function clearCanvas() {
    resetDrawing();
}

function undoStroke() {
    // Since we only have one stroke now, undo = clear
    resetDrawing();
}

function checkDrawingComplete() {
    const hasDrawing = hasCompletedDrawing && currentStroke.length > 1;
    const hasLocation = state.location !== null;

    document.getElementById('next-to-match').disabled = !(hasDrawing && hasLocation);
}

// Distance selector handling
function initDistanceSelector() {
    const slider = document.getElementById('distance-slider');
    const valueDisplay = document.getElementById('distance-value');

    // Set initial display from slider value
    const initialValue = parseFloat(slider.value);
    state.desiredDistance = initialValue;
    valueDisplay.textContent = initialValue.toFixed(1);

    // Update display when slider changes
    slider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        state.desiredDistance = value;
        valueDisplay.textContent = value.toFixed(1);
    });
}

// Map Initialization
function initLocationMap() {
    locationMap = new maplibregl.Map({
        container: 'location-map',
        style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        center: [-0.1276, 51.5074], // Default to London
        zoom: 12
    });

    locationMap.addControl(new maplibregl.NavigationControl(), 'top-right');

    // Add click handler for location selection
    locationMap.on('click', (e) => {
        setLocation(e.lngLat.lat, e.lngLat.lng);
    });
}

function initResultsMap() {
    resultsMap = new maplibregl.Map({
        container: 'results-map',
        style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        center: [-0.1276, 51.5074],
        zoom: 13
    });

    resultsMap.addControl(new maplibregl.NavigationControl(), 'top-right');

    // Map will be populated with routes when available
}

function initExportMap() {
    exportMap = new maplibregl.Map({
        container: 'export-map',
        style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        center: [-0.1276, 51.5074],
        zoom: 13
    });

    exportMap.addControl(new maplibregl.NavigationControl(), 'top-right');
}

// Location handling
function setLocation(lat, lng) {
    state.location = { lat, lng };

    // Remove old marker if exists
    if (locationMarker) {
        locationMarker.remove();
    }

    // Add new marker
    const el = document.createElement('div');
    el.className = 'location-marker';

    locationMarker = new maplibregl.Marker({ element: el })
        .setLngLat([lng, lat])
        .addTo(locationMap);

    // Update display
    const display = document.getElementById('location-display');
    display.textContent = `Location: ${lat.toFixed(4)}, ${lng.toFixed(4)}`;
    display.classList.add('active');

    // Pan to location
    locationMap.flyTo({ center: [lng, lat], zoom: 13 });

    checkDrawingComplete();
}

function useCurrentLocation() {
    if (!navigator.geolocation) {
        alert('Geolocation is not supported by your browser');
        return;
    }

    const btn = document.getElementById('use-current-location');
    btn.textContent = '📍 Getting location...';
    btn.disabled = true;

    navigator.geolocation.getCurrentPosition(
        (position) => {
            const lat = position.coords.latitude;
            const lng = position.coords.longitude;

            setLocation(lat, lng);

            btn.textContent = '📍 Location Set';
            btn.disabled = false;
        },
        (error) => {
            alert('Unable to get your location');
            btn.textContent = '📍 Use My Location';
            btn.disabled = false;
        }
    );
}

// Route rendering
function displayRoutes(routes) {
    console.log('Displaying routes:', routes);
    state.matchedRoutes = routes;
    const container = document.getElementById('routes-list-container');
    container.innerHTML = '';

    if (!routes || routes.length === 0) {
        container.innerHTML = '<p class="no-results">No matching routes found. Try a different shape or location.</p>';
        return;
    }

    console.log(`Found ${routes.length} routes to display`);

    // Add all routes to the list
    routes.forEach((route, index) => {
        const item = document.createElement('div');
        item.className = 'route-item';
        item.dataset.routeId = route.id || index;

        item.innerHTML = `
            <div class="route-item-header">
                <span class="route-name">${route.name || `Route ${index + 1}`}</span>
                <span class="route-match">${route.match_score || route.matchScore || '85'}% match</span>
            </div>
            <div class="route-details">
                <span>📏 ${route.distance || '2.3'} mi</span>
                <span>⏱️ ${route.duration || '25'} min</span>
            </div>
        `;

        item.addEventListener('click', () => selectRoute(route, item, index));
        container.appendChild(item);
    });

    // Render routes on map
    renderRoutesOnMap();

    // Automatically select the first route
    if (routes.length > 0) {
        const firstItem = container.querySelector('.route-item');
        selectRoute(routes[0], firstItem, 0);
    }
}

function selectRoute(route, element, index) {
    // Remove previous selection
    document.querySelectorAll('.route-item').forEach(item => {
        item.classList.remove('selected');
    });

    // Add selection to clicked item
    element.classList.add('selected');

    state.selectedRoute = route;

    // Enable export button
    document.getElementById('next-to-export').disabled = false;

    // Highlight selected route on map
    highlightRouteOnMap(index);

    // Zoom to the selected route
    if (route.coordinates && route.coordinates.length > 0) {
        fitMapToRoutes(resultsMap, [route]);
    }
}

// Map helper functions
function renderRoutesOnMap() {
    if (!resultsMap.loaded()) {
        resultsMap.on('load', () => renderRoutesOnMap());
        return;
    }

    // Clear existing routes
    clearRouteLayersFromMap(resultsMap);

    // Add all routes
    state.matchedRoutes.forEach((route, index) => {
        addRouteToMap(resultsMap, route, index);
    });

    // Fit map to show all routes
    if (state.matchedRoutes.length > 0 && state.matchedRoutes[0].coordinates) {
        fitMapToRoutes(resultsMap, state.matchedRoutes);
    }
}

function addRouteToMap(map, route, index) {
    if (!route.coordinates || route.coordinates.length === 0) return;

    const sourceId = `route-${index}`;
    const layerId = `route-layer-${index}`;

    // Convert coordinates to GeoJSON format
    const geojson = {
        type: 'Feature',
        properties: {
            name: route.name,
            index: index
        },
        geometry: {
            type: 'LineString',
            coordinates: route.coordinates.map(coord => [coord.lng, coord.lat])
        }
    };

    // Remove existing source/layer if it exists
    if (map.getLayer(layerId)) {
        map.removeLayer(layerId);
    }
    if (map.getSource(sourceId)) {
        map.removeSource(sourceId);
    }

    // Add source
    map.addSource(sourceId, {
        type: 'geojson',
        data: geojson
    });

    // Add layer with default styling
    const colors = ['#667eea', '#10b981', '#f59e0b'];
    const color = colors[index % colors.length];

    map.addLayer({
        id: layerId,
        type: 'line',
        source: sourceId,
        layout: {
            'line-join': 'round',
            'line-cap': 'round'
        },
        paint: {
            'line-color': color,
            'line-width': 4,
            'line-opacity': 0.7
        }
    });
}

function clearRouteLayersFromMap(map) {
    if (!map || !map.loaded()) return;

    // Get all layers safely
    const style = map.getStyle();
    if (!style || !style.layers) return;

    // Remove all route layers and sources
    style.layers.forEach(layer => {
        if (layer.id.startsWith('route-layer-')) {
            if (map.getLayer(layer.id)) {
                map.removeLayer(layer.id);
            }
        }
    });

    const sources = Object.keys(style.sources || {});
    sources.forEach(sourceId => {
        if (sourceId.startsWith('route-')) {
            if (map.getSource(sourceId)) {
                map.removeSource(sourceId);
            }
        }
    });
}

function highlightRouteOnMap(selectedIndex) {
    const colors = ['#667eea', '#10b981', '#f59e0b'];

    state.matchedRoutes.forEach((route, index) => {
        const layerId = `route-layer-${index}`;

        if (resultsMap.getLayer(layerId)) {
            const isSelected = index === selectedIndex;

            resultsMap.setPaintProperty(
                layerId,
                'line-color',
                isSelected ? '#ffffff' : colors[index % 3]
            );
            resultsMap.setPaintProperty(
                layerId,
                'line-width',
                isSelected ? 6 : 4
            );
            resultsMap.setPaintProperty(
                layerId,
                'line-opacity',
                isSelected ? 1.0 : 0.4
            );
        }
    });
}

function fitMapToRoutes(map, routes) {
    const allCoords = routes.flatMap(route =>
        route.coordinates.map(coord => [coord.lng, coord.lat])
    );

    if (allCoords.length === 0) return;

    const bounds = allCoords.reduce((bounds, coord) => {
        return bounds.extend(coord);
    }, new maplibregl.LngLatBounds(allCoords[0], allCoords[0]));

    map.fitBounds(bounds, {
        padding: 50,
        duration: 1000
    });
}

function displaySelectedRoute() {
    const container = document.getElementById('selected-route-info');

    if (!state.selectedRoute) {
        container.innerHTML = '<p class="no-selection">No route selected yet...</p>';
        return;
    }

    const route = state.selectedRoute;
    container.innerHTML = `
        <div class="stat">
            <span class="stat-label">Distance</span>
            <span class="stat-value">${route.distance || '2.3'} miles</span>
        </div>
        <div class="stat">
            <span class="stat-label">Estimated Time</span>
            <span class="stat-value">${route.duration || '25'} minutes</span>
        </div>
        <div class="stat">
            <span class="stat-label">Match Score</span>
            <span class="stat-value">${route.match_score || route.matchScore || '85'}%</span>
        </div>
        <div class="stat">
            <span class="stat-label">Elevation Gain</span>
            <span class="stat-value">${route.elevation_gain || route.elevationGain || '120'} ft</span>
        </div>
    `;

    // Enable download button
    document.getElementById('download-gpx').disabled = false;

    // Enable publish button
    document.getElementById('publish-route').disabled = false;

    // Display route on export map
    if (exportMap.loaded()) {
        renderExportMap();
    } else {
        exportMap.on('load', renderExportMap);
    }
}

function renderExportMap() {
    if (!state.selectedRoute) return;

    clearRouteLayersFromMap(exportMap);
    addRouteToMap(exportMap, state.selectedRoute, 0);

    // Make it white/highlighted
    setTimeout(() => {
        if (exportMap.getLayer('route-layer-0')) {
            exportMap.setPaintProperty('route-layer-0', 'line-color', '#ffffff');
            exportMap.setPaintProperty('route-layer-0', 'line-width', 6);
            exportMap.setPaintProperty('route-layer-0', 'line-opacity', 1);
        }
    }, 100);

    // Fit map to route
    if (state.selectedRoute.coordinates && state.selectedRoute.coordinates.length > 0) {
        fitMapToRoutes(exportMap, [state.selectedRoute]);
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize canvas
    initCanvas();

    // Initialize maps
    initLocationMap();
    initResultsMap();
    initExportMap();

    // Initialize distance selector
    initDistanceSelector();

    // Panel 1 controls
    document.getElementById('clear-canvas').addEventListener('click', clearCanvas);
    document.getElementById('undo-stroke').addEventListener('click', undoStroke);
    document.getElementById('generate-circle').addEventListener('click', generateCircle);
    document.getElementById('use-current-location').addEventListener('click', useCurrentLocation);

    document.getElementById('next-to-match').addEventListener('click', async () => {
        // Save drawing data (now a single stroke)
        state.drawingData = currentStroke;

        // Navigate to results page immediately
        scrollToPanel(1);

        // Resize maps after scroll
        setTimeout(() => {
            resultsMap.resize();
        }, 500);

        // Show loading in results panel
        const routesList = document.getElementById('routes-list-container');
        routesList.innerHTML = '<p class="no-results">Finding routes...</p>';

        try {
            // Call API to find matching routes
            const response = await apiCall('/match-routes', {
                method: 'POST',
                body: JSON.stringify({
                    location: state.location,
                    shape: state.drawingData,
                    desired_distance_miles: state.desiredDistance
                })
            });

            console.log('API Response:', response);

            // Check if we got routes
            if (!response.routes || response.routes.length === 0) {
                routesList.innerHTML = '<p class="no-results">No routes found. Try a different location or distance.</p>';
                return;
            }

            // Display the routes
            displayRoutes(response.routes);

            // Update results map center
            if (state.location && resultsMap) {
                resultsMap.setCenter([state.location.lng, state.location.lat]);
                resultsMap.setZoom(13);
            }

        } catch (error) {
            console.error('Route matching error:', error);
            routesList.innerHTML = `<p class="no-results">Failed to find routes: ${error.message}<br><br>Please try again or choose a different location.</p>`;
        }
    });

    // Panel 2 controls
    document.getElementById('back-to-draw').addEventListener('click', () => {
        scrollToPanel(0);
    });

    document.getElementById('next-to-export').addEventListener('click', () => {
        displaySelectedRoute();
        scrollToPanel(2);

        // Resize export map after scroll
        setTimeout(() => {
            exportMap.resize();
        }, 500);
    });

    // Panel 3 controls
    document.getElementById('back-to-match').addEventListener('click', () => {
        scrollToPanel(1);
    });

    document.getElementById('start-over').addEventListener('click', () => {
        // Reset state
        state.drawingData = null;
        state.location = null;
        state.selectedRoute = null;
        state.matchedRoutes = [];

        // Clear canvas
        resetDrawing();

        // Clear location
        if (locationMarker) {
            locationMarker.remove();
            locationMarker = null;
        }
        const display = document.getElementById('location-display');
        display.textContent = 'Click on the map or use your current location';
        display.classList.remove('active');

        // Reset buttons
        document.getElementById('next-to-export').disabled = true;
        document.getElementById('download-gpx').disabled = true;

        scrollToPanel(0);
    });

    document.getElementById('download-gpx').addEventListener('click', async () => {
        if (!state.selectedRoute) {
            alert('No route selected');
            return;
        }

        try {
            // Call the GPX export endpoint
            const response = await fetch(`${API_BASE}/export-gpx`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    route: state.selectedRoute,
                    location: state.location
                })
            });

            if (!response.ok) {
                throw new Error(`Export failed: ${response.status}`);
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
            console.error('GPX download failed:', error);
            alert(`Failed to download GPX: ${error.message}`);
        }
    });

    // Image upload functionality
    const imageUploadInput = document.getElementById('image-upload');
    const uploadImageBtn = document.getElementById('upload-image-btn');
    const uploadStatus = document.getElementById('upload-status');

    uploadImageBtn.addEventListener('click', () => {
        imageUploadInput.click();
    });

    imageUploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Show upload status
        uploadStatus.textContent = 'Processing image...';
        uploadImageBtn.disabled = true;

        try {
            // Create FormData for file upload
            const formData = new FormData();
            formData.append('image', file);
            formData.append('num_points', '800');  // Adjust for detail level

            // Upload image to backend
            const response = await fetch(`${API_BASE}/image-to-line`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status}`);
            }

            const data = await response.json();

            // Clear canvas
            resetDrawing();

            // Draw the converted line on canvas
            const canvas = document.getElementById('drawing-canvas');
            const ctx = canvas.getContext('2d');

            // Scale points to fit canvas
            const [imgWidth, imgHeight] = data.image_size;
            const scaleX = canvas.width / imgWidth;
            const scaleY = canvas.height / imgHeight;
            const scale = Math.min(scaleX, scaleY);

            // Center the image
            const offsetX = (canvas.width - imgWidth * scale) / 2;
            const offsetY = (canvas.height - imgHeight * scale) / 2;

            // Draw the line
            ctx.strokeStyle = '#2c3e50';
            ctx.lineWidth = 2;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.beginPath();

            data.points.forEach((point, index) => {
                const x = point.x * scale + offsetX;
                const y = point.y * scale + offsetY;

                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });

            ctx.stroke();

            // Save the points to state
            state.drawingData = data.points;

            // Mark drawing as completed
            hasCompletedDrawing = true;
            currentStroke = data.points;

            // Update status
            uploadStatus.textContent = `✓ Loaded ${data.num_points} points from image`;
            uploadStatus.style.color = '#27ae60';

            // Check if we can enable the next button
            checkDrawingComplete();

        } catch (error) {
            console.error('Image upload failed:', error);
            uploadStatus.textContent = `✗ Upload failed: ${error.message}`;
            uploadStatus.style.color = '#e74c3c';
        } finally {
            uploadImageBtn.disabled = false;
            // Clear the input so the same file can be uploaded again
            imageUploadInput.value = '';
        }
    });

    // Publish route functionality
    document.getElementById('publish-route').addEventListener('click', async () => {
        if (!state.selectedRoute) return;

        const publishButton = document.getElementById('publish-route');
        const publishStatus = document.getElementById('publish-status');

        // Disable button and show loading state
        publishButton.disabled = true;
        publishButton.textContent = '⏳ Publishing...';
        publishStatus.style.display = 'none';

        try {
            // Prepare route data for publishing
            const routeData = {
                name: document.getElementById('route-name').value || 'My Route',
                distance: state.selectedRoute.distance,
                duration: state.selectedRoute.duration,
                coordinates: state.selectedRoute.coordinates,
                location: state.location,
                elevation_gain: state.selectedRoute.elevation_gain
            };

            // Call backend API to publish route
            const response = await fetch(`${API_BASE}/publish-route`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(routeData)
            });

            if (!response.ok) {
                throw new Error('Failed to publish route');
            }

            const data = await response.json();
            const routeId = data.id;
            const shareLink = `${window.location.origin}/route/${routeId}`;

            // Show the share link section
            document.getElementById('share-link-section').style.display = 'flex';
            document.getElementById('share-link').value = shareLink;

            // Show success message
            publishStatus.textContent = '✓ Route published successfully! Share the link below.';
            publishStatus.style.display = 'block';
            publishStatus.style.color = '#10b981';

            // Update button
            publishButton.textContent = '✓ Published';
            publishButton.style.background = '#10b981';

            console.log('Route published with ID:', routeId);
            console.log('Share link:', shareLink);

        } catch (error) {
            console.error('Failed to publish route:', error);
            publishStatus.textContent = '✗ Failed to publish route. Please try again.';
            publishStatus.style.display = 'block';
            publishStatus.style.color = '#e74c3c';

            publishButton.textContent = '🌐 Publish to Community';
            publishButton.disabled = false;
        }
    });

    document.getElementById('copy-link').addEventListener('click', async () => {
        const shareLinkInput = document.getElementById('share-link');
        const copyButton = document.getElementById('copy-link');

        try {
            // Copy to clipboard
            await navigator.clipboard.writeText(shareLinkInput.value);

            // Update button to show success
            const originalText = copyButton.textContent;
            copyButton.textContent = '✓ Copied!';
            copyButton.style.backgroundColor = '#27ae60';

            // Reset after 2 seconds
            setTimeout(() => {
                copyButton.textContent = originalText;
                copyButton.style.backgroundColor = '';
            }, 2000);
        } catch (error) {
            console.error('Failed to copy:', error);
            alert('Failed to copy link. Please copy it manually.');
        }
    });
});
