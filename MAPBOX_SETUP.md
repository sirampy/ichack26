# Mapbox API Integration Setup

## 1. Get Your Mapbox API Token

1. Go to https://account.mapbox.com/auth/signup/
2. Sign up for a free account (no credit card required)
3. After signup, you'll be on the dashboard
4. Your **Default Public Token** will be shown - copy it
5. Free tier includes: **100,000 requests/month** (plenty for a hackathon!)

## 2. Install Dependencies

```bash
cd /home/alex/Documents/projects/ichack26
source .venv/bin/activate  # Activate your virtual environment
pip install -r requirements.txt
```

## 3. Set Your API Token

### Option A: Environment Variable (Recommended)
```bash
export MAPBOX_TOKEN="pk.eyJ1IjoieW91ci11c2VybmFtZSIsImEiOiJ5b3VyLXRva2VuIn0..."
```

### Option B: Add to your shell profile (permanent)
```bash
echo 'export MAPBOX_TOKEN="pk.eyJ1IjoieW91ci11c2VybmFtZSIsImEiOiJ5b3VyLXRva2VuIn0..."' >> ~/.bashrc
source ~/.bashrc
```

### Option C: Set before running Flask (quick test)
```bash
MAPBOX_TOKEN="pk.eyJ1IjoieW91ci11c2VybmFtZSIsImEiOiJ5b3VyLXRva2VuIn0..." python app.py
```

## 4. Run Your App

```bash
python app.py
```

Your app should now be running at http://localhost:5000

## 5. Test It

1. Open http://localhost:5000 in your browser
2. Draw a shape on the canvas
3. Click on the map to set a location
4. Click "Find Routes"
5. You should see a matched route appear!

## How It Works Now

1. User draws a shape on canvas
2. Shape gets transformed to geographic coordinates
3. Coordinates are sent to Mapbox Map Matching API
4. Mapbox snaps the shape to real walkable roads
5. You get back a beautiful, realistic route!

## Troubleshooting

### "Mapbox API token not configured"
- Make sure you've exported the MAPBOX_TOKEN environment variable
- Restart your terminal after setting it
- Check that the token starts with `pk.`

### "Mapbox API error: 401"
- Your token is invalid or expired
- Generate a new token from your Mapbox dashboard

### "Could not match route to roads"
- The location might be in an area with no walkable paths
- Try a different location (cities work best)
- Try a simpler shape

### API Rate Limits
- Free tier: 100,000 requests/month
- That's ~3,300 requests per day
- More than enough for development and demos!

## Alternative: OSRM Free API (No token needed)

If you don't want to sign up for Mapbox, you can use the free OSRM API:
- Change the URL to: `http://router.project-osrm.org/match/v1/foot/...`
- No token required!
- Less reliable (public server), but great for quick testing
