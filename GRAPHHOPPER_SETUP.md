# GraphHopper API Setup

**No billing info required!** ✅

## 1. Get Your Free GraphHopper API Key

1. Go to https://graphhopper.com/dashboard/
2. Click **"Sign Up"** (top right)
3. Fill in basic info - **NO credit card required**
4. Verify your email
5. Go to dashboard: https://graphhopper.com/dashboard/
6. Your API key will be shown at the top
7. Copy it (looks like: `a1b2c3d4-e5f6-7890-abcd-1234567890ab`)

**Free Tier:** 500 requests per day (perfect for demos!)

## 2. Set Your API Key

```bash
cd /home/alex/Documents/projects/ichack26
source .venv/bin/activate

# Set the API key
export GRAPHHOPPER_KEY="your-api-key-here"
```

## 3. Run Your App

```bash
python app.py
```

Visit http://localhost:5000

## 4. Test It

1. Draw a shape on the canvas
2. Click a location on the map
3. Set your desired distance
4. Click "Find Routes"
5. See your matched route appear! 🎉

## Make It Permanent (Optional)

Add to your `~/.bashrc` so you don't have to export it every time:

```bash
echo 'export GRAPHHOPPER_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## Quick Start Command

```bash
# One-liner to run (replace with your key)
GRAPHHOPPER_KEY="your-api-key-here" python app.py
```

## How It Works

1. Your drawn shape → Geographic coordinates
2. Coordinates sent to GraphHopper Map Matching API
3. GraphHopper snaps them to real walking paths
4. Returns beautiful, realistic route!

## Troubleshooting

### "GraphHopper API key not configured"
```bash
# Make sure you've exported the key
echo $GRAPHHOPPER_KEY  # Should print your key

# If empty, set it again
export GRAPHHOPPER_KEY="your-key-here"
```

### "GraphHopper API error: 401"
- Your API key is invalid
- Get a new one from the dashboard

### "GraphHopper API error: 429" (Rate limit)
- You've hit the 500 requests/day limit
- Wait until tomorrow or upgrade your plan

### "Could not match route to roads"
- Try a location in a city (better road coverage)
- Simplify your shape (fewer complex turns)
- Try a different area

## API Limits

| Plan | Requests/Day | Cost |
|------|--------------|------|
| Free | 500 | $0 |
| Standard | Unlimited | $99/month |

500/day = plenty for hackathon demos and development!
