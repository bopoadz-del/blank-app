# Unified Deployment Guide

This repository uses a unified deployment approach where the frontend and backend are built together and served through FastAPI.

## Architecture

```
┌─────────────────────────────────────────┐
│          Multi-Stage Dockerfile         │
├─────────────────────────────────────────┤
│  Stage 1: Build Frontend (Node 18)      │
│  - npm ci                                │
│  - npm run build → dist/                 │
├─────────────────────────────────────────┤
│  Stage 2: Build Backend (Python 3.11)   │
│  - Create virtualenv                     │
│  - Install Python dependencies           │
├─────────────────────────────────────────┤
│  Stage 3: Final Runtime Image            │
│  - Copy Python virtualenv                │
│  - Copy backend code                     │
│  - Copy frontend/dist → backend/frontend/dist │
│  - Run as non-root user                  │
└─────────────────────────────────────────┘
```

## How It Works

### Frontend Serving

The FastAPI application (`backend/app/main.py`) is configured to:

1. **Serve frontend at root (`/`)**: The built frontend (`frontend/dist`) is mounted at the root path with SPA-friendly routing (`html=True`)
2. **Health check at `/health`**: Platform health checks remain at the dedicated `/health` endpoint
3. **API routes under `/api/v1`**: All API endpoints are prefixed with `/api/v1` to avoid conflicts with frontend routes

### Path Resolution

```python
# In backend/app/main.py:
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"      # backend/frontend/dist

# Can override via environment variable:
FRONTEND_DIST_PATH=/custom/path
```

### Fallback Behavior

If the frontend build is not found, the API returns a JSON message at `/` indicating the frontend is missing.

## Building and Running

### Local Docker Build

```bash
# Build the unified image
docker build -t blank-unified .

# Run with default port
docker run -p 8000:8000 blank-unified

# Run with custom port (simulating Render)
docker run -p 3000:3000 -e PORT=3000 blank-unified
```

### Verify Deployment

```bash
# Frontend
curl http://localhost:8000/           # Should serve index.html

# Health check
curl http://localhost:8000/health     # Should return {"status": "ok", ...}

# API endpoints
curl http://localhost:8000/api/v1/formulas
```

## Render Deployment

When deploying to Render:

1. **Use this Dockerfile**: Point your Render Web Service to the root `Dockerfile`
2. **Port configuration**: Render automatically sets the `$PORT` environment variable, which the app respects
3. **Build process**: Render will run the multi-stage build automatically

### Render Configuration

```yaml
# In render.yaml or Web Service settings:
buildCommand: ""  # Not needed - uses Dockerfile
startCommand: ""  # Not needed - CMD in Dockerfile handles this
```

## Development Workflow

### Frontend Development

```bash
cd frontend
npm install
npm run dev  # Runs on port 3000 with proxy to backend
```

### Backend Development

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Testing the Unified Build Locally

```bash
# Build frontend
cd frontend && npm run build

# Copy frontend to backend (for local testing)
mkdir -p ../backend/frontend
cp -r dist ../backend/frontend/dist

# Run backend (will serve frontend)
cd ../backend
uvicorn app.main:app --reload
```

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Server port | `8000` | No (Render sets this) |
| `FRONTEND_DIST_PATH` | Custom frontend path | `backend/frontend/dist` | No |
| `DATABASE_URL` | PostgreSQL connection | - | Yes |
| `SECRET_KEY` | JWT secret | - | Yes |

## Testing

Run the deployment tests:

```bash
cd backend
pytest tests/test_frontend_mounting.py -v
```

Tests verify:
- ✅ Frontend path resolution
- ✅ Multi-stage Dockerfile structure
- ✅ .dockerignore optimization
- ✅ FastAPI frontend mounting logic

## Troubleshooting

### Frontend not appearing

1. Check if `backend/frontend/dist/index.html` exists in the container
2. Check logs for "Frontend mounted at / from..." message
3. Verify the build stage completed successfully

### API routes not working

- Ensure API routes are registered with `/api/v1` prefix
- Check that StaticFiles is mounted AFTER API route registration

### Port issues

- Render sets `$PORT` dynamically - don't hardcode port 8000
- Use `${PORT:-8000}` syntax to support both local and Render

## File Structure

```
repo-root/
├── Dockerfile                    # Multi-stage unified build
├── .dockerignore                 # Optimize build context
├── frontend/
│   ├── package.json
│   └── src/
└── backend/
    ├── app/
    │   └── main.py              # Frontend mounting logic
    ├── requirements.txt
    └── frontend/
        └── dist/                # Frontend copied here during build
            └── index.html
```

## Benefits

1. **Single deployment**: One Docker image contains both frontend and backend
2. **Simplified infrastructure**: No need for separate frontend hosting or reverse proxy
3. **Lower costs**: Single service on Render instead of two
4. **Better performance**: No CORS issues, reduced latency
5. **Easier development**: Consistent deployment across environments
