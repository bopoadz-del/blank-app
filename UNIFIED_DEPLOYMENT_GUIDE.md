# Unified UI+API Deployment Guide

This document explains the unified deployment configuration where the built frontend is served by FastAPI.

## Overview

The unified deployment uses a multi-stage Docker build to:
1. Build the React frontend with Node.js
2. Install Python backend dependencies into a virtualenv
3. Combine both into a single runtime image where FastAPI serves both the API and the frontend

## Architecture

```
User Request
     │
     ▼
FastAPI Application (port $PORT or 8000)
     │
     ├─► / (root) ────────► Frontend (React SPA from /dist)
     ├─► /health ─────────► Health check endpoint
     ├─► /metrics ────────► Prometheus metrics
     ├─► /docs ───────────► API documentation
     └─► /api/v1/* ───────► API endpoints
```

## How It Works

### Backend Changes (backend/app/main.py)

The backend now includes logic to mount the frontend at the root:

1. **Frontend Path Detection**: Looks for `backend/frontend/dist` directory
2. **Conditional Mounting**: If frontend exists with `index.html`, mounts it at root
3. **SPA Support**: Uses `StaticFiles(html=True)` for client-side routing
4. **Fallback**: If frontend not found, returns JSON response at root

```python
# Paths: this file sits at backend/app, so climb to backend/
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

# Allow overriding via env var (useful for tests)
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST_PATH", str(FRONTEND_DIST)))

if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
```

### Multi-Stage Dockerfile

**Stage 1: Frontend Builder**
- Uses `node:18-alpine` for small image size
- Installs dependencies with `npm ci`
- Builds production bundle with `npm run build`
- Output: `frontend/dist` directory

**Stage 2: Backend Builder**
- Uses `python:3.11-slim`
- Creates a Python virtualenv at `/opt/venv`
- Installs backend dependencies from `requirements.txt`
- Prepares backend code

**Stage 3: Runtime**
- Uses `python:3.11-slim` as final base
- Copies virtualenv from Stage 2
- Copies backend code from Stage 2
- Copies `frontend/dist` from Stage 1 into `backend/frontend/dist`
- Runs as non-root user for security
- Starts uvicorn with Render's `$PORT` environment variable

## Deployment

### Local Testing

```bash
# Build the image
docker build -t blank-unified .

# Run the container
docker run -p 8000:8000 -e PORT=8000 blank-unified

# Test endpoints
curl http://localhost:8000/health      # Health check
curl http://localhost:8000/            # Frontend (HTML)
curl http://localhost:8000/api/v1/...  # API endpoints
```

### Render Deployment

Render will automatically:
1. Build the Docker image
2. Set the `$PORT` environment variable
3. Start the container
4. Route traffic to your application

**Render Configuration:**
- **Service Type**: Web Service
- **Build Command**: (automatic, uses Dockerfile)
- **Start Command**: (automatic, uses Dockerfile CMD)
- **Environment Variables**: Set any required env vars (DATABASE_URL, etc.)

## Environment Variables

- `PORT`: Server port (set by Render automatically, defaults to 8000)
- `FRONTEND_DIST_PATH`: Override frontend directory path (for testing)
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- Other backend-specific variables (see backend/.env.example)

## Frontend Build Output

The frontend build must output to `frontend/dist` with at least:
- `index.html`: Main HTML file
- `assets/`: JavaScript, CSS, and other assets

Vite (used by the frontend) creates this structure by default:

```
frontend/dist/
├── index.html
├── assets/
│   ├── index-[hash].js
│   ├── index-[hash].css
│   └── ...
└── ...
```

## API Endpoints

All existing API endpoints remain accessible:

- `GET /health` - Health check (always accessible)
- `GET /metrics` - Prometheus metrics
- `GET /docs` - OpenAPI documentation
- `GET /api/v1/*` - All API endpoints

## SPA Routing

The frontend uses client-side routing (React Router). The configuration:

```python
app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
```

The `html=True` parameter ensures:
- Unknown routes (404s) serve `index.html`
- Client-side router handles navigation
- Deep links work correctly

## Troubleshooting

### Frontend Not Loading

**Symptom**: API JSON response at root instead of HTML

**Check:**
1. Verify frontend was built: `ls frontend/dist/index.html`
2. Check Docker logs for "Frontend mounted from..." message
3. Verify frontend/dist was copied into the image:
   ```bash
   docker run --rm -it blank-unified ls -la /app/backend/frontend/dist
   ```

### Frontend 404 Errors

**Symptom**: Frontend shows 404 for some routes

**Solution**: This is expected if frontend routing is not configured. The `html=True` parameter should handle this, but ensure your frontend router is set up correctly.

### API Endpoints Not Working

**Symptom**: 404 errors for API calls

**Check:**
1. Ensure API base URL is correct (check frontend `.env` or config)
2. Verify API endpoints are prefixed with `/api/v1`
3. Check browser console for CORS errors

### Port Issues on Render

**Symptom**: Application fails to start on Render

**Solution**: Render sets `$PORT` automatically. The Dockerfile CMD uses `${PORT:-8000}` to support both Render (with $PORT) and local testing (default 8000).

## Security Considerations

1. **Non-root User**: Container runs as `appuser` (non-root)
2. **Minimal Base Image**: Uses `python:3.11-slim` for smaller attack surface
3. **No Secrets in Image**: Use environment variables for secrets
4. **CORS**: Configure CORS properly for production (see backend/app/main.py)

## Performance Optimization

1. **Multi-stage Build**: Keeps final image small (~300-400MB)
2. **Layer Caching**: Package manifests copied first for better caching
3. **Virtualenv**: Isolates Python dependencies
4. **Static File Serving**: FastAPI serves static files efficiently

## Development Workflow

**Local Development** (separate frontend/backend):
```bash
# Terminal 1: Backend
cd backend
uvicorn app.main:app --reload

# Terminal 2: Frontend  
cd frontend
npm run dev
```

**Production Build** (unified):
```bash
docker build -t blank-unified .
docker run -p 8000:8000 blank-unified
```

## Testing

Run the unified deployment tests:

```bash
pytest tests/test_unified_deployment.py -v
```

These tests verify:
- Health endpoint is accessible
- Root endpoint behavior with/without frontend
- API endpoints still work
- Frontend mounting logic

## Maintenance

### Updating Dependencies

**Frontend:**
```bash
cd frontend
npm update
npm audit fix
```

**Backend:**
```bash
cd backend
pip install -U -r requirements.txt
pip freeze > requirements.txt
```

### Rebuilding

After any code changes:
```bash
docker build -t blank-unified .
docker run -p 8000:8000 blank-unified
```

## Additional Resources

- FastAPI Static Files: https://fastapi.tiangolo.com/tutorial/static-files/
- Docker Multi-stage Builds: https://docs.docker.com/build/building/multi-stage/
- Render Deployment: https://render.com/docs/deploy-fastapi
