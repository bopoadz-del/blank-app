# Unified Deployment Guide

This guide explains how to deploy the frontend and backend as a unified application using the `Dockerfile.unified`.

## Overview

The unified deployment packages both the React frontend and FastAPI backend into a single Docker image. The backend serves the built frontend as static files, eliminating the need for separate deployment of the frontend.

## Architecture

```
┌─────────────────────────────────────────┐
│  Docker Container (Port 8000)           │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  FastAPI Backend                  │ │
│  │                                   │ │
│  │  ┌─────────────────────────────┐ │ │
│  │  │  Static File Mount          │ │ │
│  │  │  (frontend/dist)            │ │ │
│  │  │  - index.html               │ │ │
│  │  │  - /assets/*                │ │ │
│  │  └─────────────────────────────┘ │ │
│  │                                   │ │
│  │  API Routes:                      │ │
│  │  - /api/v1/*                      │ │
│  │  - /health                        │ │
│  │  - /metrics                       │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Route Handling

- **`/`** - Serves `index.html` (React app entry point)
- **`/assets/*`** - Serves frontend static assets (JS, CSS, images)
- **`/api/v1/*`** - API endpoints (JSON responses)
- **`/health`** - Health check endpoint
- **`/metrics`** - Prometheus metrics endpoint
- **`/docs`** - Swagger API documentation

## How It Works

### 1. Build Process

The `Dockerfile.unified` uses a multi-stage build:

**Stage 1: Frontend Builder**
- Base: `node:18-alpine`
- Installs npm dependencies
- Runs `npm run build` to create production frontend
- Output: `frontend/dist/` directory

**Stage 2: Backend Builder**
- Base: `python:3.11-slim`
- Installs Python dependencies from `backend/requirements.txt`
- Prepares backend application files

**Stage 3: Final Runtime Image**
- Base: `python:3.11-slim`
- Copies Python packages and backend code from Stage 2
- Copies built frontend from Stage 1 to `backend/frontend/dist/`
- Runs `uvicorn` web server

### 2. Frontend Serving

The FastAPI backend (`backend/app/main.py`) includes logic to:

1. Check if `backend/frontend/dist/` exists
2. If yes, mount it as static files at the root path
3. If no, serve a JSON response indicating the backend-only mode

```python
from fastapi.staticfiles import StaticFiles
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/app -> backend
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
```

### 3. API Route Priority

FastAPI routes are registered **before** the static file mount, ensuring they take precedence:

1. API routes are registered first (`/api/v1/*`, `/health`, `/metrics`)
2. Static files are mounted last at `/`
3. Specific routes override the catch-all static mount

## Building the Image

### Local Build

```bash
docker build -f Dockerfile.unified -t blank-app:unified .
```

### Build Options

For faster builds during development, you can use BuildKit:

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.unified -t blank-app:unified .
```

## Running the Container

### Basic Run

```bash
docker run -p 8000:8000 blank-app:unified
```

### With Environment Variables

```bash
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:pass@host:5432/db" \
  -e PORT=8000 \
  blank-app:unified
```

### With Custom Port (for Render)

```bash
docker run -p 3000:3000 -e PORT=3000 blank-app:unified
```

## Deployment on Render

### Configuration

1. **Service Type**: Web Service
2. **Environment**: Docker
3. **Dockerfile Path**: `Dockerfile.unified`
4. **Port**: Render will automatically set the `$PORT` environment variable

### Render Settings

- **Build Command**: (automatic - uses Dockerfile)
- **Start Command**: (automatic - uses CMD from Dockerfile)
- **Health Check Path**: `/health`

### Environment Variables

Set these in the Render dashboard:

- `DATABASE_URL` - PostgreSQL connection string
- `SECRET_KEY` - JWT secret key
- `REDIS_URL` - Redis connection string (optional)
- Any other app-specific variables from `.env.example`

### Auto-Deploy

The Dockerfile is configured to work with Render's auto-deploy:

1. Push changes to your repository
2. Render detects the push
3. Builds the Docker image using `Dockerfile.unified`
4. Deploys the new image
5. The app starts on Render's assigned `$PORT`

## Testing Locally

### 1. Build the Image

```bash
docker build -f Dockerfile.unified -t blank-app:unified .
```

### 2. Run the Container

```bash
docker run -p 8000:8000 -e PORT=8000 blank-app:unified
```

### 3. Access the Application

- **Frontend**: http://localhost:8000/
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Endpoint**: http://localhost:8000/api/v1/formulas

### 4. Verify Frontend

Open http://localhost:8000/ in your browser. You should see the React application load.

### 5. Verify API

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/formulas
```

## Development Workflow

### Option 1: Separate Frontend/Backend (Development)

During development, run frontend and backend separately:

```bash
# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
```

The frontend dev server (Vite) proxies API requests to the backend.

### Option 2: Unified Docker (Production-like)

Test the production build locally:

```bash
docker build -f Dockerfile.unified -t blank-app:unified .
docker run -p 8000:8000 blank-app:unified
```

## Troubleshooting

### Issue: Frontend not loading

**Check 1**: Verify frontend was built
```bash
ls frontend/dist/
# Should show: index.html, assets/
```

**Check 2**: Check backend logs
```bash
docker logs <container-id>
# Look for: "Frontend static files mounted from..."
```

**Check 3**: Verify the route
- Try accessing http://localhost:8000/ directly
- Check browser console for errors

### Issue: API returns 404

**Check**: Ensure API routes use the correct prefix

All API routes should be under `/api/v1/`:
- ✓ `/api/v1/formulas`
- ✗ `/formulas` (will be caught by static files)

### Issue: Docker build fails on pip install

This can happen due to SSL certificate issues. The Dockerfile includes `--trusted-host` flags:

```dockerfile
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

If issues persist, try building without BuildKit:
```bash
DOCKER_BUILDKIT=0 docker build -f Dockerfile.unified -t blank-app:unified .
```

### Issue: SPA routes return 404

This is expected behavior. The React app uses client-side routing:

1. Initial page load at `/` serves `index.html`
2. React Router loads and handles all routes client-side
3. User navigation doesn't hit the server
4. Browser refresh at `/dashboard` will return 404

To fix browser refresh on SPA routes, you would need to add a catch-all route handler. However, for most deployments, users navigate through the app rather than directly accessing deep routes.

## Production Considerations

### 1. Environment Variables

Never commit secrets! Use environment variables:
- Set them in Render dashboard
- Use `.env` files locally (git-ignored)

### 2. Database Migrations

Run migrations before starting the app:
```bash
alembic upgrade head
```

For Docker, you can add this to an entrypoint script.

### 3. Static File Caching

The built frontend assets have content hashes in filenames (e.g., `index-D5bjsvh5.js`), enabling aggressive caching:

```
Cache-Control: public, max-age=31536000, immutable
```

FastAPI's StaticFiles middleware handles this automatically.

### 4. Health Checks

Render uses `/health` for health checks. The endpoint:
- Checks database connectivity
- Checks Redis (if configured)
- Returns 200 if healthy, 503 if degraded

### 5. Monitoring

- Prometheus metrics: `/metrics`
- Application logs: `docker logs <container>`
- Render logs: Available in Render dashboard

## Migrating from Separate Deployments

If you previously deployed frontend and backend separately:

1. **Delete the Static Site** on Render (frontend)
2. **Update the Web Service** to use `Dockerfile.unified`
3. **Remove CORS restrictions** if the frontend and backend were on different domains
4. **Update environment variables** in the Web Service
5. **Deploy** the unified application

The backend's CORS middleware is already configured to accept all origins during development. For production, tighten this in `backend/app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## File Reference

- **`Dockerfile.unified`** - Multi-stage Docker build configuration
- **`.dockerignore`** - Files excluded from Docker build context
- **`backend/app/main.py`** - FastAPI app with frontend serving logic
- **`frontend/vite.config.ts`** - Frontend build configuration
- **`frontend/package.json`** - Frontend dependencies and build scripts

## Additional Resources

- [FastAPI Static Files Documentation](https://fastapi.tiangolo.com/tutorial/static-files/)
- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Render Docker Deployment](https://render.com/docs/docker)
