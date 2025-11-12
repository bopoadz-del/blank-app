# Quick Start: Unified Deployment

## TL;DR

Single Docker image serves both React frontend and FastAPI backend.

```bash
# Build
docker build -f Dockerfile.unified -t blank-app:unified .

# Run
docker run -p 8000:8000 blank-app:unified

# Access
open http://localhost:8000
```

## What Changed?

### Before (Separate Deployments)
```
┌─────────────┐         ┌──────────────┐
│  Frontend   │ ──────> │   Backend    │
│  (Render)   │  API    │   (Render)   │
│  Port 3000  │  calls  │   Port 8000  │
└─────────────┘         └──────────────┘
```

### After (Unified Deployment)
```
┌────────────────────────────────┐
│     Single Container           │
│  ┌──────────┐  ┌────────────┐ │
│  │ Frontend │  │  Backend   │ │
│  │  Static  │  │  FastAPI   │ │
│  └──────────┘  └────────────┘ │
│        Port 8000               │
└────────────────────────────────┘
```

## Routes

| Path | Serves | Example |
|------|--------|---------|
| `/` | React App | `index.html` |
| `/assets/*` | Static Files | `main.js`, `styles.css` |
| `/api/v1/*` | API Endpoints | `/api/v1/formulas` |
| `/health` | Health Check | `{"status": "healthy"}` |
| `/docs` | Swagger UI | API Documentation |

## Development

### Option 1: Separate (Fast iteration)
```bash
# Terminal 1 - Backend
cd backend
uvicorn app.main:app --reload

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### Option 2: Unified (Production-like)
```bash
docker build -f Dockerfile.unified -t app .
docker run -p 8000:8000 app
```

## Deployment (Render)

1. **Dockerfile**: `Dockerfile.unified`
2. **Port**: Auto (uses `$PORT` env var)
3. **Health Check**: `/health`
4. **Environment Variables**: Set in Render dashboard

## Files

- `Dockerfile.unified` - Build configuration
- `backend/app/main.py` - Includes frontend mounting
- `.dockerignore` - Optimizes build
- `UNIFIED_DEPLOYMENT.md` - Full documentation

## Common Commands

```bash
# Build image
docker build -f Dockerfile.unified -t app .

# Run with custom port
docker run -p 3000:3000 -e PORT=3000 app

# Run with environment file
docker run -p 8000:8000 --env-file .env app

# Shell into running container
docker exec -it <container-id> sh

# View logs
docker logs <container-id>

# Stop container
docker stop <container-id>
```

## Troubleshooting

**Frontend not loading?**
- Check: `docker logs <container-id>` for "Frontend static files mounted"
- Verify: Frontend was built: `ls frontend/dist/`

**API returns 404?**
- Ensure routes use `/api/v1/` prefix
- Check: Visit `/docs` to see all routes

**Docker build fails?**
- SSL issue: Already handled with `--trusted-host` flags
- Network issue: Check internet connection

## Next Steps

1. Review `UNIFIED_DEPLOYMENT.md` for details
2. Test locally with Docker
3. Deploy to Render
4. Monitor with `/health` and `/metrics`
