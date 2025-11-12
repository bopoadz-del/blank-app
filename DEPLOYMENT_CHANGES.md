# Unified Deployment - Quick Reference

## Summary of Changes

This PR implements a unified deployment where the frontend (React) and backend (FastAPI) are built together and served from a single Docker container.

## Files Added/Modified

### New Files
- ✅ **Dockerfile** (root) - Multi-stage build for frontend + backend
- ✅ **.dockerignore** (root) - Optimizes Docker build context
- ✅ **UNIFIED_DEPLOYMENT.md** - Complete deployment guide
- ✅ **backend/tests/test_frontend_mounting.py** - Tests for the deployment
- ✅ **test_deployment.sh** - Verification script

### Modified Files
- ✅ **backend/app/main.py** - Added frontend mounting logic
- ✅ **.gitignore** - Added backend/frontend/dist exclusion

## Key Features

### 1. Multi-Stage Dockerfile
```
Stage 1: Build Frontend (Node 18)
  ↓
Stage 2: Build Backend (Python 3.11 + venv)
  ↓
Stage 3: Final Runtime (frontend + backend)
```

### 2. FastAPI Changes
- **Serves frontend at `/`**: SPA-friendly routing with `html=True`
- **Health check at `/health`**: Unchanged, for platform monitoring
- **API routes at `/api/v1`**: All existing API routes work as before

### 3. Path Structure
```
/app/backend/
├── app/
│   └── main.py              # Frontend mounting logic
└── frontend/
    └── dist/                # Built frontend copied here
        ├── index.html
        └── assets/
```

## How to Use

### Local Testing
```bash
# Run the verification script
./test_deployment.sh

# Or manually:
cd frontend && npm run build
mkdir -p ../backend/frontend
cp -r dist ../backend/frontend/dist
cd ../backend
uvicorn app.main:app
```

### Docker Build
```bash
docker build -t blank-unified .
docker run -p 8000:8000 -e PORT=8000 blank-unified
```

### Render Deployment
1. Push to GitHub
2. Create Web Service on Render
3. Point to repository root
4. Render automatically uses Dockerfile
5. Set environment variables

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `PORT` | Server port (Render sets this) | No (defaults to 8000) |
| `FRONTEND_DIST_PATH` | Override frontend location | No |
| `DATABASE_URL` | PostgreSQL connection | Yes |
| `SECRET_KEY` | JWT secret | Yes |

## Request Routing

```
GET /                    → Frontend (index.html)
GET /about               → Frontend (SPA routing)
GET /health              → Backend health check
GET /metrics             → Backend Prometheus metrics
GET /api/v1/formulas     → Backend API
POST /api/v1/formulas    → Backend API
```

## Testing

```bash
cd backend
pytest tests/test_frontend_mounting.py -v
```

All tests verify:
- ✅ Frontend path resolution
- ✅ Dockerfile structure
- ✅ .dockerignore contents
- ✅ Main.py mounting code

## Benefits

1. **Single deployment** - One container instead of two services
2. **No CORS issues** - Same origin for frontend and backend
3. **Simplified infrastructure** - No reverse proxy needed
4. **Cost effective** - One service instead of two on Render
5. **Consistent environments** - Dev and prod work the same way

## Backward Compatibility

- ✅ All existing API routes work unchanged
- ✅ Health checks work as before
- ✅ Metrics endpoint unchanged
- ✅ Database connections unchanged

## Security Features

- ✅ Non-root user in container
- ✅ Multi-stage build (smaller attack surface)
- ✅ .dockerignore prevents sensitive files
- ✅ Virtualenv isolation for Python deps

## Performance

- Frontend assets served directly (no proxy overhead)
- Static file caching enabled
- Gzip compression supported
- Single network hop for API calls

## Troubleshooting

### Frontend not showing?
```bash
# Check if frontend was built
ls backend/frontend/dist/index.html

# Check logs
docker logs <container-id>
# Should see: "Frontend mounted at / from..."
```

### API not working?
- Ensure routes have `/api/v1` prefix
- Check that API routes are registered BEFORE `app.mount()`

### Port issues?
- Use `${PORT:-8000}` syntax in CMD
- Render sets PORT dynamically

## Next Steps

After merging this PR:
1. Update Render service to use new Dockerfile
2. Verify deployment works
3. Update any documentation referencing separate deployments
4. Remove old frontend hosting (if applicable)

## Questions?

See `UNIFIED_DEPLOYMENT.md` for detailed documentation.
