# Deployment Guide - UI + Backend Integration

## Overview

This guide explains how to deploy the Formula Execution API with the React UI. There are two deployment options:

1. **Unified Deployment** (Recommended) - Backend serves both API and UI from single URL
2. **Separate Deployments** - Backend and Frontend as separate services

## Option 1: Unified Deployment (Recommended)

### Advantages
- ✅ Single URL for both UI and API
- ✅ Simpler configuration
- ✅ No CORS issues
- ✅ Lower cost (1 service instead of 2)
- ✅ Easier to manage

### How It Works
1. Frontend is built during backend deployment
2. Backend serves frontend static files at root `/`
3. API endpoints available at `/api/v1/*`
4. API docs available at `/docs`

### Deploy to Render.com

#### Using render-unified.yaml

1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New +" → "Blueprint"
4. Select `render-unified.yaml`
5. Click "Apply"

#### Manual Setup

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `ml-platform-unified`
   - **Environment**: `Python`
   - **Build Command**:
     ```bash
     pip install --upgrade pip &&
     pip install -r backend/requirements.txt &&
     cd frontend &&
     npm install &&
     VITE_API_URL=https://YOUR-SERVICE-NAME.onrender.com npm run build &&
     cd ..
     ```
   - **Start Command**:
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Health Check Path**: `/health`

5. Add Environment Variables (see below)
6. Click "Create Web Service"

### Local Testing

```bash
# Build frontend
cd frontend
VITE_API_URL=http://localhost:8000 npm run build
cd ..

# Start backend (serves both API and UI)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Access:
# - UI: http://localhost:8000/
# - API Docs: http://localhost:8000/docs
# - Health: http://localhost:8000/health
```

## Option 2: Separate Deployments

### Advantages
- ✅ Independent scaling
- ✅ Separate deployments
- ✅ Can use CDN for frontend

### Deploy Backend

Use original `render.yaml` configuration:

```yaml
- type: web
  name: ml-platform-backend
  env: python
  buildCommand: "cd backend && pip install -r requirements.txt"
  startCommand: "cd backend && chmod +x start.sh && ./start.sh"
```

### Deploy Frontend

Use static site hosting:

```yaml
- type: web
  name: ml-platform-frontend
  env: static
  buildCommand: "cd frontend && npm install && npm run build"
  staticPublishPath: frontend/dist
  envVars:
    - key: VITE_API_URL
      value: https://ml-platform-backend.onrender.com
```

### Access Points
- Frontend: `https://ml-platform-frontend.onrender.com`
- Backend API: `https://ml-platform-backend.onrender.com/api/v1/*`

## Environment Variables

### Required
```
DATABASE_URL       # PostgreSQL connection string (from database)
SECRET_KEY         # JWT secret (auto-generated)
```

### API Configuration
```
API_V1_PREFIX      # /api/v1
CORS_ORIGINS       # * (or specific origins)
API_KEY_ENABLED    # false (for simpler setup)
```

### Optional Features
```
GOOGLE_OAUTH_CLIENT_ID         # For Google Drive integration
GOOGLE_OAUTH_REDIRECT_URI      # OAuth callback URL
OPENAI_API_KEY                 # For AI features
MLFLOW_TRACKING_URI            # For experiment tracking
```

## Verification

After deployment, verify:

### 1. Health Check
```bash
curl https://YOUR-SERVICE.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production"
}
```

### 2. UI Access
Open browser: `https://YOUR-SERVICE.onrender.com/`

You should see the React UI login page.

### 3. API Access
```bash
curl https://YOUR-SERVICE.onrender.com/docs
```

Should return Swagger UI HTML.

### 4. Test API Endpoint
```bash
curl -H "X-API-Key: test-api-key" \
     https://YOUR-SERVICE.onrender.com/api/v1/formulas/list
```

## Troubleshooting

### Issue: Getting JSON instead of UI

**Cause**: Frontend not built or not found

**Solution**:
1. Check build logs for frontend build errors
2. Verify `frontend/dist` directory exists after build
3. Ensure `VITE_API_URL` is set correctly during build

### Issue: White screen / React errors

**Cause**: API URL not configured correctly

**Solution**:
1. Check browser console for errors
2. Verify `VITE_API_URL` matches your backend URL
3. Rebuild frontend with correct API URL

### Issue: CORS errors

**Cause**: CORS not configured for your frontend URL

**Solution**:
1. Set `CORS_ORIGINS=*` in backend environment variables
2. Or set specific origins: `CORS_ORIGINS=https://your-frontend.com`

### Issue: 404 on page refresh

**Cause**: SPA routing not configured

**Solution**:
This is handled automatically in unified deployment. For separate deployments, ensure your static host has rewrite rules.

## Architecture Diagrams

### Unified Deployment
```
┌─────────────────────────────────────────┐
│     https://ml-platform.onrender.com     │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │      FastAPI Backend (Python)      │ │
│  │                                    │ │
│  │  • Serves React UI at /           │ │
│  │  • API at /api/v1/*               │ │
│  │  • Docs at /docs                  │ │
│  │  • Health at /health              │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Separate Deployment
```
┌──────────────────────────┐     ┌──────────────────────────┐
│  Frontend (Static Site)   │────▶│   Backend (API Server)   │
│                          │     │                          │
│  ml-platform-frontend    │     │  ml-platform-backend     │
│  .onrender.com           │     │  .onrender.com           │
│                          │     │                          │
│  • React UI              │     │  • API endpoints         │
│  • Client-side routing   │     │  • Database access       │
└──────────────────────────┘     └──────────────────────────┘
```

## Cost Optimization

### Free Tier (Render)
- 1 unified service: **Free**
- Includes: 750 hours/month, automatic SSL

### Paid Tier ($7/month per service)
- Better for production
- No sleep after inactivity
- Custom domains
- More resources

## Next Steps

1. **Deploy**: Follow unified deployment guide above
2. **Test**: Verify UI and API work correctly
3. **Configure**: Add environment variables for features you need
4. **Customize**: Update branding, colors in frontend
5. **Monitor**: Check logs and health endpoint

## Support

For issues:
1. Check deployment logs in Render dashboard
2. Test locally first with instructions above
3. Verify environment variables are set
4. Check browser console for frontend errors
5. Use `/health` endpoint to verify backend is running

## Additional Resources

- **API Documentation**: `/docs` on your deployed URL
- **Standalone API Guide**: `API_STANDALONE_GUIDE.md`
- **Original README**: `README.md`
- **Render Docs**: https://render.com/docs
