# Quick Deployment Checklist - UI + Backend

## ✅ Pre-Deployment Checklist

### Code Preparation
- [x] Frontend built successfully (`frontend/dist` exists)
- [x] Backend modified to serve frontend
- [x] All tests passing (20/20)
- [x] API endpoints verified working
- [x] Health check endpoint working

### Configuration Files Ready
- [x] `render-unified.yaml` - Unified deployment config
- [x] `render.yaml` - Separate deployment config (alternative)
- [x] `DEPLOYMENT_UI_BACKEND.md` - Deployment guide
- [x] `.gitignore` - Excludes build artifacts

## 🚀 Deployment Steps

### For Unified Deployment (Recommended)

1. **Push Code to GitHub**
   ```bash
   git push origin main  # or your branch
   ```

2. **Deploy on Render.com**
   - Option A: Use Blueprint
     - Go to https://dashboard.render.com
     - Click "New +" → "Blueprint"
     - Select `render-unified.yaml`
     - Click "Apply"
   
   - Option B: Manual Web Service
     - Click "New +" → "Web Service"
     - Connect GitHub repo
     - Use build command from `render-unified.yaml`
     - Set environment variables
     - Deploy

3. **Set Environment Variables**
   - `DATABASE_URL` (from database)
   - `SECRET_KEY` (auto-generate)
   - `CORS_ORIGINS=*`
   - `API_KEY_ENABLED=false`

4. **Wait for Deployment** (~5-10 minutes)

5. **Verify Deployment**
   - Health: `https://YOUR-SERVICE.onrender.com/health`
   - UI: `https://YOUR-SERVICE.onrender.com/`
   - API Docs: `https://YOUR-SERVICE.onrender.com/docs`

## 🧪 Post-Deployment Testing

### 1. Health Check
```bash
curl https://YOUR-SERVICE.onrender.com/health
```
Expected: `{"status":"healthy","version":"1.0.0",...}`

### 2. UI Access
Open in browser: `https://YOUR-SERVICE.onrender.com/`
Expected: React login page

### 3. API Documentation
Open: `https://YOUR-SERVICE.onrender.com/docs`
Expected: Swagger UI with API endpoints

### 4. API Test
```bash
curl -H "X-API-Key: test-api-key" \
     https://YOUR-SERVICE.onrender.com/api/v1/formulas/list
```
Expected: JSON array of formulas

### 5. UI Navigation
- [ ] Login page loads
- [ ] Can navigate to dashboard
- [ ] Can access formula catalog
- [ ] Can execute formulas
- [ ] Admin panel accessible (if admin user)

## 📋 Current Status

### ✅ Completed
- Backend serves frontend at root `/`
- Frontend built and integrated
- All tests passing
- Deployment configurations ready
- Documentation complete

### 📝 URLs (After Deployment)

**If using unified deployment:**
- Main URL: `https://ml-platform-unified.onrender.com`
- UI: `https://ml-platform-unified.onrender.com/`
- API: `https://ml-platform-unified.onrender.com/api/v1/`
- Docs: `https://ml-platform-unified.onrender.com/docs`

**If using separate deployments:**
- Frontend: `https://ml-platform-frontend.onrender.com`
- Backend: `https://ml-platform-backend.onrender.com`

## 🔧 Troubleshooting

### Issue: Still seeing JSON at root

**Solution:**
1. Check Render build logs for frontend build errors
2. Verify `frontend/dist` directory was created during build
3. Rebuild with correct `VITE_API_URL`

### Issue: White screen

**Solution:**
1. Check browser console for errors
2. Verify API URL in frontend env vars
3. Check CORS settings

### Issue: API not responding

**Solution:**
1. Check `/health` endpoint
2. Review Render service logs
3. Verify environment variables

## 📦 What's Deployed

### Backend (Python/FastAPI)
- Formula execution engine
- API endpoints at `/api/v1/*`
- Swagger docs at `/docs`
- Health check at `/health`
- Serves frontend at `/`

### Frontend (React/TypeScript)
- Login page
- Dashboard
- Formula Catalog
- Formula Execution Interface
- Admin Panel
- Auditor Dashboard

## 🎯 Success Criteria

- [x] Code ready for deployment
- [ ] Service deployed on Render
- [ ] UI loads in browser
- [ ] UI connects to backend
- [ ] Users can login
- [ ] Users can execute formulas
- [ ] Admin features work
- [ ] All pages navigate correctly

## 📞 Next Steps

1. **Deploy** using steps above
2. **Test** all functionality
3. **Share** deployment URL
4. **Monitor** for any issues
5. **Iterate** based on feedback

## 🔗 Quick Links

- **Deployment Guide**: `DEPLOYMENT_UI_BACKEND.md`
- **API Guide**: `API_STANDALONE_GUIDE.md`
- **README**: `README.md`
- **Render Dashboard**: https://dashboard.render.com

---

## 💡 Pro Tips

1. **Free Tier Limitations**
   - Service sleeps after 15 min inactivity
   - First request after sleep takes ~30 seconds
   - Use "starter" plan ($7/mo) for always-on

2. **Custom Domain**
   - Can add custom domain in Render settings
   - Free SSL included

3. **Monitoring**
   - Check service logs in Render dashboard
   - Use `/health` endpoint for uptime monitoring
   - Consider adding error tracking (Sentry, etc.)

4. **Performance**
   - Frontend is cached by browser
   - API responses are fast
   - Database queries optimized

---

**Ready to deploy? Follow the steps above!** 🚀
