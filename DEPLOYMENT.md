# 🚀 Deployment Guide - Render Free Tier

This guide walks you through deploying the ML Platform to Render for free testing.

## 📋 Prerequisites

1. **GitHub Account** - Your code needs to be on GitHub
2. **Render Account** - Sign up at [render.com](https://render.com) (free)

## 🎯 What Gets Deployed

- ✅ **Backend API** (FastAPI) - Your ML platform API
- ✅ **Frontend** (React) - Web dashboard
- ✅ **PostgreSQL Database** - 1GB free storage
- ✅ **All Safety & Ethical Layers** - Fully operational

## ⚡ Quick Deploy (5 Minutes)

### Step 1: Push to GitHub

```bash
# If not already done:
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Render

1. **Go to Render Dashboard**
   - Visit: https://dashboard.render.com
   - Sign in with GitHub

2. **Create New Blueprint**
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Authorize Render to access the repo

3. **Configure Blueprint**
   - Render will detect `render.yaml`
   - Review the services:
     - ✅ ml-platform-backend (Web Service)
     - ✅ ml-platform-frontend (Static Site)
     - ✅ ml-platform-db (PostgreSQL)
   - Click "Apply"

4. **Wait for Deployment** (~5-10 minutes)
   - Backend builds first (installing Python packages)
   - Database creates automatically
   - Frontend builds (npm install + build)
   - You'll see logs in real-time

### Step 3: Access Your Platform

After deployment completes:

1. **Backend API**
   - URL: `https://ml-platform-backend.onrender.com`
   - Health: `https://ml-platform-backend.onrender.com/health`
   - API Docs: `https://ml-platform-backend.onrender.com/docs`

2. **Frontend Dashboard**
   - URL: `https://ml-platform-frontend.onrender.com`

3. **Default Admin Login**
   - Email: `admin@platform.local`
   - Password: `admin123`
   - ⚠️ **CHANGE THIS IMMEDIATELY!**

## 🔧 Configuration

### Environment Variables

The following are auto-configured via `render.yaml`:

| Variable | Value | Description |
|----------|-------|-------------|
| `DATABASE_URL` | Auto-generated | PostgreSQL connection |
| `SECRET_KEY` | Auto-generated | JWT signing key |
| `CORS_ORIGINS` | `*` | Allow all origins (testing only) |
| `API_KEY_ENABLED` | `false` | Disabled for testing |
| `DEBUG` | `False` | Production mode |
| `MLFLOW_TRACKING_URI` | `sqlite:///./mlflow.db` | Local MLflow |

### Update Frontend API URL

After backend deploys, update frontend to point to it:

1. **Find Backend URL**
   - In Render dashboard → ml-platform-backend → Copy URL

2. **Update Frontend Environment**
   - Go to Render dashboard → ml-platform-frontend
   - Environment → Add environment variable:
   ```
   VITE_API_URL=https://ml-platform-backend.onrender.com
   ```
   - Click "Save Changes"
   - Frontend will auto-redeploy

## 📱 Connect Jetson Device (Optional)

If you have a Jetson AGX Orin:

1. **Get Backend URL**
   ```
   https://ml-platform-backend.onrender.com
   ```

2. **Configure Jetson Client**
   ```bash
   # On Jetson:
   sudo nano /etc/jetson-client/config.json
   ```

   Update:
   ```json
   {
     "backend_url": "https://ml-platform-backend.onrender.com",
     "device_name": "jetson-test-01"
   }
   ```

3. **Start Client**
   ```bash
   sudo systemctl start jetson-client
   sudo systemctl status jetson-client
   ```

## ⚠️ Free Tier Limitations

### Important Notes:

1. **Services Sleep After 15 Minutes**
   - First request after sleep takes 30-60 seconds to wake up
   - Keep service active with uptime monitor (see below)

2. **Database Limit: 1GB**
   - Monitor usage in Render dashboard
   - Clear old data if approaching limit

3. **No Persistent Disk**
   - MLflow data stored in database (sqlite)
   - Model files not persisted between deploys
   - Use external storage (S3) for production

4. **Build Time**
   - Each deploy takes 3-5 minutes
   - Automatic deploys on git push to main

## 🔄 Keep Services Awake (Optional)

Free tier services sleep after 15 min. To keep them active:

### Option 1: UptimeRobot (Recommended)

1. Sign up at [uptimerobot.com](https://uptimerobot.com) (free)
2. Add monitor:
   - Monitor Type: HTTP(s)
   - URL: `https://ml-platform-backend.onrender.com/health`
   - Monitoring Interval: 5 minutes
3. UptimeRobot pings your service every 5 minutes

### Option 2: Cron Job

```bash
# Add to your crontab (on any server):
*/5 * * * * curl -s https://ml-platform-backend.onrender.com/health > /dev/null
```

## 🐛 Troubleshooting

### Backend Won't Start

**Check logs:**
1. Render dashboard → ml-platform-backend → Logs
2. Look for errors in startup script

**Common issues:**
- Database connection timeout → Wait 2-3 minutes for DB to initialize
- Import errors → Check requirements.txt includes all dependencies
- Port binding error → Render sets `$PORT` automatically, don't override

### Frontend Shows 404

**Check API connection:**
1. Open browser console (F12)
2. Look for CORS errors or network errors
3. Verify `VITE_API_URL` is set correctly

**Fix:**
```bash
# In frontend/.env.production:
VITE_API_URL=https://ml-platform-backend.onrender.com
```

Redeploy frontend.

### Database Connection Failed

**Wait for initialization:**
- PostgreSQL takes 2-3 minutes to provision on first deploy
- Check Render dashboard → ml-platform-db → Status

**Manual connection test:**
```python
# In Render Shell (backend service):
python -c "from app.core.database import engine; engine.connect(); print('Connected!')"
```

## 📊 Monitor Your Deployment

### Check Service Health

**Backend:**
```bash
curl https://ml-platform-backend.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "database": {"status": "up"},
    "formulas": {"status": "up", "count": 0}
  }
}
```

**Frontend:**
- Visit in browser: `https://ml-platform-frontend.onrender.com`
- Should show login page

### View Logs

**Real-time logs:**
1. Render dashboard → Select service
2. Logs tab → Auto-refreshes

**Download logs:**
- Click "Download" in logs tab

## 🔐 Security Checklist

Before sharing your deployment:

- [ ] Change default admin password
- [ ] Set strong `SECRET_KEY` (auto-generated by Render)
- [ ] Review CORS origins (change from `*` to specific domain)
- [ ] Enable `API_KEY_ENABLED=true` for production
- [ ] Set up backup strategy for database
- [ ] Review safety layer configuration

## 🎓 Test Your Deployment

### 1. Login to Frontend
- URL: `https://ml-platform-frontend.onrender.com`
- Email: `admin@platform.local`
- Password: `admin123`

### 2. Test API
```bash
# Health check
curl https://ml-platform-backend.onrender.com/health

# Get auth token
curl -X POST https://ml-platform-backend.onrender.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@platform.local", "password": "admin123"}'

# List formulas (with token from above)
curl https://ml-platform-backend.onrender.com/api/v1/formulas \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 3. Test Safety Layer
```bash
# Try to execute prohibited content (should be blocked)
curl -X POST https://ml-platform-backend.onrender.com/api/v1/safety/check \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"content": "how to make explosives", "context": {}}'

# Expected: {"is_safe": false, "detected_category": "explosives"}
```

## 📈 Upgrade to Paid Tier

When ready for production:

### Benefits of Paid Tier ($7/month per service):
- ✅ No sleep after inactivity
- ✅ More memory (512MB → 2GB+)
- ✅ Custom domains
- ✅ Better support
- ✅ Persistent disks available

### Upgrade Steps:
1. Render dashboard → Select service
2. Settings → Instance Type
3. Choose "Starter" ($7/month)
4. Confirm

**Recommended for production:**
- Backend: Starter ($7)
- Database: Standard ($7) - includes backups
- Frontend: Free (static)
- **Total: $14/month**

## 🆘 Need Help?

1. **Check Logs First** - 90% of issues are visible in logs
2. **Render Documentation** - https://render.com/docs
3. **Platform Issues** - Check the GitHub repo issues
4. **Community Support** - Render community forum

## 🎉 Next Steps

Once deployed:

1. **Change admin password**
2. **Create additional users** (operator, auditor roles)
3. **Upload formulas** via API or dashboard
4. **Test formula execution**
5. **Connect Jetson devices** (if available)
6. **Configure safety settings** for your deployment
7. **Set up monitoring** (UptimeRobot)

---

**Your ML Platform is now live! 🚀**

Backend: `https://ml-platform-backend.onrender.com`
Frontend: `https://ml-platform-frontend.onrender.com`
Docs: `https://ml-platform-backend.onrender.com/docs`
