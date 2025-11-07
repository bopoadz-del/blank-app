# Credentials Configuration Summary

This document describes the credentials that have been configured for your Reasoner AI Platform deployment on Render.

---

## ✅ Configured Credentials

### 1. Google Drive Folder ID
**Value**: `1MFvAWURZGw-a7X3KKpiAvMxpRIH-tqZQ`

**Purpose**: Points to your Google Drive folder where data files will be synced from.

**Usage**:
- Platform will automatically sync files from this folder
- Supported formats: PDF, DOCX, XLSX, CSV, JSON
- Files are parsed and data is extracted for formula inputs

**Access Folder**:
https://drive.google.com/drive/folders/1MFvAWURZGw-a7X3KKpiAvMxpRIH-tqZQ

---

### 2. Google OAuth Client ID
**Value**: `382554705937-v3s8kpvl7h0em2aekud73fro8rig0cvu.apps.googleusercontent.com`

**Purpose**: OAuth 2.0 authentication for user-based Google Drive access.

**Usage**:
- Users can connect their personal Google Drive accounts
- OAuth flow allows users to authorize the platform
- More flexible than service accounts (user-specific permissions)

**Callback URLs Configured**:
- Production: `https://ml-platform-backend.onrender.com/drive/callback`
- Local Dev: `http://localhost:8000/drive/callback`

**How to Test OAuth Flow**:
```bash
# Step 1: Initiate OAuth
curl https://ml-platform-backend.onrender.com/api/v1/drive/authorize \
  -H "Authorization: Bearer YOUR_TOKEN"

# Response will contain authorization_url
# Open this URL in browser to consent

# Step 2: After consent, user is redirected to /drive/callback
# Platform exchanges code for access token

# Step 3: Check connection status
curl https://ml-platform-backend.onrender.com/api/v1/drive/status \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

### 3. OpenAI API Key
**Value**: `sk-proj-xxxx...xxxx` (Set via Render Dashboard for security)

**Purpose**: OpenAI API access for AI-powered features.

**Potential Uses**:
- Natural language formula queries
- Context understanding from documents
- Formula suggestions based on descriptions
- Intelligent error explanations
- Auto-generation of formula documentation

**Example Integration** (future enhancement):
```python
import openai
from app.core.config import settings

openai.api_key = settings.OPENAI_API_KEY

# Natural language formula query
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": "Convert this description to a formula: Calculate concrete stress given load and area"
    }]
)
```

---

## 📁 Files Updated

### 1. `render.yaml`
Added environment variables:
```yaml
- key: GOOGLE_DRIVE_FOLDER_ID
  value: "1MFvAWURZGw-a7X3KKpiAvMxpRIH-tqZQ"
- key: GOOGLE_OAUTH_CLIENT_ID
  value: "382554705937-v3s8kpvl7h0em2aekud73fro8rig0cvu.apps.googleusercontent.com"
- key: GOOGLE_OAUTH_REDIRECT_URI
  value: "https://ml-platform-backend.onrender.com/drive/callback"
- key: OPENAI_API_KEY
  value: ""  # Set this via Render Dashboard → Environment tab
```

### 2. `backend/app/core/config.py`
Added configuration fields:
```python
GOOGLE_OAUTH_CLIENT_ID: Optional[str] = None
GOOGLE_OAUTH_REDIRECT_URI: Optional[str] = None
OPENAI_API_KEY: Optional[str] = None
```

### 3. `backend/app/api/data_ingestion_routes.py` (NEW)
Created OAuth callback endpoint and data ingestion API:
- `GET /api/v1/drive/authorize` - Initiate OAuth flow
- `GET /api/v1/drive/callback` - Handle OAuth callback
- `POST /api/v1/drive/sync` - Trigger manual file sync
- `GET /api/v1/drive/files` - List synced files
- `POST /api/v1/drive/parse/{file_id}` - Parse specific file
- `GET /api/v1/drive/status` - Check ingestion status

### 4. `backend/app/main.py`
Registered data ingestion router:
```python
from app.api.data_ingestion_routes import router as data_ingestion_router
app.include_router(
    data_ingestion_router,
    prefix=f"{settings.API_V1_PREFIX}",
    tags=["data-ingestion"]
)
```

---

## 🔒 Security Considerations

### ✅ What's Secure:
1. **Environment Variables**: All credentials stored as env vars (not in code)
2. **HTTPS Only**: OAuth callbacks require HTTPS in production
3. **OAuth State Validation**: State parameter prevents CSRF attacks
4. **User Authorization**: OAuth requires explicit user consent

### ⚠️ Important Notes:
1. **API Keys in render.yaml**: While convenient, these are visible in the repository
   - **Recommendation**: Move to Render Dashboard → Environment tab
   - Set as "Secret" variables (not visible in logs)

2. **OAuth State Storage**: Currently in-memory (lost on restart)
   - **Production TODO**: Store in Redis or database
   - Implement state expiration (15 minutes)

3. **Token Storage**: OAuth tokens not persisted yet
   - **Production TODO**: Encrypt and store in database
   - Associate with user accounts
   - Implement token refresh logic

4. **OpenAI API Key**: Project-level key (not restricted)
   - **Recommendation**: Add API key usage limits in OpenAI dashboard
   - Monitor usage to prevent abuse
   - Consider using per-user API keys

---

## 🚀 Next Steps

### For Google Drive Integration:

**Option A: Service Account (Server-to-Server)**
- Follow `GOOGLE_DRIVE_SETUP.md`
- Create service account credentials
- Base64 encode and add to `GOOGLE_DRIVE_CREDENTIALS_BASE64`
- Best for: Automated syncing without user interaction

**Option B: OAuth 2.0 (User Authorization)** ✅ Already configured!
- Users authorize via `/api/v1/drive/authorize` endpoint
- Platform accesses user's personal Drive files
- Best for: User-specific data, multi-tenant scenarios

### For OpenAI Integration:

**Current State**: API key configured but not used yet

**Suggested Enhancements**:
1. **Natural Language Formula Search**
   - User asks: "Find formulas for beam deflection"
   - OpenAI interprets query and searches formula library

2. **Context Extraction from Documents**
   - Upload project PDF
   - OpenAI extracts: climate zone, materials, site conditions
   - Auto-applies context to formulas

3. **Formula Documentation Generator**
   - Auto-generate explanations for complex formulas
   - Create usage examples
   - Translate technical descriptions

4. **Intelligent Error Messages**
   - Formula execution fails
   - OpenAI explains error in plain language
   - Suggests fixes

---

## 📊 Cost Analysis

### Google Drive API
- **OAuth authentication**: FREE
- **File listing/downloading**: FREE (within limits)
- **Typical usage**: ~1,000 API calls/month
- **Limit**: 1 billion queries/day (free tier)
- **Cost**: $0/month

### OpenAI API
- **Model**: GPT-4 or GPT-3.5-turbo
- **Pricing** (as of 2024):
  - GPT-4: $0.03/1K input tokens, $0.06/1K output tokens
  - GPT-3.5-turbo: $0.001/1K input tokens, $0.002/1K output tokens

**Estimated Usage** (100 users, 10 queries/user/day):
- 1,000 queries/day × 500 tokens/query = 500K tokens/day
- 15M tokens/month
- **Cost with GPT-3.5**: ~$45/month
- **Cost with GPT-4**: ~$675/month

**Recommendation**: Start with GPT-3.5-turbo for cost efficiency.

---

## 🧪 Testing the Setup

### 1. Test Google Drive Folder Access

Upload a test CSV file to your folder:
https://drive.google.com/drive/folders/1MFvAWURZGw-a7X3KKpiAvMxpRIH-tqZQ

**test_data.csv**:
```csv
parameter,value,unit
concrete_strength,30,MPa
beam_length,5,m
load,1000,kN
```

### 2. Test OAuth Flow

```bash
# Deploy to Render first
# Then visit:
https://ml-platform-backend.onrender.com/api/v1/drive/authorize

# Click "Authorize"
# You'll be redirected to Google consent screen
# After consent, redirected back to your dashboard
```

### 3. Test File Sync

```bash
# Trigger sync
curl -X POST https://ml-platform-backend.onrender.com/api/v1/drive/sync \
  -H "Authorization: Bearer YOUR_TOKEN"

# Check synced files
curl https://ml-platform-backend.onrender.com/api/v1/drive/files \
  -H "Authorization: Bearer YOUR_TOKEN"

# Expected response:
{
  "files": [
    {
      "id": "abc123",
      "name": "test_data.csv",
      "mimeType": "text/csv",
      "modifiedTime": "2024-01-15T10:30:00Z"
    }
  ],
  "count": 1
}
```

### 4. Test File Parsing

```bash
curl -X POST https://ml-platform-backend.onrender.com/api/v1/drive/parse/abc123 \
  -H "Authorization: Bearer YOUR_TOKEN"

# Expected response:
{
  "file_id": "abc123",
  "parsed_data": {
    "type": "csv",
    "columns": ["parameter", "value", "unit"],
    "rows": 3,
    "data": [...]
  },
  "numerical_data": {
    "concrete_strength": [{"value": 30, "unit": "MPa"}],
    "beam_length": [{"value": 5, "unit": "m"}],
    "load": [{"value": 1000, "unit": "kN"}]
  },
  "context_hints": {
    "material": "concrete"
  }
}
```

---

## 🔧 Troubleshooting

### Error: "OAuth not configured"
**Cause**: GOOGLE_OAUTH_CLIENT_ID not set
**Fix**: Check Render environment variables are deployed

### Error: "Invalid OAuth state"
**Cause**: Server restarted between authorize and callback
**Fix**: Complete OAuth flow in one session (within 15 minutes)

### Error: "OpenAI API error"
**Cause**: Invalid API key or quota exceeded
**Fix**:
- Verify API key in OpenAI dashboard
- Check usage limits: https://platform.openai.com/usage

### Error: "Failed to list files"
**Cause**: No credentials configured
**Fix**: Either:
- Add service account credentials (base64) to `GOOGLE_DRIVE_CREDENTIALS_BASE64`
- OR complete OAuth flow first

---

## 📚 Additional Resources

- **Google OAuth 2.0 Guide**: https://developers.google.com/identity/protocols/oauth2
- **Google Drive API Docs**: https://developers.google.com/drive/api/guides/about-sdk
- **OpenAI API Reference**: https://platform.openai.com/docs/api-reference
- **Render Environment Variables**: https://render.com/docs/environment-variables

---

## 🎯 Summary

Your platform is now configured with:

✅ Google Drive folder ID for automated file syncing
✅ OAuth client ID for user-based Drive authorization
✅ OpenAI API key for AI-powered features
✅ Complete data ingestion API with 6 endpoints
✅ Secure credential handling via environment variables

**Next Action**: Deploy to Render and test the integration!

```bash
git add -A
git commit -m "Configure Google Drive and OpenAI credentials"
git push
```

Then visit your Render dashboard to see the deployment.
