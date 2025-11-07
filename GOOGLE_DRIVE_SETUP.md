# Google Drive Integration Setup Guide

This guide walks you through setting up Google Drive as a data source for the Reasoner AI Platform.

## Overview

The platform can automatically sync and parse files from Google Drive:
- **Supported formats**: PDF, DOCX, XLSX, CSV, JSON
- **Auto-extraction**: Numerical data, context hints, tables
- **Periodic sync**: Configurable interval (default: 1 hour)
- **Use cases**: Formula inputs, training data, validation datasets

---

## Step 1: Create Google Cloud Project

### 1.1 Go to Google Cloud Console
Visit: https://console.cloud.google.com/

### 1.2 Create New Project
1. Click **Select a project** (top left)
2. Click **NEW PROJECT**
3. Enter project name: `reasoner-ai-platform`
4. Click **CREATE**

---

## Step 2: Enable Google Drive API

### 2.1 Navigate to APIs & Services
1. In the Cloud Console, open the navigation menu (☰)
2. Go to **APIs & Services** → **Library**

### 2.2 Enable Drive API
1. Search for `Google Drive API`
2. Click on **Google Drive API**
3. Click **ENABLE**

---

## Step 3: Create Service Account

### 3.1 Navigate to Credentials
1. Go to **APIs & Services** → **Credentials**
2. Click **+ CREATE CREDENTIALS**
3. Select **Service Account**

### 3.2 Configure Service Account
1. **Service account name**: `reasoner-drive-connector`
2. **Service account ID**: (auto-generated)
3. **Description**: `Service account for Reasoner AI Platform to access Google Drive`
4. Click **CREATE AND CONTINUE**

### 3.3 Grant Permissions (Optional)
- Skip this step for now
- Click **CONTINUE** → **DONE**

---

## Step 4: Create Service Account Key

### 4.1 Download Credentials JSON
1. In **Credentials**, find your service account
2. Click on the service account email
3. Go to **KEYS** tab
4. Click **ADD KEY** → **Create new key**
5. Select **JSON** format
6. Click **CREATE**
7. Save the downloaded JSON file securely

**IMPORTANT**: This JSON contains sensitive credentials. Never commit it to git!

### 4.2 JSON Format
Your credentials file looks like this:
```json
{
  "type": "service_account",
  "project_id": "reasoner-ai-platform",
  "private_key_id": "abc123...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "reasoner-drive-connector@reasoner-ai-platform.iam.gserviceaccount.com",
  "client_id": "123456789...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/..."
}
```

---

## Step 5: Share Google Drive Folder

### 5.1 Create or Select Folder
1. Go to Google Drive
2. Create a new folder for platform data: `Reasoner AI Data`
3. Right-click → **Get link**
4. Copy the **Folder ID** from the URL:
   ```
   https://drive.google.com/drive/folders/1aBcDeFgHiJkLmNoPqRsTuVwXyZ123456
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                            This is your FOLDER_ID
   ```

### 5.2 Share with Service Account
1. Right-click the folder → **Share**
2. Add the service account email:
   ```
   reasoner-drive-connector@reasoner-ai-platform.iam.gserviceaccount.com
   ```
3. Set permission to **Viewer** (read-only)
4. Uncheck **Notify people**
5. Click **Share**

---

## Step 6: Configure Render Environment Variables

### 6.1 Prepare Credentials for Render

**Option A: Base64 Encode (Recommended)**

On Linux/Mac:
```bash
cat credentials.json | base64 -w 0 > credentials_base64.txt
```

On Windows (PowerShell):
```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("credentials.json")) > credentials_base64.txt
```

**Option B: JSON String (Escape Quotes)**

Manually escape the JSON:
```json
{\"type\":\"service_account\",\"project_id\":\"reasoner-ai-platform\",...}
```

### 6.2 Add to Render Dashboard

1. Go to Render Dashboard: https://dashboard.render.com
2. Select your **ml-platform-backend** service
3. Go to **Environment** tab
4. Add/update these variables:

| Key | Value | Notes |
|-----|-------|-------|
| `GOOGLE_DRIVE_CREDENTIALS_PATH` | `/tmp/gd_credentials.json` | Path where credentials will be written |
| `GOOGLE_DRIVE_CREDENTIALS_BASE64` | `<your base64 encoded JSON>` | The base64 string from step 6.1 |
| `GOOGLE_DRIVE_FOLDER_ID` | `1aBcDeFgHiJkLmNoPqRsTuVwXyZ123456` | Your folder ID from step 5.1 |
| `DATA_INGESTION_INTERVAL` | `3600` | Sync interval in seconds (1 hour) |

5. Click **Save Changes**

### 6.3 Update render.yaml (Already Done)

The `render.yaml` has been updated with Google Drive environment variables. When you redeploy, these will be set automatically.

---

## Step 7: Decode Credentials on Startup

### 7.1 Update start.sh

Add this to `backend/start.sh` before the server starts:

```bash
#!/bin/bash
set -e

# Decode Google Drive credentials if provided
if [ ! -z "$GOOGLE_DRIVE_CREDENTIALS_BASE64" ]; then
    echo "Setting up Google Drive credentials..."
    echo "$GOOGLE_DRIVE_CREDENTIALS_BASE64" | base64 -d > /tmp/gd_credentials.json
    export GOOGLE_DRIVE_CREDENTIALS_PATH="/tmp/gd_credentials.json"
    echo "Google Drive credentials ready"
fi

# ... rest of start.sh
```

---

## Step 8: Test Integration

### 8.1 Local Testing (Optional)

Create a test script `test_gdrive.py`:

```python
from backend.app.services.data_ingestion import GoogleDriveConnector

connector = GoogleDriveConnector(
    credentials_path="/path/to/credentials.json",
    folder_id="your-folder-id"
)

# List files
files = connector.list_files(limit=10)
print(f"Found {len(files)} files:")
for file in files:
    print(f"  - {file['name']} ({file['mimeType']})")

# Sync folder
synced = connector.sync_folder(local_cache_dir="./drive_cache")
print(f"\nSynced {len(synced)} files to ./drive_cache")
```

Run:
```bash
cd backend
python test_gdrive.py
```

### 8.2 Production Testing

After deployment:

1. Upload test files to your shared Google Drive folder:
   - `test_data.csv` (with numerical columns)
   - `project_specs.pdf` (with technical data)
   - `calculations.xlsx` (with formulas/values)

2. Wait for sync interval (or trigger manually via API)

3. Check logs in Render Dashboard:
   ```
   Setting up Google Drive credentials...
   Google Drive credentials ready
   Google Drive authenticated successfully
   Found 3 files in Google Drive
   Synced: test_data.csv
   Synced: project_specs.pdf
   Synced: calculations.xlsx
   ```

---

## Step 9: Use Data Ingestion API

### 9.1 Trigger Manual Sync

```bash
curl -X POST https://ml-platform-backend.onrender.com/api/v1/data/sync \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 9.2 List Synced Files

```bash
curl https://ml-platform-backend.onrender.com/api/v1/data/files \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 9.3 Parse File and Extract Data

```bash
curl -X POST https://ml-platform-backend.onrender.com/api/v1/data/parse \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"file_id": "abc123xyz"}'
```

---

## Architecture Overview

```
Google Drive Folder
    ├── test_data.csv
    ├── project_specs.pdf
    └── calculations.xlsx
           │
           │ (Service Account Authentication)
           ↓
   GoogleDriveConnector
           │
           │ (Download & Cache)
           ↓
   FileParser (auto-detect type)
           │
           ├─→ parse_csv() → DataFrame
           ├─→ parse_pdf() → Text + Tables
           └─→ parse_excel() → Sheets + Data
                     │
                     ↓
           DataExtractor
                     │
                     ├─→ extract_numerical_data()
                     └─→ extract_context_hints()
                              │
                              ↓
                    Formula Execution
```

---

## Security Best Practices

### ✅ DO:
- Use service accounts (not OAuth user credentials)
- Grant minimum permissions (Viewer only)
- Store credentials as environment variables
- Use Base64 encoding for JSON in env vars
- Rotate credentials every 90 days
- Monitor access logs in Google Cloud Console

### ❌ DON'T:
- Commit credentials to git
- Share service account keys publicly
- Grant Editor/Owner permissions
- Use personal Google accounts
- Store credentials in code
- Expose credentials in logs

---

## Troubleshooting

### Error: "Google Drive authentication failed"

**Cause**: Invalid credentials path or malformed JSON

**Fix**:
```bash
# Verify base64 decoding works
echo "$GOOGLE_DRIVE_CREDENTIALS_BASE64" | base64 -d | jq .
```

### Error: "Insufficient Permission"

**Cause**: Service account doesn't have access to the folder

**Fix**:
- Re-share the folder with service account email
- Ensure permission is at least "Viewer"

### Error: "File not found"

**Cause**: Incorrect folder ID

**Fix**:
- Double-check folder ID from Google Drive URL
- Ensure folder is not in Trash

### Error: "API not enabled"

**Cause**: Google Drive API not enabled for project

**Fix**:
```bash
# Enable via gcloud CLI
gcloud services enable drive.googleapis.com --project=reasoner-ai-platform
```

### No files synced

**Cause**: DATA_INGESTION_INTERVAL not set or too long

**Fix**:
- Set `DATA_INGESTION_INTERVAL=300` (5 minutes for testing)
- Trigger manual sync via API

---

## Cost Analysis

### Google Cloud Free Tier
- **Drive API calls**: 1 billion queries/day free
- **Storage**: Not charged (uses your Drive quota)
- **Service accounts**: Free

### Expected Usage
- **Sync interval**: 1 hour = 24 API calls/day
- **File list query**: ~1 API call per sync
- **Downloads**: 1 API call per new/modified file
- **Monthly total**: ~1,000 API calls (well within free tier)

**Result**: Google Drive integration is FREE for typical usage.

---

## Next Steps

After setup is complete:

1. **Upload Training Data**: Add CSV/Excel files with formula test cases
2. **Auto-Context Detection**: Platform will analyze context hints from files
3. **Formula Validation**: Use extracted data for empirical validation
4. **Continuous Learning**: Platform retrains as new data arrives

---

## API Integration Example

### Automated Workflow

```python
from app.services.data_ingestion import (
    GoogleDriveConnector,
    FileParser,
    DataExtractor
)

# 1. Sync files from Drive
connector = GoogleDriveConnector(
    credentials_path=settings.GOOGLE_DRIVE_CREDENTIALS_PATH,
    folder_id=settings.GOOGLE_DRIVE_FOLDER_ID
)

synced_files = connector.sync_folder(local_cache_dir="/tmp/drive_cache")

# 2. Parse each file
for file_info in synced_files:
    parsed = FileParser.parse_file(file_info['local_path'])

    # 3. Extract numerical data
    numerical_data = DataExtractor.extract_numerical_data(parsed)

    # 4. Extract context hints
    context = DataExtractor.extract_context_hints(parsed)

    # 5. Use data for formula execution
    formula_result = execute_formula_with_context(
        inputs=numerical_data,
        context=context
    )

    # 6. Store results in database
    store_execution_result(result=formula_result)
```

---

## Support

For issues with Google Drive integration:

1. Check Render logs for error messages
2. Verify credentials JSON is valid
3. Test locally first before deploying
4. Review Google Cloud Console logs: https://console.cloud.google.com/logs

Platform logs location:
```
Render Dashboard → ml-platform-backend → Logs
```

Filter for Google Drive messages:
```
"Google Drive" OR "data_ingestion"
```
