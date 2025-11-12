from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os

app = FastAPI(title="Blank App - Unified UI+API")

# CORS - restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths: this file sits at backend/app, so climb to backend/
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST_PATH", str(FRONTEND_DIST)))

# Ensure health is always available and defined before any SPA mounting
@app.get("/health")
async def health():
    return {"status": "ok"}

if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
    # Mount only the static assets directory so it won't shadow API routes
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists() and assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    # Serve index.html for the root
    @app.get("/", include_in_schema=False)
    async def _index():
        return FileResponse(FRONTEND_DIST / "index.html")

    # Catch-all for SPA routes but avoid catching API or asset/health paths
    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_catch_all(full_path: str):
        # Prevent catching API or asset paths
        if full_path.startswith("api") or full_path.startswith("assets") or full_path == "health":
            raise HTTPException(status_code=404)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    async def root():
        return JSONResponse(
            {"status": "backend", "message": "Frontend not found. Build the frontend and include frontend/dist in the deployment."},
            status_code=200,
        )
