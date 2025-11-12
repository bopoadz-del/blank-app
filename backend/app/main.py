from pathlib import Path
from fastapi import FastAPI
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

if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

    @app.get("/__index")
    async def _index():
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    async def root():
        return JSONResponse(
            {"status": "backend", "message": "Frontend not found. Build the frontend and include frontend/dist in the deployment."},
            status_code=200,
        )

@app.get("/health")
async def health():
    return {"status": "ok"}

# Include API routers under /api to avoid collision with SPA routes
# app.include_router(api_router, prefix="/api")
