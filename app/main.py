"""FastAPI application entry point"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.rate_limit import rate_limiter
from app.api.v1 import formulas
from app.schemas.formula import HealthResponse
from app.database.session import engine
from app.models import formula_execution  # Import models for table creation


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    await rate_limiter.init_redis()

    # Create database tables
    from app.database.session import Base
    Base.metadata.create_all(bind=engine)

    yield

    # Shutdown
    await rate_limiter.close_redis()


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Formula execution API with rate limiting and authentication",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint (no auth required)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns API status and version information.
    No authentication required.
    """
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT
    )


# Include API routers
app.include_router(
    formulas.router,
    prefix=f"{settings.API_V1_PREFIX}/formulas",
    tags=["formulas"]
)


# Check if frontend build exists and serve it
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
frontend_available = frontend_dist.exists() and frontend_dist.is_dir()

if frontend_available:
    # Mount static assets
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    
    # Serve index.html for root and all frontend routes
    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    @app.get("/catalog", include_in_schema=False)
    @app.get("/formulas", include_in_schema=False)
    @app.get("/admin", include_in_schema=False)
    @app.get("/admin/certifications", include_in_schema=False)
    @app.get("/auditor", include_in_schema=False)
    @app.get("/login", include_in_schema=False)
    async def serve_spa():
        """Serve the React SPA"""
        return FileResponse(frontend_dist / "index.html")
else:
    # API-only mode - serve API information at root
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "docs": "/docs",
            "health": "/health",
            "frontend": "not_available",
            "note": "Frontend UI not found. Access API docs at /docs"
        }
