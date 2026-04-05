from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.database import connect_to_mongo, close_mongo_connection
from app.api import auth, match, admin, health
import os

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-Powered Sketch & Face Identification System"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()
    
    # Create necessary directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)
    os.makedirs(settings.CUFS_GALLERY_PATH, exist_ok=True)
    os.makedirs(settings.CELEBA_GALLERY_PATH, exist_ok=True)
    
    print("✅ Application startup complete")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()
    print("👋 Application shutdown complete")

# Mount static files for serving images
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/results", StaticFiles(directory=settings.RESULTS_DIR), name="results")

# Mount gallery directories for serving match images
from pathlib import Path
# Resolve project root relative to backend directory
backend_dir = Path(__file__).parent.parent  # app/ -> backend/
project_root = (backend_dir / settings.PROJECT_ROOT).resolve()

cufs_gallery_path = project_root / "data" / "raw" / "cufs" / "photos"
celeba_gallery_path = project_root / "data" / "raw" / "celeba" / "photos"

# Only mount if directories exist
if cufs_gallery_path.exists():
    app.mount("/gallery/cufs", StaticFiles(directory=str(cufs_gallery_path)), name="cufs_gallery")
    print(f"✅ Mounted CUFS gallery: {cufs_gallery_path}")
else:
    print(f"⚠️  CUFS gallery not found: {cufs_gallery_path}")

if celeba_gallery_path.exists():
    app.mount("/gallery/celeba", StaticFiles(directory=str(celeba_gallery_path)), name="celeba_gallery")
    print(f"✅ Mounted CelebA gallery: {celeba_gallery_path}")
else:
    print(f"⚠️  CelebA gallery not found: {celeba_gallery_path}")

# Include routers
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(match.router, prefix=settings.API_V1_STR)
app.include_router(admin.router, prefix=settings.API_V1_STR)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Sketch2Face Hybrid API",
        "version": settings.VERSION,
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )