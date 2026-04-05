from fastapi import APIRouter, Depends
from app.database import get_database
from datetime import datetime

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "service": "Sketch2Face API"
    }

@router.get("/db")
async def database_health(db = Depends(get_database)):
    """Check database connection"""
    if db is None:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": "MongoDB not available",
            "timestamp": datetime.utcnow()
        }
    
    try:
        # Ping database
        await db.command("ping")
        
        # Get collection counts
        collections_info = {}
        try:
            collections_info = {
                "users": await db["users"].count_documents({}),
                "match_logs": await db["match_logs"].count_documents({}),
                "suspects": await db["suspects"].count_documents({})
            }
        except:
            pass
        
        return {
            "status": "healthy",
            "database": "connected",
            "collections": collections_info,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }