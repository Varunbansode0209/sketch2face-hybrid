from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Dict
from app.models.user import User
from app.models.suspect import SuspectCreate, Suspect, SuspectUploadResponse
from app.services.auth_service import get_current_admin_user
from app.core.ai_engine import ai_engine
from app.utils.image_utils import save_upload_file, generate_unique_filename, validate_image
from app.database import get_database
from app.core.config import settings
from datetime import datetime
import os
import numpy as np

from bson import ObjectId

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.post("/upload-suspect", response_model=SuspectUploadResponse)
async def upload_suspect(
    file: UploadFile = File(...),
    gallery: str = Form(...),
    name: str = Form(None),
    description: str = Form(None),
    gender: str = Form(None),
    age: int = Form(None),
    current_user: User = Depends(get_current_admin_user),
    db = Depends(get_database)
):
    """
    Upload new suspect image to gallery (Police only)
    
    - **file**: Suspect image
    - **gallery**: "cufs" or "celeba"
    - **name**: Optional suspect name
    """
    
    if gallery not in ["cufs", "celeba"]:
        raise HTTPException(status_code=400, detail="Invalid gallery")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    if db is None:
        raise HTTPException(status_code=503, detail="Database is unavailable")
    
    try:
        # Determine gallery path
        gallery_path = settings.CUFS_GALLERY_PATH if gallery == "cufs" else settings.CELEBA_GALLERY_PATH
        
        # Save file
        unique_filename = generate_unique_filename(file.filename)
        save_path = os.path.join(gallery_path, unique_filename)
        await save_upload_file(file, save_path)
        
        # Validate image
        if not validate_image(save_path):
            os.remove(save_path)
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Extract coordinates (embeddings)
        coordinates = await ai_engine.extract_features(save_path)
        
        # Save to database
        suspect_data = {
            "name": name,
            "description": description,
            "gender": gender,
            "age": age,
            "gallery": gallery,
            "image_path": save_path,
            "embedding_path": f"{save_path}.npy",  # Kept for backward compat
            "coordinates": coordinates,  # Live JSON vector
            "uploaded_by": current_user.email,
            "created_at": datetime.utcnow()
        }
        
        # Also save the .npy file independently to the disk just in case the legacy system needs it
        np.save(f"{save_path}.npy", np.array(coordinates))
        
        result = await db["suspects"].insert_one(suspect_data)
        
        return SuspectUploadResponse(
            suspect_id=str(result.inserted_id),
            message="Suspect uploaded successfully. Rebuild gallery to include in matching.",
            image_path=save_path
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

@router.post("/rebuild-gallery")
async def rebuild_gallery(
    gallery: str,
    current_user: User = Depends(get_current_admin_user)
):
    """
    Rebuild gallery embeddings (Police only)
    
    This triggers the ML pipeline to regenerate all embeddings
    """
    
    if gallery not in ["cufs", "celeba", "all"]:
        raise HTTPException(status_code=400, detail="Invalid gallery")
    
    try:
        if gallery == "all":
            cufs_result = await ai_engine.rebuild_gallery("cufs")
            celeba_result = await ai_engine.rebuild_gallery("celeba")
            return {
                "cufs": cufs_result,
                "celeba": celeba_result
            }
        else:
            result = await ai_engine.rebuild_gallery(gallery)
            return result
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gallery rebuild failed: {str(e)}"
        )

@router.get("/logs")
async def get_system_logs(
    current_user: User = Depends(get_current_admin_user),
    db = Depends(get_database),
    limit: int = 50
):
    """Get recent system logs (Police only)"""
    
    logs = await db["match_logs"].find().sort("timestamp", -1).limit(limit).to_list(length=limit)
    
    return [{
        "query_id": str(log["_id"]),
        "user": log["uploaded_by"],
        "gallery": log["gallery"],
        "timestamp": log["timestamp"],
        "top1_score": log["top_matches"][0]["similarity_score"] if log["top_matches"] else 0,
        "decision": log["decision_intelligence"]["final_decision"]
    } for log in logs]

@router.get("/statistics")
async def get_statistics(
    current_user: User = Depends(get_current_admin_user),
    db = Depends(get_database)
):
    """Get system statistics (Police only)"""
    
    if db is None:
        return {
            "total_queries": 0,
            "total_users": 0,
            "total_suspects": 0,
            "high_confidence_matches": 0,
            "confidence_rate": 0
        }
    
    total_queries = await db["match_logs"].count_documents({})
    total_users = await db["users"].count_documents({})
    total_suspects = await db["suspects"].count_documents({})
    
    # High confidence matches
    high_confidence = await db["match_logs"].count_documents({
        "decision_intelligence.final_decision": "HIGH_CONFIDENCE_MATCH"
    })
    
    return {
        "total_queries": total_queries,
        "total_users": total_users,
        "total_suspects": total_suspects,
        "high_confidence_matches": high_confidence,
        "confidence_rate": high_confidence / total_queries if total_queries > 0 else 0
    }

@router.get("/suspects")
async def list_suspects(
    gallery: str = None,
    current_user: User = Depends(get_current_admin_user),
    db = Depends(get_database),
    limit: int = 50
):
    """List all suspects in gallery (Police only)"""
    if db is None:
        return []
    
    query = {"gallery": gallery} if gallery else {}
    suspects = await db["suspects"].find(query).limit(limit).to_list(length=limit)
    
    return [{
        "id": str(s["_id"]),
        "name": s.get("name"),
        "description": s.get("description"),
        "gender": s.get("gender"),
        "age": s.get("age"),
        "gallery": s["gallery"],
        "image_path": s["image_path"],
        "created_at": s["created_at"]
    } for s in suspects]

@router.delete("/suspects/{suspect_id}")
async def delete_suspect(
    suspect_id: str,
    current_user: User = Depends(get_current_admin_user),
    db = Depends(get_database)
):
    """Delete a suspect image (Admin only)"""
    try:
        suspect = await db["suspects"].find_one({"_id": ObjectId(suspect_id)})
        if not suspect:
            raise HTTPException(status_code=404, detail="Suspect not found")
        
        # Delete from DB
        await db["suspects"].delete_one({"_id": ObjectId(suspect_id)})
        
        # Optionally remove physical files
        try:
            if os.path.exists(suspect["image_path"]):
                os.remove(suspect["image_path"])
        except:
            pass
            
        return {"message": "Suspect deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users")
async def get_all_users(
    current_user: User = Depends(get_current_admin_user),
    db = Depends(get_database),
    limit: int = 50
):
    """Get all users (Admin only)"""
    if db is None:
        return {"users": []}
        
    users = await db["users"].find().limit(limit).to_list(length=limit)
    return {
        "users": [{
            "id": str(u["_id"]),
            "email": u["email"],
            "name": u["name"],
            "role": u.get("role", "public"),
            "createdAt": u["created_at"]
        } for u in users]
    }

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(get_current_admin_user),
    db = Depends(get_database)
):
    """Delete user (Admin only)"""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
        
    try:
        result = await db["users"].delete_one({"_id": ObjectId(user_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        return {"message": "User deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))