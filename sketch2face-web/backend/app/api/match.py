from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List
import numpy as np
from app.models.user import User
from app.models.match_log import MatchResponse, MatchResult, DecisionIntelligence
from app.services.auth_service import get_current_user, get_current_user_optional
from typing import Optional
from app.core.ai_engine import ai_engine
from app.core.decision_engine import decision_engine
from app.utils.image_utils import save_upload_file, generate_unique_filename, validate_image
from app.database import get_database
from app.core.config import settings
from datetime import datetime
import os

router = APIRouter(prefix="/match", tags=["Matching"])

@router.post("/run", response_model=MatchResponse)
async def run_match(
    file: UploadFile = File(...),
    gallery: str = Form(...),
    current_user: Optional[User] = Depends(get_current_user_optional),  # Optional for testing
    db = Depends(get_database)  # Will be None if MongoDB unavailable
):
    """
    Run face matching against selected gallery
    
    - **file**: Image or sketch file
    - **gallery**: "cufs" or "celeba"
    """
    
    # Validate gallery
    if gallery not in ["cufs", "celeba"]:
        raise HTTPException(status_code=400, detail="Invalid gallery. Must be 'cufs' or 'celeba'")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file
        unique_filename = generate_unique_filename(file.filename)
        upload_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
        await save_upload_file(file, upload_path)
        
        # Validate image
        if not validate_image(upload_path):
            os.remove(upload_path)
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run AI matching pipeline
        ai_results = await ai_engine.run_matching(upload_path, gallery)
        
        # Parse AI results
        top_matches = [
            MatchResult(
                match_id=match["id"],
                similarity_score=match["score"],
                image_path=match["image_path"],
                name=match.get("name")
            )
            for match in ai_results.get("top_matches", [])
        ]
        
        # Run decision intelligence (use data from AI results)
        decision = decision_engine.analyze(
            top_matches=[m.dict() for m in top_matches],
            gallery=gallery,
            query_embedding=np.array(ai_results.get("query_embedding")) if ai_results.get("query_embedding") else None,
            gallery_embeddings=None,  # Will use fallback density check
            cross_gallery_data=None  # Can be enhanced later
        )
        
        # Create match log
        match_log = {
            "uploaded_by": current_user.email if current_user else "anonymous",
            "gallery": gallery,
            "input_image_path": upload_path,
            "generated_image_path": ai_results.get("generated_image"),
            "top_matches": [m.dict() for m in top_matches],
            "heatmap_path": ai_results.get("heatmap"),
            "decision_intelligence": decision.dict(),
            "timestamp": datetime.utcnow()
        }
        
        # Save to database (optional - continue even if DB is unavailable)
        query_id = "no-db"
        if db is not None:
            try:
                result = await db["match_logs"].insert_one(match_log)
                query_id = str(result.inserted_id)
            except Exception as db_error:
                # Log but don't fail the request
                print(f"⚠️  Database save failed (continuing anyway): {str(db_error)}")
                query_id = f"no-db-{datetime.utcnow().timestamp()}"
        else:
            print("⚠️  Database not available, skipping save")
        
        # Return response
        return MatchResponse(
            query_id=query_id,
            input_image=upload_path,
            generated_image=ai_results.get("generated_image"),
            top_matches=top_matches,
            heatmap=ai_results.get("heatmap"),
            decision_intelligence=decision,
            timestamp=match_log["timestamp"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Match processing failed: {str(e)}"
        )

@router.get("/history", response_model=List[MatchResponse])
async def get_match_history(
    current_user: Optional[User] = Depends(get_current_user_optional),
    db = Depends(get_database),  # Will be None if MongoDB unavailable
    limit: int = 10
):
    """Get user's match history"""
    
    if not current_user:
        return []  # Return empty if not authenticated
    
    if db is None:
        return []  # Return empty if database not available
    
    try:
        matches = await db["match_logs"].find(
            {"uploaded_by": current_user.email}
        ).sort("timestamp", -1).limit(limit).to_list(length=limit)
    except Exception as e:
        print(f"⚠️  Database query failed: {str(e)}")
        return []
    
    result = []
    for match in matches:
        result.append(MatchResponse(
            query_id=str(match["_id"]),
            input_image=match["input_image_path"],
            generated_image=match.get("generated_image_path"),
            top_matches=[MatchResult(**m) for m in match["top_matches"]],
            heatmap=match.get("heatmap_path"),
            decision_intelligence=DecisionIntelligence(**match["decision_intelligence"]),
            timestamp=match["timestamp"]
        ))
    
    return result

@router.get("/{query_id}", response_model=MatchResponse)
async def get_match_by_id(
    query_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db = Depends(get_database)
):
    """Get specific match result by query ID"""
    from bson import ObjectId
    
    try:
        match = await db["match_logs"].find_one({"_id": ObjectId(query_id)})
        
        if not match:
            raise HTTPException(status_code=404, detail="Match not found")
        
        # Check ownership (or investigator access) - only if user is authenticated
        if current_user:
            if match["uploaded_by"] != current_user.email and current_user.role not in ["investigator", "admin"]:
                raise HTTPException(status_code=403, detail="Not authorized to view this match")
        # If not authenticated, still allow viewing (for testing)
        
        return MatchResponse(
            query_id=str(match["_id"]),
            input_image=match["input_image_path"],
            generated_image=match.get("generated_image_path"),
            top_matches=[MatchResult(**m) for m in match["top_matches"]],
            heatmap=match.get("heatmap_path"),
            decision_intelligence=DecisionIntelligence(**match["decision_intelligence"]),
            timestamp=match["timestamp"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))