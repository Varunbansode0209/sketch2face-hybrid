from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class MatchResult(BaseModel):
    match_id: str
    similarity_score: float
    image_path: str
    name: Optional[str] = None

class DecisionIntelligence(BaseModel):
    reliability_score: float
    density_risk: str
    consistency_verdict: str
    final_decision: str

class MatchLogCreate(BaseModel):
    uploaded_by: str
    gallery: str
    input_image_path: str
    generated_image_path: Optional[str] = None
    top_matches: List[MatchResult]
    heatmap_path: Optional[str] = None
    decision_intelligence: DecisionIntelligence

class MatchLog(MatchLogCreate):
    id: str = Field(alias="_id")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True

class MatchRequest(BaseModel):
    gallery: str  # "cufs" or "celeba"

class MatchResponse(BaseModel):
    query_id: str
    input_image: str
    generated_image: Optional[str] = None
    top_matches: List[MatchResult]
    heatmap: Optional[str] = None
    decision_intelligence: DecisionIntelligence
    timestamp: datetime