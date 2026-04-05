from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class SuspectBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    gallery: str  # "cufs" or "celeba"
    image_path: str
    embedding_path: Optional[str] = None
    coordinates: Optional[list[float]] = None

class SuspectCreate(SuspectBase):
    uploaded_by: str

class Suspect(SuspectBase):
    id: str = Field(alias="_id")
    uploaded_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True

class SuspectUploadResponse(BaseModel):
    suspect_id: str
    message: str
    image_path: str