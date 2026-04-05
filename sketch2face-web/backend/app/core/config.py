import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from dotenv import load_dotenv

# Force load
load_dotenv(override=True)

class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "Sketch2Face Hybrid"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "sketch2face_hybrid_db"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AI System
    # Paths relative to backend directory
    PROJECT_ROOT: str = "../.."  # Go up to sketch2face-hybrid-backup root
    UPLOAD_DIR: str = "./uploads"
    RESULTS_DIR: str = "./results"
    
    # Galleries (paths in main project)
    CUFS_GALLERY_PATH: str = "../../data/raw/cufs/photos"
    CELEBA_GALLERY_PATH: str = "../../data/raw/celeba/photos"
    
    # CORS
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()