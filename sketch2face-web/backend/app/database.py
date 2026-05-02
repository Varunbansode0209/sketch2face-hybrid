from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import os
from app.core.config import settings

class Database:
    client: Optional[AsyncIOMotorClient] = None
    connected: bool = False
    
db = Database()

async def get_database():
    """Get database instance, returns None if not connected"""
    if db.client is None or not db.connected:
        return None
    return db.client[settings.DATABASE_NAME]

import certifi

async def connect_to_mongo():
    """Connect to MongoDB, gracefully handle connection failures"""
    try:
        db.client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            tlsCAFile=certifi.where()
        )
        # Test connection
        await db.client.admin.command('ping')
        db.connected = True
        print(f"✅ Connected to MongoDB at {settings.MONGODB_URL}")
    except Exception as e:
        db.connected = False
        print(f"⚠️  MongoDB not available: {str(e)}")
        print("   Continuing without database (matches will work but won't be saved)")

async def close_mongo_connection():
    """Close MongoDB connection"""
    if db.client:
        db.client.close()
        db.connected = False
        print("❌ Closed MongoDB connection")