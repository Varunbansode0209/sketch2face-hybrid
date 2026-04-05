"""
Database Initialization Script

Creates indexes for better query performance.
Run this after setting up MongoDB.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

async def init_database():
    """Initialize database with indexes"""
    print("🔌 Connecting to MongoDB...")
    
    try:
        client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=5000
        )
        
        # Test connection
        await client.admin.command('ping')
        print(f"✅ Connected to MongoDB at {settings.MONGODB_URL}")
        
        db = client[settings.DATABASE_NAME]
        print(f"📦 Using database: {settings.DATABASE_NAME}")
        
        # Create indexes
        print("\n📊 Creating indexes...")
        
        # Users collection
        await db["users"].create_index("email", unique=True)
        print("  ✅ users.email (unique)")
        
        # Match logs collection
        await db["match_logs"].create_index("uploaded_by")
        print("  ✅ match_logs.uploaded_by")
        await db["match_logs"].create_index("timestamp")
        print("  ✅ match_logs.timestamp")
        await db["match_logs"].create_index("gallery")
        print("  ✅ match_logs.gallery")
        await db["match_logs"].create_index([("uploaded_by", 1), ("timestamp", -1)])
        print("  ✅ match_logs.uploaded_by + timestamp (compound)")
        
        # Suspects collection
        await db["suspects"].create_index("gallery")
        print("  ✅ suspects.gallery")
        await db["suspects"].create_index("uploaded_by")
        print("  ✅ suspects.uploaded_by")
        await db["suspects"].create_index("name")
        print("  ✅ suspects.name")
        
        print("\n✅ Database initialization complete!")
        print(f"\n📈 Collections ready:")
        print(f"   - users")
        print(f"   - match_logs")
        print(f"   - suspects")
        
        client.close()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Make sure MongoDB is running and connection string is correct.")
        print(f"   Current URL: {settings.MONGODB_URL}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(init_database())
