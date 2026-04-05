"""
Test Database Connection

Quick script to verify MongoDB is working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.database import connect_to_mongo, get_database, db

async def test_database():
    """Test database connection and operations"""
    print("=" * 60)
    print("🧪 Testing MongoDB Connection")
    print("=" * 60)
    
    # Test 1: Connection
    print("\n1️⃣ Testing Connection...")
    try:
        await connect_to_mongo()
        if db.connected:
            print("   ✅ Connection successful!")
        else:
            print("   ❌ Connection failed")
            return
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return
    
    # Test 2: Database access
    print("\n2️⃣ Testing Database Access...")
    try:
        database = await get_database()
        if database is None:
            print("   ❌ Database not available")
            return
        print(f"   ✅ Database '{settings.DATABASE_NAME}' accessible")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return
    
    # Test 3: Collections
    print("\n3️⃣ Checking Collections...")
    try:
        collections = await database.list_collection_names()
        print(f"   📦 Found {len(collections)} collections:")
        for col in collections:
            count = await database[col].count_documents({})
            print(f"      - {col}: {count} documents")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 4: Write test
    print("\n4️⃣ Testing Write Operation...")
    try:
        test_collection = database["test_connection"]
        result = await test_collection.insert_one({
            "test": True,
            "timestamp": asyncio.get_event_loop().time()
        })
        print(f"   ✅ Write successful (ID: {result.inserted_id})")
        
        # Clean up
        await test_collection.delete_one({"_id": result.inserted_id})
        print("   🧹 Test document cleaned up")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 5: Indexes
    print("\n5️⃣ Checking Indexes...")
    try:
        users_indexes = await database["users"].index_information()
        print(f"   📊 users collection: {len(users_indexes)} indexes")
        
        match_logs_indexes = await database["match_logs"].index_information()
        print(f"   📊 match_logs collection: {len(match_logs_indexes)} indexes")
    except Exception as e:
        print(f"   ⚠️  Index check failed: {str(e)}")
        print("   💡 Run 'python scripts/init_database.py' to create indexes")
    
    print("\n" + "=" * 60)
    print("✅ Database test complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_database())
