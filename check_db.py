import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv("c:/Users/ADMIN/OneDrive/Desktop/sketch2face-hybrid-backup/sketch2face-web/backend/.env")
url = os.getenv("MONGODB_URL")

async def test():
    print(f"Connecting to: {url.split('@')[-1] if '@' in url else url}")
    client = AsyncIOMotorClient(url)
    db = client["sketch2face"]
    
    users = await db["users"].count_documents({})
    suspects = await db["suspects"].count_documents({})
    
    print(f"Users found: {users}")
    print(f"Suspects found: {suspects}")

asyncio.run(test())
