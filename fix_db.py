import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv("c:/Users/ADMIN/OneDrive/Desktop/sketch2face-hybrid-backup/sketch2face-web/backend/.env")
url = os.getenv("MONGODB_URL")

print(f"Connecting to MongoDB...")
client = MongoClient(url)

# The web app uses sketch2face_db, but migrate_to_atlas pushed to sketch2face
src_db = client["sketch2face"]
dest_db = client["sketch2face_db"]

# Count source suspects
count = src_db["suspects"].count_documents({})
print(f"Found {count} suspects in wrong 'sketch2face' database.")

if count > 0:
    print("Moving them to 'sketch2face_db'...")
    # Fetch all
    docs = list(src_db["suspects"].find())
    
    # Optional: Clear existing generic ones from destination to avoid duplicates
    dest_db["suspects"].delete_many({})
    
    # Insert to destination
    dest_db["suspects"].insert_many(docs)
    print("Successfully moved all suspects to the correct database container!")
else:
    print("No suspects found to move.")
    
# Debug: Also print what users exist in the destination DB to confirm admin is there
users_count = dest_db["users"].count_documents({})
print(f"Registered Users in app DB: {users_count}")
