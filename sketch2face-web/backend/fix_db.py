import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv(".env")
url = os.getenv("MONGODB_URL")

print("Connecting to MongoDB to resolve storage quota...")
client = MongoClient(url)

src_legacy = client["sketch2face"]
src_db = client["sketch2face_db"]
dest_db = client["sketch2face_hybrid_db"]

# 1. Grab all suspects into memory to protect data
all_docs = {}
print("Loading all suspects into memory...")

for db_obj in [src_legacy, src_db, dest_db]:
    docs = list(db_obj["suspects"].find())
    for d in docs:
        # Dictionary inherently deduplicates based on _id
        all_docs[str(d["_id"])] = d

print(f"Total unique suspects backed up in RAM: {len(all_docs)}")

# 2. DROP EVERYTHING to entirely free up the 512MB Quota!
print("Dropping all duplicate collections to instantly clear hundreds of megabytes of space...")
src_legacy["suspects"].drop()
src_db["suspects"].drop()
dest_db["suspects"].drop() # Rebuild from the deduplicated list

# 3. Safely insert all of them into the target destination
print("Space successfully cleared! Writing the unified collection into sketch2face_hybrid_db...")

suspects_to_insert = list(all_docs.values())
if suspects_to_insert:
    # Insert in chunks if necessary, but 40k usually fits in one insert_many
    # Split into 5000 chunks just to be ultra safe for the payload limits (16MB BSON limit)
    chunk_size = 5000
    total_inserted = 0
    for i in range(0, len(suspects_to_insert), chunk_size):
        chunk = suspects_to_insert[i:i + chunk_size]
        dest_db["suspects"].insert_many(chunk)
        total_inserted += len(chunk)
        print(f"  -> Inserted {total_inserted} / {len(suspects_to_insert)}")
        
    print(f"✅ SUCCESS: {total_inserted} suspects securely moved to 'sketch2face_hybrid_db'!")
else:
    print("No suspects found to insert.")

print(f"\nFinal Space Optimization Complete! Check the Frontend!")
