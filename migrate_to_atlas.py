import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

# Load configuration
BASE_DIR = Path(r"c:\Users\ADMIN\OneDrive\Desktop\sketch2face-hybrid-backup")
load_dotenv(BASE_DIR / "sketch2face-web" / "backend" / ".env")

MONGO_URI = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
print(f"Connecting to MongoDB: {MONGO_URI.split('@')[-1] if '@' in MONGO_URI else MONGO_URI}")

client = MongoClient(MONGO_URI)
db = client["sketch2face"]  # Assuming DB name is sketch2face similar to config
collection = db["suspects"]

def migrate_gallery(gallery_name, index_path, npy_path, photo_dir):
    print(f"\nMigrating {gallery_name.upper()} gallery...")
    
    if not (BASE_DIR / index_path).exists() or not (BASE_DIR / npy_path).exists():
        print(f"Skipping {gallery_name} - files not found.")
        return

    with open(BASE_DIR / index_path, "r") as f:
        index = json.load(f)
        
    embeddings = np.load(BASE_DIR / npy_path)
    
    if len(index) != len(embeddings):
        print(f"Warning: {gallery_name} index length ({len(index)}) != embeddings length ({len(embeddings)})")
    
    records = []
    batch_size = 500
    total_inserted = 0

    print(f"Processing {len(index)} records from {gallery_name}...")
    
    for i, filename in enumerate(index):
        # Convert numpy floats to python native floats
        vector = embeddings[i].tolist()
        
        # Build document based on app/models/suspect.py SuspectBase schema
        doc = {
            "name": f"{gallery_name.upper()} Dataset Person {i+1}",
            "description": f"Imported automatically from local {gallery_name} raw dataset for presentation.",
            "gender": "Unknown",
            "age": None,
            "gallery": gallery_name,
            "image_path": str(BASE_DIR / photo_dir / filename),
            "embedding_path": str(BASE_DIR / npy_path), # Ref back to master npy for compat
            "coordinates": vector,
            "uploaded_by": "admin_migration_script",
            "created_at": datetime.utcnow()
        }
        records.append(doc)
        
        # Batch insert
        if len(records) >= batch_size:
            collection.insert_many(records)
            total_inserted += len(records)
            print(f"  Inserted {total_inserted}/{len(index)}...")
            records = []
            
    # Insert remaining
    if records:
        collection.insert_many(records)
        total_inserted += len(records)
        print(f"  Inserted {total_inserted}/{len(index)}...")

    print(f"Finished {gallery_name.upper()}: {total_inserted} inserted.")

if __name__ == "__main__":
    # Clear existing if needed? Or just append. We'll just append to be safe.
    print(f"Current document count in MongoDB suspects: {collection.count_documents({})}")
    print("Clearing generic automated test subjects if any to prevent duplicates...")
    # Optional logic to clear old: collection.delete_many({"uploaded_by": "admin_migration_script"})
    
    # 1. CUFS
    migrate_gallery(
        gallery_name="cufs",
        index_path="embeddings/gallery/cufs_index.json",
        npy_path="embeddings/gallery/cufs_gallery.npy",
        photo_dir="data/raw/cufs/photos"
    )
    
    # 2. CelebA
    migrate_gallery(
        gallery_name="celeba",
        index_path="embeddings/gallery/celeba_index.json",
        npy_path="embeddings/gallery/celeba_gallery.npy",
        photo_dir="data/raw/celeba/photos"
    )
    
    print(f"\nMigration Complete! Total documents now: {collection.count_documents({})}")
