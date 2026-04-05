import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv("c:/Users/ADMIN/OneDrive/Desktop/sketch2face-hybrid-backup/sketch2face-web/backend/.env")
url = os.getenv("MONGODB_URL")

print("Connecting to MongoDB...")
client = MongoClient(url)

for db_name in client.list_database_names():
    if "sketch" not in db_name.lower() and db_name not in ["admin", "local", "config"]:
        continue
    print(f"\nDatabase: {db_name}")
    db = client[db_name]
    for coll_name in db.list_collection_names():
        count = db[coll_name].count_documents({})
        print(f"  - Collection '{coll_name}': {count} documents")
