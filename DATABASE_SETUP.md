# Database Setup Guide

## Overview

The application uses **MongoDB** to store:
- **Users** - Authentication and user profiles
- **Match Logs** - History of all face matching queries
- **Suspects** - Photos uploaded by police officers for gallery matching

---

## 🚀 Quick Setup

### Option 1: Install MongoDB Locally (Recommended for Development)

#### Windows:
1. **Download MongoDB Community Server:**
   - Visit: https://www.mongodb.com/try/download/community
   - Select: Windows, MSI package
   - Download and install

2. **Start MongoDB Service:**
   ```powershell
   # MongoDB should start automatically as a Windows service
   # Or start manually:
   net start MongoDB
   ```

3. **Verify Installation:**
   ```powershell
   mongod --version
   ```

#### macOS:
```bash
# Using Homebrew
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

#### Linux (Ubuntu/Debian):
```bash
# Install MongoDB
sudo apt-get install -y mongodb

# Start MongoDB
sudo systemctl start mongodb
sudo systemctl enable mongodb
```

---

### Option 2: Use MongoDB Atlas (Cloud - Free Tier)

1. **Sign up:** https://www.mongodb.com/cloud/atlas/register
2. **Create a free cluster** (M0 - Free tier)
3. **Get connection string:**
   - Click "Connect" → "Connect your application"
   - Copy the connection string
   - Replace `<password>` with your database password

4. **Update `.env` file:**
   ```env
   MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
   ```

---

## ⚙️ Configuration

### 1. Create `.env` file

Create `sketch2face-web/backend/.env`:

```env
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=sketch2face_db

# Security (Change in production!)
SECRET_KEY=your-secret-key-change-this-in-production-use-a-random-string
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Optional: MongoDB with authentication
# MONGODB_URL=mongodb://username:password@localhost:27017/?authSource=admin
```

### 2. Default Settings

The app uses these defaults (from `config.py`):
- **MongoDB URL:** `mongodb://localhost:27017`
- **Database Name:** `sketch2face_db`
- **Port:** `27017` (default MongoDB port)

---

## 🗄️ Database Collections

The application automatically creates these collections:

### 1. `users`
Stores user accounts for authentication.

**Schema:**
```json
{
  "_id": ObjectId,
  "email": "user@example.com",
  "name": "John Doe",
  "hashed_password": "bcrypt_hash",
  "role": "police" | "investigator",
  "created_at": ISODate
}
```

### 2. `match_logs`
Stores all face matching queries and results.

**Schema:**
```json
{
  "_id": ObjectId,
  "uploaded_by": "user@example.com",
  "gallery": "cufs" | "celeba",
  "input_image_path": "uploads/xxx.jpg",
  "generated_image_path": "results/generated_xxx.jpg",
  "top_matches": [
    {
      "match_id": "123",
      "similarity_score": 0.85,
      "image_path": "gallery/celeba/xxx.jpg",
      "name": "person_name"
    }
  ],
  "heatmap_path": "results/heatmaps/xxx.jpg",
  "decision_intelligence": {
    "reliability_score": 82,
    "density_risk": "MEDIUM",
    "consistency_verdict": "CONSISTENT",
    "final_decision": "ACCEPTED"
  },
  "timestamp": ISODate
}
```

### 3. `suspects`
Stores suspect photos uploaded by police officers.

**Schema:**
```json
{
  "_id": ObjectId,
  "name": "Suspect Name",
  "gallery": "cufs" | "celeba",
  "image_path": "data/raw/cufs/photos/xxx.jpg",
  "embedding_path": "path/to/embedding.npy",
  "uploaded_by": "police@example.com",
  "created_at": ISODate
}
```

---

## 🔧 Database Initialization (Optional)

### Create Indexes for Better Performance

Run this script to create indexes:

```python
# scripts/init_database.py
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

async def init_database():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    
    # Create indexes
    await db["users"].create_index("email", unique=True)
    await db["match_logs"].create_index("uploaded_by")
    await db["match_logs"].create_index("timestamp")
    await db["match_logs"].create_index("gallery")
    await db["suspects"].create_index("gallery")
    await db["suspects"].create_index("uploaded_by")
    
    print("✅ Database indexes created successfully!")
    client.close()

if __name__ == "__main__":
    asyncio.run(init_database())
```

Run it:
```bash
cd sketch2face-web/backend
python scripts/init_database.py
```

---

## ✅ Testing Database Connection

### 1. Check Backend Startup

When you start the backend, you should see:
```
✅ Connected to MongoDB at mongodb://localhost:27017
✅ Application startup complete
```

If MongoDB is not running, you'll see:
```
⚠️  MongoDB not available: [WinError 10061]...
   Continuing without database (matches will work but won't be saved)
```

### 2. Test Health Endpoint

```bash
# Check database health
curl http://localhost:8000/api/health/db
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "2025-01-22T..."
}
```

### 3. Test Registration

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "test123",
    "name": "Test User",
    "role": "investigator"
  }'
```

---

## 🐛 Troubleshooting

### Issue: "Connection refused" or "Target machine actively refused"

**Solution:**
1. Check if MongoDB is running:
   ```bash
   # Windows
   net start MongoDB
   
   # macOS/Linux
   sudo systemctl status mongodb
   ```

2. Verify MongoDB is listening on port 27017:
   ```bash
   # Windows
   netstat -an | findstr 27017
   
   # macOS/Linux
   netstat -an | grep 27017
   ```

### Issue: "Authentication failed"

**Solution:**
- If MongoDB has authentication enabled, update `.env`:
  ```env
  MONGODB_URL=mongodb://username:password@localhost:27017/?authSource=admin
  ```

### Issue: "Database not found"

**Solution:**
- MongoDB creates databases automatically on first write
- No need to create the database manually
- Just ensure MongoDB is running and connection string is correct

---

## 📊 Database Management Tools

### MongoDB Compass (GUI)
- Download: https://www.mongodb.com/products/compass
- Connect to: `mongodb://localhost:27017`
- Browse collections, run queries, view data

### MongoDB Shell (CLI)
```bash
# Connect to MongoDB
mongosh

# Or with connection string
mongosh "mongodb://localhost:27017"

# Use database
use sketch2face_db

# View collections
show collections

# Query match logs
db.match_logs.find().pretty()

# Count documents
db.match_logs.countDocuments()
```

---

## 🔒 Production Considerations

1. **Enable Authentication:**
   - Create admin user in MongoDB
   - Update `MONGODB_URL` with credentials

2. **Use Environment Variables:**
   - Never commit `.env` file
   - Use secure secret keys

3. **Backup Strategy:**
   ```bash
   # Backup database
   mongodump --db sketch2face_db --out /backup/path
   
   # Restore database
   mongorestore --db sketch2face_db /backup/path/sketch2face_db
   ```

4. **Connection Pooling:**
   - Already configured in `database.py`
   - Motor (async driver) handles connection pooling automatically

---

## ✅ Next Steps

1. **Install MongoDB** (if not already installed)
2. **Start MongoDB service**
3. **Create `.env` file** with MongoDB URL
4. **Restart backend** - should connect automatically
5. **Test connection** via health endpoint
6. **Create indexes** (optional, for performance)

Once MongoDB is running, the backend will automatically:
- Save match history
- Store user accounts
- Enable authentication
- Store suspect photos

---

## 📝 Summary

- **Database:** MongoDB
- **Default URL:** `mongodb://localhost:27017`
- **Database Name:** `sketch2face_db`
- **Collections:** `users`, `match_logs`, `suspects`
- **Status:** Optional (app works without it, but features are limited)

The app gracefully handles MongoDB being unavailable, but for full functionality (user accounts, match history, suspect management), MongoDB should be running.
