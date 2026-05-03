# Backend Integration Guide

## ✅ Integration Status

The frontend is now fully integrated with the backend! Here's what's connected:

### 1. **API Endpoints** ✅
- ✅ `/api/match/run` - Face matching (works without auth for testing)
- ✅ `/api/match/history` - Match history (requires auth)
- ✅ `/api/auth/login` - User login
- ✅ `/api/auth/register` - User registration
- ✅ `/api/auth/me` - Get current user profile

### 2. **Frontend Components** ✅
- ✅ Match page - Calls backend API for face matching
- ✅ Login page - Authenticates with backend
- ✅ Register page - Creates new users
- ✅ History page - Fetches match history from backend
- ✅ Axios interceptors - Automatically adds auth tokens

---

## 🚀 How to Run

### Step 1: Start Backend
```bash
# Activate conda environment
conda activate sketch2face-hybrid

# Navigate to backend
cd sketch2face-web/backend

# Start FastAPI server
python -m app.main
```
Backend runs on: `http://localhost:8000`

### Step 2: Start Frontend
```bash
# Navigate to frontend
cd sketch2face-web/frontend

# Install dependencies (if not done)
npm install

# Start dev server
npm run dev
```
Frontend runs on: `http://localhost:5173` (or 3000)

---

## 🔧 Configuration

### Backend CORS
The backend is configured to allow requests from:
- `http://localhost:3000`
- `http://localhost:5173`
- `http://localhost:5174`

### Frontend API URL
The frontend is configured to call:
- `http://localhost:8000/api` (default)

To change this, create `.env` file in `frontend/`:
```env
VITE_API_URL=http://localhost:8000/api
```

---

## 📋 API Response Formats

### Match Response (`/api/match/run`)
```json
{
  "query_id": "string",
  "input_image": "path/to/image.jpg",
  "generated_image": "path/to/generated.jpg",
  "top_matches": [
    {
      "match_id": "string",
      "similarity_score": 0.87,
      "image_path": "path/to/match.jpg",
      "name": "Subject A-2847"
    }
  ],
  "decision_intelligence": {
    "reliability_score": 84,
    "density_risk": "LOW",
    "consistency_verdict": "CONSISTENT",
    "final_decision": "ACCEPTED"
  },
  "timestamp": "2026-01-19T12:00:00Z"
}
```

### History Response (`/api/match/history`)
Returns array of MatchResponse objects.

### Login Response (`/api/auth/login`)
```json
{
  "access_token": "jwt_token_here",
  "token_type": "bearer"
}
```

---

## 🧪 Testing the Integration

### 1. Test Match Endpoint
1. Go to `/match` page
2. Upload an image
3. Select gallery (CelebA or CUFS)
4. Click "Analyze Image"
5. Should see results with Decision Intelligence

### 2. Test Login
1. Go to `/login` page
2. Enter email and password
3. Should redirect to `/match` on success
4. Token saved in localStorage

### 3. Test History
1. Login first
2. Go to `/history` page
3. Should show your match history

---

## ⚠️ Common Issues

### CORS Error
- **Problem**: Frontend can't call backend
- **Solution**: Check `BACKEND_CORS_ORIGINS` in `backend/app/core/config.py`
- **Fix**: Add your frontend URL to the list

### 401 Unauthorized
- **Problem**: Token expired or missing
- **Solution**: Login again to get new token
- **Note**: Match endpoint works without auth (for testing)

### Image Not Loading
- **Problem**: Image paths are relative
- **Solution**: Backend serves images at `/uploads` and `/results`
- **Fix**: Frontend converts paths to full URLs

### MongoDB Connection Error
- **Problem**: MongoDB not running
- **Solution**: Start MongoDB:
  ```bash
  # Windows
  net start MongoDB
  
  # Or check if running
  mongosh
  ```

---

## 📝 Next Steps

1. **Test the full flow:**
   - Register → Login → Match → History

2. **Verify image serving:**
   - Check if uploaded/generated images display correctly

3. **Test with real data:**
   - Upload actual sketch/photo
   - Verify matching works

4. **Production setup:**
   - Update CORS origins
   - Set proper SECRET_KEY
   - Configure MongoDB connection string
   - Set up environment variables

---

## ✅ Integration Checklist

- ✅ Frontend API calls match backend endpoints
- ✅ Auth token handling (interceptors)
- ✅ CORS configured
- ✅ Error handling in place
- ✅ Response format matching
- ✅ Image URL handling
- ✅ History page integration
- ✅ Match page integration
- ✅ Login/Register integration

**Everything is ready to test!** 🎉
