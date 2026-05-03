# Backend Review & Integration Guide

## 📋 Executive Summary

**Status:** Backend structure is good, but needs integration fixes with your existing AI system.

**Critical Issues:**
1. ❌ `ai_engine.py` calls `src.run_full_system` incorrectly (uses CLI args, but system uses interactive input)
2. ❌ `decision_engine.py` duplicates logic - should use your existing `src.decision` modules
3. ⚠️ Config paths point to wrong locations
4. ⚠️ Missing integration wrapper for your existing pipeline

**What's Good:**
- ✅ FastAPI structure is clean
- ✅ Database models are well-designed
- ✅ API endpoints match PRD requirements
- ✅ Authentication structure is in place

---

## 🔍 Detailed Review

### 1. **AI Engine Integration** (`app/core/ai_engine.py`)

**Problem:**
```python
# Current code tries:
cmd = ["python", "-m", "src.run_full_system", "--input", ...]
# But your actual system uses interactive input!
```

**Your actual system:**
```python
# src/run_full_system.py uses:
gallery_name = input("👉 Select gallery (cufs / celeba): ")
sketch_input = input("👉 Enter sketch image path: ")
```

**Solution:** Create a wrapper function that calls your system programmatically.

---

### 2. **Decision Engine Duplication** (`app/core/decision_engine.py`)

**Problem:**
- Backend has its own Decision Intelligence implementation
- You already have `src.decision` modules (MRS, GDA, CGCC)
- Duplication = maintenance nightmare

**Solution:** Import and use your existing `src.decision` modules directly.

---

### 3. **Config Paths** (`app/core/config.py`)

**Problem:**
```python
AI_CORE_PATH: str = "../ai_core"  # Wrong path!
```

**Solution:** Point to actual project root.

---

## 🛠️ Fixes Required

### Fix 1: Create AI Pipeline Wrapper

Create `sketch2face-web/backend/app/core/ai_pipeline_wrapper.py` that:
- Calls your existing `src.run_full_system` functions directly (not via CLI)
- Returns structured results
- Handles errors gracefully

### Fix 2: Update Decision Engine

Modify `app/core/decision_engine.py` to:
- Import from `src.decision` instead of reimplementing
- Use your existing MRS, GDA, CGCC modules

### Fix 3: Fix Config Paths

Update `app/core/config.py` to point to correct locations.

### Fix 4: Create Result Parser

Create utility to parse outputs from your pipeline into API response format.

---

## 📝 Step-by-Step Implementation Plan

### Phase 1: Fix Critical Integration (Priority 1)

1. **Create AI Pipeline Wrapper**
   - File: `app/core/ai_pipeline_wrapper.py`
   - Function: `run_matching_pipeline(image_path, gallery_name)`
   - Returns: Structured dict with matches, heatmap, generated image paths

2. **Update AI Engine**
   - Replace subprocess calls with direct function calls
   - Use wrapper function

3. **Fix Decision Engine**
   - Import from `src.decision`
   - Remove duplicate logic

4. **Fix Config**
   - Update paths to point to project root

### Phase 2: Test Integration (Priority 2)

1. Test API endpoint `/api/match/run`
2. Verify results match CLI output
3. Test Decision Intelligence integration

### Phase 3: Frontend Integration (Priority 3)

1. Connect frontend to fixed backend
2. Test end-to-end flow

---

## 🎯 Next Steps

**Option A: I fix the backend integration now** (Recommended)
- I'll create the wrapper functions
- Fix all integration issues
- Make it work with your existing system

**Option B: You review first, then I fix**
- You check the fixes I propose
- Then I implement

**Option C: Manual fix guide**
- I provide detailed code snippets
- You implement yourself

---

## ⚠️ Important Notes

1. **Don't modify `src.run_full_system.py`** - Keep it as-is for CLI use
2. **Create wrapper functions** - Bridge between FastAPI and your system
3. **Use existing Decision Intelligence** - Don't duplicate logic
4. **Test incrementally** - Fix one thing, test, then move on

---

## 📊 Current Backend Structure

```
sketch2face-web/backend/
├── app/
│   ├── main.py              ✅ Good
│   ├── database.py          ✅ Good (needs MongoDB setup)
│   ├── core/
│   │   ├── config.py        ⚠️ Needs path fixes
│   │   ├── ai_engine.py     ❌ Needs wrapper integration
│   │   └── decision_engine.py ❌ Should use src.decision
│   ├── api/
│   │   ├── match.py         ✅ Good structure
│   │   ├── auth.py          ✅ Good structure
│   │   └── admin.py         ✅ Good structure
│   └── models/
│       └── match_log.py     ✅ Good
└── requirements.txt         ✅ Good
```

---

**Ready to proceed?** Tell me which option you prefer (A, B, or C) and I'll start fixing!
