# Alternative Approaches to Improve CelebA Matching

## Current Issue
- Generated faces are blurry/poor quality
- Confidence margin too low (0.0221 < 0.05 threshold)
- Face detection sometimes fails
- Need to present at least 3-5 matches

---

## APPROACH 1: Gallery-Specific Thresholds (Easiest, 5 min)
**Why:** CelebA has 40k identities vs CUFS ~200, so naturally has tighter similarity scores.

**Implementation:**
- Lower `CONF_MARGIN` for CelebA gallery (e.g., 0.02 instead of 0.05)
- Keep CUFS thresholds unchanged

**Pros:** ✅ Zero code changes to pipeline, just config
**Cons:** ⚠️ May accept some false positives

---

## APPROACH 2: Image Enhancement Post-Processing (Quick, 15 min)
**Why:** Generated faces are blurry → enhance before face detection/embedding.

**Implementation:**
- Apply sharpening filter to generated image
- Optional: denoising, contrast enhancement
- Then proceed with face detection

**Pros:** ✅ Improves face detection success rate
**Cons:** ⚠️ May introduce artifacts

---

## APPROACH 3: Sketch-to-Sketch Matching (Medium, 1-2 hrs)
**Why:** Your original idea - match sketches directly, bypassing blurry generation.

**Implementation:**
1. Pre-compute sketch embeddings for all CelebA photos (convert photos → sketches → embeddings)
2. Match input sketch embedding against sketch gallery embeddings
3. Show top-K matches

**Pros:** ✅ Bypasses generation quality issues entirely
**Cons:** ⚠️ Requires building new gallery, sketch embeddings may be less discriminative

---

## APPROACH 4: Hybrid Matching (Medium, 1-2 hrs)
**Why:** Combine sketch-to-sketch AND photo-to-photo scores for robustness.

**Implementation:**
1. Generate photo from sketch (current pipeline)
2. Also compute sketch embedding directly
3. Match both: photo embedding vs photo gallery, sketch embedding vs sketch gallery
4. Weighted fusion: `final_score = 0.6 * photo_score + 0.4 * sketch_score`

**Pros:** ✅ More robust, leverages both modalities
**Cons:** ⚠️ Requires sketch gallery, more complex

---

## APPROACH 5: Better Fine-Tuning (Longer, 2-4 hrs)
**Why:** Current generator (5 epochs) may be under-trained.

**Implementation:**
- Resume training to 15-20 epochs
- Try different learning rates (5e-5, 1e-4)
- Increase batch size if GPU memory allows
- Add more CUFS regularization (increase CUFS ratio to 15-20%)

**Pros:** ✅ Addresses root cause (generation quality)
**Cons:** ⚠️ Time-consuming, may still not fully solve domain gap

---

## APPROACH 6: Ensemble Multiple Generators (Advanced, 2-3 hrs)
**Why:** Average outputs from multiple generator checkpoints for better quality.

**Implementation:**
- Train 2-3 generator variants (different seeds/hyperparameters)
- Generate 2-3 photos from same sketch
- Average or vote on best matches

**Pros:** ✅ More stable results
**Cons:** ⚠️ Requires multiple training runs, slower inference

---

## RECOMMENDED COMBINATION

**For immediate presentation:**
1. **Approach 1** (Gallery-specific thresholds) - 5 min
2. **Approach 2** (Image enhancement) - 15 min

**For better long-term solution:**
3. **Approach 3** (Sketch-to-sketch) OR **Approach 4** (Hybrid) - 1-2 hrs
4. **Approach 5** (Better fine-tuning) - Run in background

---

## Quick Win: Start with Approach 1 + 2

These can be implemented immediately and should improve results enough for presentation.
