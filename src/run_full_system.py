import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from src.generation.pix2pix_infer import generate_face_from_sketch
from src.embedding.arcface_infer import get_embedding
from src.preprocess.detect_face import detect_faces
from src.matching.visualize_matches import visualize_topk
from src.explainablity.save_heatmap import save_heatmap_case
from src.utils.combine_result import combine_final_results
from src.preprocess.sketch_validator import is_forensic_style
from src.preprocess.normalize_sketch import normalize_to_cufs
from src.decision.reliability_score import compute_match_reliability
from src.decision.gallery_density import compute_gallery_density
from src.decision.cross_gallery_check import cross_gallery_consistency


# ================= CONFIG =================
OUT_ROOT = Path("processed/final_demo")
GEN_DIR  = OUT_ROOT / "generated"
TOPK_DIR = OUT_ROOT / "topk"
HEAT_DIR = OUT_ROOT / "heatmaps"

TOP_K = 5
THRESHOLD = 0.30

CONF_MARGIN_CUFS   = 0.05
CONF_MARGIN_CELEBA = 0.02
# =========================================


    # ---------- Gallery Loader ----------
def load_gallery(name):
    """Load gallery embeddings, index, and photo directory."""
    if name == "cufs":
        return (
            np.load("embeddings/gallery/cufs_gallery.npy"),
            json.load(open("embeddings/gallery/cufs_index.json")),
            Path("data/raw/cufs/photos")
        )

    elif name == "celeba":
        return (
            np.load("embeddings/gallery/celeba_gallery.npy"),
            json.load(open("embeddings/gallery/celeba_index.json")),
            Path("data/raw/celeba/photos")
        )

    else:
        raise ValueError("Invalid gallery selection")


def load_all_galleries():
    """Load both galleries for cross-gallery consistency check."""
    try:
        celeba_gallery = np.load("embeddings/gallery/celeba_gallery.npy")
        celeba_gallery = celeba_gallery / np.linalg.norm(celeba_gallery, axis=1, keepdims=True)
    except FileNotFoundError:
        celeba_gallery = None
    
    try:
        cufs_gallery = np.load("embeddings/gallery/cufs_gallery.npy")
        cufs_gallery = cufs_gallery / np.linalg.norm(cufs_gallery, axis=1, keepdims=True)
    except FileNotFoundError:
        cufs_gallery = None
    
    return celeba_gallery, cufs_gallery


def extract_face(img):
    boxes = detect_faces(img)
    if boxes:
        x1, y1, x2, y2 = boxes[0]
        return img[y1:y2, x1:x2]
    return None


def fallback_center_crop(img):
    h, w, _ = img.shape
    min_dim = min(h, w)
    cx, cy = w // 2, h // 2
    half = min_dim // 2

    crop = img[
        cy - half : cy + half,
        cx - half : cx + half
    ]
    return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)


def main():

    print("\n================ FINAL SYSTEM DEMO ================\n")

    gallery_name = input("👉 Select gallery (cufs / celeba): ").strip().lower()
    gallery, index, GALLERY_DIR = load_gallery(gallery_name)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)

    print(f"✔ Loaded {gallery_name.upper()} gallery with {gallery.shape[0]} identities")

    USE_GENERATION = (gallery_name == "cufs")

    sketch_input = input("\n👉 Enter sketch image path: ").strip()
    sketch_path = Path(sketch_input)

    if not sketch_path.exists():
        print("❌ Invalid sketch path")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    TOPK_DIR.mkdir(parents=True, exist_ok=True)
    HEAT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- STEP 1 ----------
    print("\n▶ STEP 1: Loading input image")
    sketch = cv2.imread(str(sketch_path))
    if sketch is None:
        print("❌ Failed to load image")
        return

    # ---------- STEP 2 ----------
    print("▶ STEP 2: Sketch style validation")
    if gallery_name == "cufs":
        if not is_forensic_style(sketch):
            print("⚠️ Non-forensic sketch → normalizing")
            sketch = normalize_to_cufs(sketch)
        else:
            print("✔ Forensic-style sketch detected")
    else:
        print("✔ CelebA mode – style check skipped")

    # ---------- STEP 3 ----------
    if USE_GENERATION:
        print("▶ STEP 3: Generating photo (CUFS Pix2Pix)")
        generated = generate_face_from_sketch(sketch, gallery_type="cufs")

        # Light enhancement (CUFS only)
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]) * 0.5
        generated = cv2.filter2D(generated, -1, kernel)
        generated = cv2.bilateralFilter(generated, 5, 50, 50)

    else:
        print("▶ STEP 3: Skipping generation (CelebA direct matching)")
        gray = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        generated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    gen_path = GEN_DIR / f"generated_{ts}.jpg"
    cv2.imwrite(str(gen_path), generated)
    print("✔ Image saved:", gen_path)

    # ---------- STEP 4 ----------
    print("▶ STEP 4: Face detection")
    face = extract_face(generated)

    if face is None:
        print("⚠️ Face detector failed → fallback center crop")
        face = fallback_center_crop(generated)
    else:
        face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)

    print("✔ Face ready for embedding")

    # ---------- STEP 5 ----------
    print("▶ STEP 5: Generating embedding")
    query_emb = get_embedding(face)
    query_emb = query_emb / np.linalg.norm(query_emb)

    # ---------- STEP 6 ----------
    print("▶ STEP 6: Matching against gallery")
    sims = gallery @ query_emb
    ranked = np.argsort(-sims)

    top1, top2 = ranked[0], ranked[1]
    score1, score2 = float(sims[top1]), float(sims[top2])
    margin = score1 - score2

    print("\n📊 MATCH ANALYSIS")
    print(f"Top-1 similarity: {score1:.4f}")
    print(f"Top-2 similarity: {score2:.4f}")
    print(f"Top-3 similarity: {float(sims[ranked[2]]):.4f}")
    print(f"Top-4 similarity: {float(sims[ranked[3]]):.4f}")
    print(f"Top-5 similarity: {float(sims[ranked[4]]):.4f}")
    print(f"Confidence margin: {margin:.4f}")

    # ---------- STEP 6.5: Decision Intelligence ----------
    print("\n🧠 STEP 6.5: Decision Intelligence Analysis")
    
    # Get top-K scores for reliability computation
    top_k_scores = [float(sims[ranked[i]]) for i in range(min(TOP_K, len(ranked)))]
    
    # 1. Match Reliability Scoring (MRS)
    reliability = compute_match_reliability(
        top1_score=score1,
        top2_score=score2,
        top_k_scores=top_k_scores,
        gallery_name=gallery_name
    )
    
    # 2. Gallery Density Awareness (GDA)
    density = compute_gallery_density(
        query_emb=query_emb,
        gallery_embs=gallery,
        k=50
    )
    
    # 3. Cross-Gallery Consistency Check (CGCC)
    celeba_gallery_emb, cufs_gallery_emb = load_all_galleries()
    consistency = cross_gallery_consistency(
        query_emb=query_emb,
        celeba_gallery=celeba_gallery_emb,
        cufs_gallery=cufs_gallery_emb
    )
    
    # Display Decision Intelligence Report
    print("\n" + "━" * 60)
    print("🧠 DECISION INTELLIGENCE REPORT")
    print("━" * 60)
    print(f"Reliability Score : {reliability['reliability_score']} ({reliability['level']})")
    for explanation in reliability['explanation'][:3]:  # Show top 3 explanations
        print(f"  • {explanation}")
    
    print(f"\nGallery Density   : {density['risk_level']}")
    print(f"  {density['message']}")
    
    print(f"\nCross Consistency : {consistency['verdict']}")
    print(f"  {consistency['message']}")
    
    # Apply safer acceptance rule (ChatGPT's safer version)
    # ACCEPT if:
    #   - reliability_score >= 70
    #   - density_risk in ["LOW", "MEDIUM"]
    #   - consistency != "INCONSISTENT" (or unavailable)
    reliability_ok = reliability['reliability_score'] >= 70
    density_ok = density['risk_level'] in ["LOW", "MEDIUM"]
    consistency_ok = consistency['verdict'] != "INCONSISTENT"  # Accepts CONSISTENT or UNAVAILABLE
    
    accepted = reliability_ok and density_ok and consistency_ok
    
    print("\n" + "━" * 60)
    if accepted:
        print("FINAL DECISION: ACCEPTED ✅")
        print("\n✅ RESULT: HIGH CONFIDENCE MATCH")
    else:
        print("FINAL DECISION: LOW CONFIDENCE ⚠️")
        reasons = []
        if not reliability_ok:
            reasons.append(f"Reliability score too low ({reliability['reliability_score']} < 70)")
        if not density_ok:
            reasons.append(f"High density risk ({density['risk_level']})")
        if not consistency_ok:
            reasons.append(f"Inconsistent across galleries ({consistency['verdict']})")
        if reasons:
            print("Reasons:", "; ".join(reasons))
        print("\n⚠️ RESULT: LOW CONFIDENCE MATCH – showing top-K candidates anyway")
    print("━" * 60)

    # ---------- STEP 7 ----------
    print("▶ STEP 7: Creating Top-K visualization")
    topk_path = TOPK_DIR / f"topk_{ts}.jpg"

    visualize_topk(
        query_img_path=gen_path,
        ranked_indices=ranked,
        scores=sims,
        index=index,
        gallery_dir=GALLERY_DIR,
        out_path=topk_path,
        accepted_index=top1 if accepted else None
    )

    # ---------- STEP 8 ----------
    print("▶ STEP 8: Generating explainability heatmap")
    heatmap_path = save_heatmap_case(
        face_img=face,
        score=score1,
        tag=f"{gallery_name.upper()}_{index[top1]}"
    )

    # ---------- STEP 9 ----------
    final_out = OUT_ROOT / f"FINAL_DEMO_{ts}.jpg"

    combine_final_results(
        sketch_path=sketch_path,
        generated_path=gen_path,
        topk_path=topk_path,
        heatmap_path=heatmap_path,
        out_path=final_out
    )

    print(f"\n🖼️ FINAL OUTPUT SAVED: {final_out}")
    print("\n🎉 SYSTEM EXECUTION COMPLETE")


if __name__ == "__main__":
    main()
