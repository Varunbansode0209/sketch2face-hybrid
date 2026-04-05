import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from src.embedding.arcface_infer import get_embedding
from src.preprocess.detect_face import detect_faces
from src.matching.visualize_matches import visualize_topk
# from src.generation.pix2pix_infer import generate_face_from_sketch
from src.explainablity.save_heatmap import save_heatmap_case



# ===================== CONFIG =====================
GALLERY_PATH = Path("embeddings/gallery/fs2k_gallery.npy")
INDEX_PATH   = Path("embeddings/index.json")

SKETCH_PATH  = Path("data/query/f1-010-01-sz1.jpg")

GALLERY_DIR  = Path("data/processed/fs2k/photos")
GEN_DIR      = Path("processed/generated")
OUT_DIR      = Path("processed/results/topk")

TOP_K        = 5
THRESHOLD    = 0.30     # rejection threshold
CONF_MARGIN  = 0.05     # confidence margin
# ==================================================


# ================= LOAD GALLERY ===================
print("▶ Loading gallery embeddings...")
gallery_embeddings = np.load(GALLERY_PATH)
gallery_embeddings = gallery_embeddings / np.linalg.norm(
    gallery_embeddings, axis=1, keepdims=True
)
print("Gallery shape:", gallery_embeddings.shape)

print("▶ Loading index...")
with open(INDEX_PATH, "r") as f:
    index = json.load(f)

assert len(index) == gallery_embeddings.shape[0], "❌ Index mismatch"
print("Index size:", len(index))
# ==================================================


def extract_face(img):
    boxes = detect_faces(img)
    if boxes:
        x1, y1, x2, y2 = boxes[0]
        return img[y1:y2, x1:x2]
    return img   # fallback for generated faces


# ================= MAIN PIPELINE ==================
def match_query(sketch_path: Path):

    # ---- STEP 1: LOAD SKETCH ----
    print("\n▶ STEP 1: Loading sketch...")
    sketch = cv2.imread(str(sketch_path))
    if sketch is None:
        print("❌ Failed to load sketch")
        return

    # ---- STEP 2: SKETCH → PHOTO ----
    # print("▶ STEP 2: Generating photo using Pix2Pix...")
    # generated = generate_face_from_sketch(sketch)

    # GEN_DIR.mkdir(parents=True, exist_ok=True)
    # ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # gen_path = GEN_DIR / f"generated_{ts}.jpg"
    # cv2.imwrite(str(gen_path), generated)
    # print(f"✅ Generated image saved: {gen_path}")

    print("▶ STEP 2: Loading pre-generated photo...")
    gen_path = Path("processed/generated/generated_latest.jpg")
    generated = cv2.imread(str(gen_path))

    if generated is None:
        print("❌ Generated image not found")   
        return


    # ---- STEP 3: FACE EXTRACTION ----
    print("▶ STEP 3: Face detection on generated image...")
    face = extract_face(generated)

    # ---- STEP 4: ARC FACE EMBEDDING ----
    print("▶ STEP 4: Generating ArcFace embedding...")
    query_emb = get_embedding(face)
    query_emb = query_emb / np.linalg.norm(query_emb)

    # ---- STEP 5: SIMILARITY MATCHING ----
    print("▶ STEP 5: Computing cosine similarities...")
    sims = gallery_embeddings @ query_emb
    ranked = np.argsort(-sims)

    top1_idx = ranked[0]
    top2_idx = ranked[1]

    top1_score = float(sims[top1_idx])
    top2_score = float(sims[top2_idx])
    confidence_margin = top1_score - top2_score

    print("\n📈 CONFIDENCE ANALYSIS")
    print(f"Top-1 similarity : {top1_score:.4f}")
    print(f"Top-2 similarity : {top2_score:.4f}")
    print(f"Confidence margin: {confidence_margin:.4f}")

    # ---- STEP 6: REJECTION ----
    if top1_score < THRESHOLD:
        print("\n🚫 RESULT: NO RELIABLE MATCH FOUND")
        return

    if confidence_margin < CONF_MARGIN:
        print("\n🚫 RESULT: AMBIGUOUS MATCH")
        return

    print("\n✅ RESULT: HIGH CONFIDENCE MATCH")

    # ---- STEP 7: VISUALIZATION ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"topk_{ts}.jpg"

    visualize_topk(
        query_img_path=gen_path,
        ranked_indices=ranked,
        scores=sims,
        index=index,
        gallery_dir=GALLERY_DIR,
        out_path=out_path,
        accepted_index=top1_idx
    )

    print(f"✅ Top-K visualization saved: {out_path}")

        # ---- STEP 8: HEATMAP EXPLAINABILITY ----
    print("\n🔥 STEP 8: Saving explainability heatmaps...")

    # Load gallery images
    top1_img = cv2.imread(str(GALLERY_DIR / index[top1_idx]))
    top2_img = cv2.imread(str(GALLERY_DIR / index[top2_idx]))
    reject_idx = ranked[-1]
    reject_img = cv2.imread(str(GALLERY_DIR / index[reject_idx]))

    # Extract faces
    top1_face = extract_face(top1_img)
    top2_face = extract_face(top2_img)
    reject_face = extract_face(reject_img)

    # Save heatmaps
    save_heatmap_case(top1_face, top1_score, "Genuine")
    save_heatmap_case(top2_face, top2_score, "Impostor")
    save_heatmap_case(reject_face, float(sims[reject_idx]), "Rejected")


# ================= ENTRY ==========================
if __name__ == "__main__":
    match_query(SKETCH_PATH)
