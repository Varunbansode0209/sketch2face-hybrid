import json
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
GALLERY_PATH = Path("embeddings/gallery/fs2k_gallery.npy")
INDEX_PATH   = Path("embeddings/index.json")
OUT_DIR      = Path("processed/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 3000
# ---------------------------------------


print("▶ Loading embeddings...")
embeddings = np.load(GALLERY_PATH)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

with open(INDEX_PATH, "r") as f:
    index = json.load(f)

N = embeddings.shape[0]
print("Total identities:", N)

rng = np.random.default_rng(42)
confidence_margins = []

print("▶ Computing confidence margins...")

for _ in range(NUM_SAMPLES):
    q = rng.integers(0, N)
    query_emb = embeddings[q]

    sims = embeddings @ query_emb
    ranked = np.argsort(-sims)

    top1 = sims[ranked[1]]   # skip self-match
    top2 = sims[ranked[2]]

    margin = float(top1 - top2)
    confidence_margins.append(margin)

confidence_margins = np.array(confidence_margins)

out_path = OUT_DIR / "confidence_margins.npy"
np.save(out_path, confidence_margins)

print(f"✅ Saved confidence margins: {out_path}")
print("Mean margin:", confidence_margins.mean())
