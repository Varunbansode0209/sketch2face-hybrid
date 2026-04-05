import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===================== CONFIG =====================
GALLERY_PATH = Path("embeddings/gallery/fs2k_gallery.npy")
INDEX_PATH   = Path("embeddings/index.json")
OUT_DIR      = Path("processed/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 5000
THRESHOLD = 0.30
# =================================================


# ================= LOAD DATA ======================
print("▶ Loading gallery embeddings...")
embeddings = np.load(GALLERY_PATH)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
print("Embeddings shape:", embeddings.shape)

print("▶ Loading index...")
with open(INDEX_PATH, "r") as f:
    index = json.load(f)

N = embeddings.shape[0]
assert N == len(index), "❌ Index mismatch"
print("Total identities:", N)
# =================================================


# ========== IMPOSTOR SIMILARITY ===================
print("▶ Computing impostor similarity distribution...")

rng = np.random.default_rng(42)
impostor_scores = []

for _ in range(NUM_SAMPLES):
    i, j = rng.choice(N, size=2, replace=False)
    sim = float(np.dot(embeddings[i], embeddings[j]))
    impostor_scores.append(sim)

impostor_scores = np.array(impostor_scores)

print("Impostor samples:", len(impostor_scores))

# ✅ SAVE FOR HOUR-4
np.save(OUT_DIR / "impostor_scores.npy", impostor_scores)
print("✅ Saved impostor scores for FAR analysis")
# =================================================


# ========== HISTOGRAM PLOT ========================
plt.figure(figsize=(8, 5))
plt.hist(impostor_scores, bins=50, alpha=0.75, color="red")
plt.axvline(
    THRESHOLD,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {THRESHOLD}"
)

plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Impostor Similarity Distribution (Open-Set Evaluation)")
plt.legend()
plt.grid(True)

hist_path = OUT_DIR / "impostor_similarity_distribution.png"
plt.savefig(hist_path, dpi=200)
plt.close()

print(f"✅ Saved similarity histogram: {hist_path}")
# =================================================


# ========== THRESHOLD ANALYSIS ====================
accept_rate = np.mean(impostor_scores >= THRESHOLD)
reject_rate = 1.0 - accept_rate

print("\n📊 OPEN-SET THRESHOLD ANALYSIS")
print(f"Threshold: {THRESHOLD}")
print(f"Impostor Acceptance Rate (FAR): {accept_rate:.4f}")
print(f"Impostor Rejection Rate: {reject_rate:.4f}")

print("\n✅ Hour-2 evaluation completed successfully")
# =================================================
