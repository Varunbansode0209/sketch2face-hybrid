from pathlib import Path
import cv2
import numpy as np
import json
from src.embedding.arcface_infer import get_embedding

PHOTO_DIR = Path("data/processed/fs2k/photos")
OUT_DIR = Path("embeddings/gallery")
OUT_DIR.mkdir(parents=True, exist_ok=True)

embeddings = []
index = []

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
image_files = sorted([
    p for p in PHOTO_DIR.iterdir()
    if p.suffix.lower() in valid_ext
])

print(f"Found {len(image_files)} gallery images")

for img_path in image_files:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Skipped unreadable image: {img_path.name}")
        continue

    emb = get_embedding(img)

    # CRITICAL safety checks
    if emb.ndim != 1:
        print(f"❌ Invalid embedding shape for {img_path.name}: {emb.shape}")
        continue

    embeddings.append(emb.astype(np.float32))
    index.append(img_path.name)

# FINAL SAVE (this is where it went wrong earlier)
if len(embeddings) == 0:
    raise RuntimeError("❌ No embeddings generated")

embeddings = np.vstack(embeddings)

np.save(OUT_DIR / "fs2k_gallery.npy", embeddings)

with open("embeddings/index.json", "w") as f:
    json.dump(index, f, indent=2)

print(f"✅ Saved {embeddings.shape[0]} embeddings")
print(f"Embedding shape: {embeddings.shape}")
