import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.embedding.arcface_infer import preprocess_face, session, input_name

SRC = Path("data/raw/celeba/photos")
OUT_DIR = Path("embeddings/gallery")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8        # CPU-safe
MAX_IMAGES = 40000    # ⬅️ CHANGE THIS (30k–50k recommended)

images = []
index = []

print("▶ Loading CelebA face images...")

for img_path in SRC.glob("*.jpg"):
    if len(images) >= MAX_IMAGES:
        break

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    images.append(preprocess_face(img))
    index.append(img_path.name)

print(f"✔ Loaded {len(images)} images")

print("▶ Running ArcFace inference (CPU, batched)...")

embeddings = []

for i in tqdm(range(0, len(images), BATCH_SIZE)):
    batch = np.concatenate(images[i:i+BATCH_SIZE], axis=0)
    emb = session.run(None, {input_name: batch})[0]
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    embeddings.append(emb)

embeddings = np.vstack(embeddings)

np.save(OUT_DIR / "celeba_gallery.npy", embeddings)
with open(OUT_DIR / "celeba_index.json", "w") as f:
    json.dump(index, f)

print("✅ CelebA embeddings saved")
print("Final gallery size:", embeddings.shape)
