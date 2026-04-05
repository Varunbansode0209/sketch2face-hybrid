import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from src.embedding.arcface_infer import get_embedding
from src.preprocess.detect_face import detect_faces

# -------- CONFIG --------
CUFS_PHOTO_DIR = Path("data/raw/cufs/photos")
OUT_PATH = Path("embeddings/gallery/cufs_gallery.npy")
INDEX_PATH = Path("embeddings/gallery/cufs_index.json")
# ------------------------


def extract_face(img):
    boxes = detect_faces(img)
    if boxes:
        x1, y1, x2, y2 = boxes[0]
        return img[y1:y2, x1:x2]
    return img


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    embeddings = []
    index = []

    photos = sorted(CUFS_PHOTO_DIR.glob("*.jpg"))
    print(f"▶ Found {len(photos)} CUFS photos")

    for img_path in tqdm(photos):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        face = extract_face(img)
        emb = get_embedding(face)

        embeddings.append(emb)
        index.append(img_path.name)

    embeddings = np.array(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    np.save(OUT_PATH, embeddings)

    import json
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)

    print(f"✅ Saved embeddings: {OUT_PATH}")
    print(f"✅ Total identities: {len(index)}")


if __name__ == "__main__":
    main()
