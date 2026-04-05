import cv2
import numpy as np
from pathlib import Path

from src.embedding.arcface_infer import get_embedding

# ---------------- CONFIG ----------------
IMG_PATH = Path("processed/generated/generated_face.jpg")
OUT_PATH = Path("processed/explainability/heatmap.jpg")

GRID_SIZE = 16        # 8x8 grid
MASK_VALUE = 0       # black mask
# ---------------------------------------


def compute_similarity(a, b):
    return float(np.dot(a, b))


def generate_heatmap(image):
    h, w, _ = image.shape
    heatmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    base_emb = get_embedding(image)

    cell_h = h // GRID_SIZE
    cell_w = w // GRID_SIZE

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            masked = image.copy()

            y1 = i * cell_h
            y2 = (i + 1) * cell_h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w

            mask = np.ones_like(image, dtype=np.float32)

            cv2.rectangle(
                mask,
                (x1, y1),
                (x2, y2),
                (0.2, 0.2, 0.2),  # soft attenuation instead of zero
                thickness=-1
            )

            masked = (image.astype(np.float32) * mask).astype(np.uint8)


            emb = get_embedding(masked)
            sim = compute_similarity(base_emb, emb)

            heatmap[i, j] = 1.0 - sim  # similarity drop

    heatmap = cv2.resize(heatmap, (w, h))
    p95 = np.percentile(heatmap, 95)
    heatmap = np.clip(heatmap / p95, 0, 1)


    return heatmap


def overlay_heatmap(image, heatmap):
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )
    return cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)


if __name__ == "__main__":
    img = cv2.imread(str(IMG_PATH))
    assert img is not None, "❌ Failed to load image"

    heatmap = generate_heatmap(img)
    result = overlay_heatmap(img, heatmap)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), result)

    print("✅ Grad-CAM–style heatmap saved:", OUT_PATH)
