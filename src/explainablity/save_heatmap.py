import cv2
import numpy as np
from pathlib import Path

OUT_DIR = Path("processed/final_demo/heatmaps")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_heatmap(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)

    heatmap = cv2.GaussianBlur(gray, (31, 31), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap


def overlay_heatmap(face_img, heatmap, alpha=0.5):
    return cv2.addWeighted(face_img, 1 - alpha, heatmap, alpha, 0)


def save_heatmap_case(*, face_img, score, tag):
    """
    Keyword-only arguments enforced for safety
    """
    heatmap = generate_heatmap(face_img)
    overlay = overlay_heatmap(face_img, heatmap)

    cv2.putText(
        overlay,
        f"{tag} | Similarity: {score:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    out_path = OUT_DIR / f"{tag}_heatmap.jpg"
    cv2.imwrite(str(out_path), overlay)

    print(f"✔ Heatmap saved: {out_path}")
    return out_path
