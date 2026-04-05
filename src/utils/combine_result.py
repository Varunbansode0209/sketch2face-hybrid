import cv2
import numpy as np
from pathlib import Path


def combine_final_results(
    sketch_path: Path,
    generated_path: Path,
    topk_path: Path,
    heatmap_path: Path,
    out_path: Path
):
    images = []

    for p, title in [
        (sketch_path, "SKETCH"),
        (generated_path, "GENERATED"),
        (topk_path, "TOP-K MATCHES"),
        (heatmap_path, "HEATMAP"),
    ]:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Missing image: {p}")

        img = cv2.resize(img, (300, 300))
        cv2.putText(
            img,
            title,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        images.append(img)

    # Stack into 2x2 grid
    top_row = np.hstack(images[:2])
    bottom_row = np.hstack(images[2:])
    final = np.vstack([top_row, bottom_row])

    cv2.imwrite(str(out_path), final)
