import cv2
import numpy as np
from pathlib import Path

TOP_K = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX


def visualize_topk(
    query_img_path,
    ranked_indices,
    scores,
    index,
    gallery_dir,
    out_path,
    accepted_index=None   # ✅ NEW
):
    # ---------- Load Query ----------
    query = cv2.imread(str(query_img_path))
    if query is None:
        print("❌ Failed to load query image for visualization")
        return

    query = cv2.resize(query, (250, 250))
    cv2.putText(
        query,
        "QUERY",
        (10, 30),
        FONT,
        0.9,
        (0, 0, 255),
        2
    )

    panels = [query]

    # ---------- Load Top-K Matches ----------
    count = 0
    for rank, idx in enumerate(ranked_indices):
        if count >= TOP_K:
            break

        img_name = index[idx]
        img_path = gallery_dir / img_name
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"⚠️ Could not load gallery image: {img_name}")
            continue

        img = cv2.resize(img, (250, 250))
        score = scores[idx]

        # ---------- BORDER LOGIC ----------
        if accepted_index is not None and idx == accepted_index:
            border_color = (0, 255, 0)   # GREEN → final match
            border_thick = 6
        else:
            border_color = (255, 255, 255)
            border_thick = 1

        cv2.rectangle(
            img,
            (0, 0),
            (249, 249),
            border_color,
            border_thick
        )

        label = f"{rank+1}: {img_name}"
        cv2.putText(
            img,
            label,
            (10, 25),
            FONT,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            img,
            f"{score:.3f}",
            (10, 50),
            FONT,
            0.8,
            border_color,
            2
        )

        panels.append(img)
        count += 1

    # ---------- Save Result ----------
    if len(panels) > 1:
        result = np.hstack(panels)
        cv2.imwrite(str(out_path), result)
    else:
        print("⚠️ No gallery images available for visualization")
