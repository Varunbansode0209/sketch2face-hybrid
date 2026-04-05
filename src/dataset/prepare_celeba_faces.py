import cv2
from pathlib import Path
from src.preprocess.detect_face import detect_faces

SRC = Path("data/raw/celeba/img_align_celeba")
DST = Path("data/raw/celeba/photos")
DST.mkdir(parents=True, exist_ok=True)

saved = 0
skipped = 0

for img_path in SRC.glob("*.jpg"):
    img = cv2.imread(str(img_path))
    if img is None:
        skipped += 1
        continue

    h, w = img.shape[:2]
    boxes = detect_faces(img)

    if not boxes:
        skipped += 1
        continue

    x1, y1, x2, y2 = boxes[0]

    # 🔒 Clamp box to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        skipped += 1
        continue

    face = img[y1:y2, x1:x2]

    # 🔒 Final validation
    if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
        skipped += 1
        continue

    face = cv2.resize(face, (112, 112))
    cv2.imwrite(str(DST / img_path.name), face)
    saved += 1

print(f"✅ CelebA face crops saved: {saved}")
print(f"⚠️ Skipped images: {skipped}")
