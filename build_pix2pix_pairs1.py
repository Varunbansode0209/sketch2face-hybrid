import cv2
from pathlib import Path

PHOTO_DIR = Path("data/raw/cufs/photos")
SKETCH_DIR = Path("data/raw/cufs/sketches")

OUT_TRAIN = Path("pytorch-CycleGAN-and-pix2pix/datasets/cufs_pix2pix/train")
OUT_TEST  = Path("pytorch-CycleGAN-and-pix2pix/datasets/cufs_pix2pix/test")

OUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUT_TEST.mkdir(parents=True, exist_ok=True)

photos = sorted(PHOTO_DIR.glob("*.jpg"))

split = int(0.8 * len(photos))

count = 0
for i, photo_path in enumerate(photos):
    name = photo_path.stem
    sketch_path = SKETCH_DIR / f"{name}-sz1.jpg"

    if not sketch_path.exists():
        continue

    photo = cv2.imread(str(photo_path))
    sketch = cv2.imread(str(sketch_path))

    if photo is None or sketch is None:
        continue

    photo = cv2.resize(photo, (256, 256))
    sketch = cv2.resize(sketch, (256, 256))

    pair = cv2.hconcat([sketch, photo])

    out_dir = OUT_TRAIN if i < split else OUT_TEST
    cv2.imwrite(str(out_dir / f"{name}.jpg"), pair)

    count += 1

print(f"✅ Pix2Pix pairs created: {count}")
