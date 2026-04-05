import cv2
from pathlib import Path

PHOTO_DIR = Path("data/raw/cufs/photos")
SKETCH_DIR = Path("data/raw/cufs/sketches")

OUT_TRAIN = Path("pytorch-CycleGAN-and-pix2pix/datasets/cufs_pix2pix/train")
OUT_TEST  = Path("pytorch-CycleGAN-and-pix2pix/datasets/cufs_pix2pix/test")

OUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUT_TEST.mkdir(parents=True, exist_ok=True)

photos = sorted(PHOTO_DIR.glob("*.jpg"))
sketches = {s.name: s for s in SKETCH_DIR.glob("*.jpg")}

TRAIN_COUNT = 150
TEST_COUNT  = 38

train_ct = 0
test_ct = 0
skip_ct = 0

for photo_path in photos:
    name = photo_path.name  # SAME name for sketch

    if name not in sketches:
        skip_ct += 1
        continue

    sketch_path = sketches[name]

    photo = cv2.imread(str(photo_path))
    sketch = cv2.imread(str(sketch_path))

    if photo is None or sketch is None:
        skip_ct += 1
        continue

    photo = cv2.resize(photo, (256, 256))
    sketch = cv2.resize(sketch, (256, 256))

    pair = cv2.hconcat([sketch, photo])

    if train_ct < TRAIN_COUNT:
        cv2.imwrite(str(OUT_TRAIN / name), pair)
        train_ct += 1
    elif test_ct < TEST_COUNT:
        cv2.imwrite(str(OUT_TEST / name), pair)
        test_ct += 1
    else:
        break

print(f"✅ Train pairs: {train_ct}")
print(f"✅ Test pairs: {test_ct}")
print(f"⚠️ Skipped pairs: {skip_ct}")
