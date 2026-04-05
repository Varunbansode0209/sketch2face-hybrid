from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

PHOTO_DIR  = Path("data/raw/cufs/photos")
SKETCH_DIR = Path("data/raw/cufs/sketches")

OUT_BASE = Path("data/pix2pix/cufs")
TRAIN_RATIO = 0.9

# Output dirs
train_photo  = OUT_BASE / "train/photos"
train_sketch = OUT_BASE / "train/sketches"
val_photo    = OUT_BASE / "val/photos"
val_sketch   = OUT_BASE / "val/sketches"

for d in [train_photo, train_sketch, val_photo, val_sketch]:
    d.mkdir(parents=True, exist_ok=True)

# Collect matched pairs
photos = {p.name: p for p in PHOTO_DIR.glob("*.jpg")}
sketches = {s.name: s for s in SKETCH_DIR.glob("*.jpg")}

common = sorted(set(photos.keys()) & set(sketches.keys()))

print(f"🔍 Total matched pairs found: {len(common)}")

assert len(common) > 0, "❌ No matched pairs found"

# Train / Val split
train_ids, val_ids = train_test_split(
    common, train_size=TRAIN_RATIO, random_state=42
)

def copy_pairs(ids, p_out, s_out):
    for name in ids:
        shutil.copy(photos[name], p_out / name)
        shutil.copy(sketches[name], s_out / name)

copy_pairs(train_ids, train_photo, train_sketch)
copy_pairs(val_ids, val_photo, val_sketch)

print(f"✅ Train pairs: {len(train_ids)}")
print(f"✅ Val pairs: {len(val_ids)}")
