from pathlib import Path

PHOTO_DIR  = Path("data/raw/cufs/photos")
SKETCH_DIR = Path("data/raw/cufs/sketches")

photos = {p.stem for p in PHOTO_DIR.glob("*.jpg")}

renamed = 0
skipped = 0

for sketch in SKETCH_DIR.glob("*.jpg"):
    base = sketch.stem.replace("-sz1", "")

    if base in photos:
        new_path = SKETCH_DIR / f"{base}.jpg"
        if sketch.name != new_path.name:
            sketch.rename(new_path)
            renamed += 1
    else:
        skipped += 1

print(f"✅ Renamed sketches: {renamed}")
print(f"⚠️ Skipped sketches: {skipped}")
