from pathlib import Path
import shutil

RAW = Path("data/raw/fs2k")
OUT_P = Path("data/processed/fs2k/photos")
OUT_S = Path("data/processed/fs2k/sketches")

OUT_P.mkdir(parents=True, exist_ok=True)
OUT_S.mkdir(parents=True, exist_ok=True)

counter = 0

photo_root = RAW / "photo"
sketch_root = RAW / "sketch"

photo_folders = sorted(photo_root.iterdir())
sketch_folders = sorted(sketch_root.iterdir())

for p_dir, s_dir in zip(photo_folders, sketch_folders):
    photos = sorted(p_dir.glob("*"))
    sketches = sorted(s_dir.glob("*"))

    min_len = min(len(photos), len(sketches))

    for i in range(min_len):
        counter += 1
        name = f"fs2k_{counter:05d}.jpg"

        shutil.copy(photos[i], OUT_P / name)
        shutil.copy(sketches[i], OUT_S / name)

print("✅ FS2K flatten completed")
print("Total identity pairs:", counter)
