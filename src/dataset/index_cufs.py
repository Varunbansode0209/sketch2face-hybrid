from pathlib import Path

photo_dir = Path("data/processed/cufs/photos")
sketch_dir = Path("data/processed/cufs/sketches")
out_file = Path("data/processed/pairs/cufs.txt")

photos = sorted(photo_dir.glob("*"))
sketches = sorted(sketch_dir.glob("*"))

n = min(len(photos), len(sketches))

out_file.parent.mkdir(parents=True, exist_ok=True)

with open(out_file, "w") as f:
    for i in range(n):
        f.write(f"{photos[i]} {sketches[i]}\n")

print("CUFS pairs indexed:", n)
