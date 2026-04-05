from pathlib import Path

def index(name, photo_dir, sketch_dir, out_file):
    photos = {p.name: p for p in Path(photo_dir).glob("*")}
    sketches = {s.name: s for s in Path(sketch_dir).glob("*")}

    keys = sorted(set(photos) & set(sketches))

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        for k in keys:
            f.write(f"{photos[k]} {sketches[k]}\n")

    print(f"{name}: {len(keys)} pairs indexed")

index(
    "CUFS",
    "data/processed/cufs/photos",
    "data/processed/cufs/sketches",
    "data/processed/pairs/cufs.txt"
)

index(
    "FS2K",
    "data/processed/fs2k/photos",
    "data/processed/fs2k/sketches",
    "data/processed/pairs/fs2k.txt"
)
