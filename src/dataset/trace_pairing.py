"""
Trace exactly which files are being paired during the preparation process.
This simulates what prepare_pix2pix_cufs.py does.
"""

from pathlib import Path
import random

# Same paths as prepare script
RAW_PHOTO_DIR = Path("data/raw/cufs/photos")
RAW_SKETCH_DIR = Path("data/raw/cufs/sketches")
PAIRS_FILE = Path("data/processed/pairs/cufs.txt")

# Load correct pairs from pairing file (same logic as prepare script)
print("Loading pairs from pairing file...")
pairs = []
if PAIRS_FILE.exists():
    with open(PAIRS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                processed_photo = Path(parts[0])
                processed_sketch = Path(parts[1])
                
                photo_name = processed_photo.name
                sketch_name = processed_sketch.name
                
                raw_photo = RAW_PHOTO_DIR / photo_name
                raw_sketch = RAW_SKETCH_DIR / sketch_name
                
                if raw_photo.exists() and raw_sketch.exists():
                    pairs.append((raw_photo, raw_sketch))

print(f"Loaded {len(pairs)} pairs")

# Shuffle with same seed
random.seed(42)
random.shuffle(pairs)

# Show first 10 pairs that will be used
print("\nFirst 10 pairs after shuffling (these will become cufs_0001.jpg through cufs_0010.jpg):")
print("-" * 70)
for idx, (photo_path, sketch_path) in enumerate(pairs[:10], 1):
    print(f"cufs_{idx:04d}.jpg will contain:")
    print(f"  Photo:   {photo_path.name}")
    print(f"  Sketch:  {sketch_path.name}")
    
    # Verify this is correct according to pairing file
    pairing_map = {}
    if PAIRS_FILE.exists():
        with open(PAIRS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    processed_photo = Path(parts[0])
                    processed_sketch = Path(parts[1])
                    pairing_map[processed_photo.name] = processed_sketch.name
    
    expected_sketch = pairing_map.get(photo_path.name, "NOT FOUND")
    if sketch_path.name == expected_sketch:
        print(f"  Status: [CORRECT]")
    else:
        print(f"  Status: [ERROR] Expected {expected_sketch}, got {sketch_path.name}")
    print()
