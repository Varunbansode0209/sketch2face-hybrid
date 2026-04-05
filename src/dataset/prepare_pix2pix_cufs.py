"""
Prepare CUFS dataset for pix2pix training.
- Selects 100-200 photos/sketches for train (using 180)
- Selects 20 photos/sketches for test
- Ensures matching filenames between photo and sketch pairs
- Sources from data/raw/cufs/photos and data/raw/cufs/sketches (NOT original_sketch)
- Uses correct pairing from data/processed/pairs/cufs.txt
"""

from pathlib import Path
import shutil
import random

# Paths
RAW_PHOTO_DIR = Path("data/raw/cufs/photos")
RAW_SKETCH_DIR = Path("data/raw/cufs/sketches")
# Try fixed pairing file first, fall back to original
PAIRS_FILE = Path("data/processed/pairs/cufs_fixed.txt")
if not PAIRS_FILE.exists():
    PAIRS_FILE = Path("data/processed/pairs/cufs.txt")

TRAIN_PHOTO_DIR = Path("data/pix2pix/train/photo")
TRAIN_SKETCH_DIR = Path("data/pix2pix/train/sketch")
TEST_PHOTO_DIR = Path("data/pix2pix/test/photo")
TEST_SKETCH_DIR = Path("data/pix2pix/test/sketch")

# Configuration
TRAIN_COUNT = 180  # Within 100-200 range
TEST_COUNT = 20

# Create output directories
TRAIN_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_SKETCH_DIR.mkdir(parents=True, exist_ok=True)
TEST_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
TEST_SKETCH_DIR.mkdir(parents=True, exist_ok=True)

# Load correct pairs from pairing file
print("Loading correct photo-sketch pairs from pairing file...")
pairs = []
if PAIRS_FILE.exists():
    with open(PAIRS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # Extract filenames from processed paths
                processed_photo = Path(parts[0])
                processed_sketch = Path(parts[1])
                
                # Map to raw directory filenames
                photo_name = processed_photo.name
                sketch_name = processed_sketch.name
                
                # Find corresponding files in raw directory
                raw_photo = RAW_PHOTO_DIR / photo_name
                raw_sketch = RAW_SKETCH_DIR / sketch_name
                
                if raw_photo.exists() and raw_sketch.exists():
                    pairs.append((raw_photo, raw_sketch))
                else:
                    print(f"Warning: Pair not found in raw directory: {photo_name} <-> {sketch_name}")
else:
    print(f"Error: Pairing file not found: {PAIRS_FILE}")
    print("Falling back to alphabetical matching (may be incorrect)")
    photos = sorted(RAW_PHOTO_DIR.glob("*.jpg"))
    sketches = sorted(RAW_SKETCH_DIR.glob("*.jpg"))
    min_count = min(len(photos), len(sketches))
    pairs = list(zip(photos[:min_count], sketches[:min_count]))

print(f"Found {len(pairs)} correct photo-sketch pairs")

# Ensure we have enough pairs
if len(pairs) < TRAIN_COUNT + TEST_COUNT:
    print(f"Warning: Only {len(pairs)} pairs available, but need {TRAIN_COUNT + TEST_COUNT}")
    TRAIN_COUNT = len(pairs) - TEST_COUNT
    if TRAIN_COUNT < 100:
        print(f"Error: Not enough pairs. Need at least {TEST_COUNT + 100}, got {len(pairs)}")
        exit(1)

# Shuffle for randomness
random.seed(42)  # For reproducibility
random.shuffle(pairs)

# Split into train and test
train_pairs = pairs[:TRAIN_COUNT]
test_pairs = pairs[TRAIN_COUNT:TRAIN_COUNT + TEST_COUNT]

print(f"\nSplit:")
print(f"  Train: {len(train_pairs)} pairs")
print(f"  Test: {len(test_pairs)} pairs")

# Copy train pairs with matching names
print(f"\nCopying train pairs...")
print("Sample pairings (first 3):")
for idx, (photo_path, sketch_path) in enumerate(train_pairs[:3], 1):
    print(f"  Pair {idx}: {photo_path.name} <-> {sketch_path.name}")

for idx, (photo_path, sketch_path) in enumerate(train_pairs, 1):
    # Use same filename for both photo and sketch
    name = f"cufs_{idx:04d}.jpg"
    
    shutil.copy(photo_path, TRAIN_PHOTO_DIR / name)
    shutil.copy(sketch_path, TRAIN_SKETCH_DIR / name)

# Copy test pairs with matching names
print(f"\nCopying test pairs...")
print("Sample pairings (first 3):")
for idx, (photo_path, sketch_path) in enumerate(test_pairs[:3], 1):
    print(f"  Pair {idx}: {photo_path.name} <-> {sketch_path.name}")

for idx, (photo_path, sketch_path) in enumerate(test_pairs, 1):
    # Use same filename for both photo and sketch
    name = f"cufs_{idx:04d}.jpg"
    
    shutil.copy(photo_path, TEST_PHOTO_DIR / name)
    shutil.copy(sketch_path, TEST_SKETCH_DIR / name)

print(f"\nDone!")
print(f"  Train: {TRAIN_PHOTO_DIR} ({len(list(TRAIN_PHOTO_DIR.glob('*.jpg')))} photos)")
print(f"         {TRAIN_SKETCH_DIR} ({len(list(TRAIN_SKETCH_DIR.glob('*.jpg')))} sketches)")
print(f"  Test:  {TEST_PHOTO_DIR} ({len(list(TEST_PHOTO_DIR.glob('*.jpg')))} photos)")
print(f"         {TEST_SKETCH_DIR} ({len(list(TEST_SKETCH_DIR.glob('*.jpg')))} sketches)")
