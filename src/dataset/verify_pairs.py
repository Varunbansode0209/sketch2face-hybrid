"""
Verify that photo-sketch pairs are correctly matched.
Shows which original files are paired together.
"""

from pathlib import Path
import cv2

# Paths
TRAIN_PHOTO_DIR = Path("data/pix2pix/train/photo")
TRAIN_SKETCH_DIR = Path("data/pix2pix/train/sketch")
RAW_PHOTO_DIR = Path("data/raw/cufs/photos")
RAW_SKETCH_DIR = Path("data/raw/cufs/sketches")
PAIRS_FILE = Path("data/processed/pairs/cufs.txt")

# Load pairing file to create a lookup
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
                photo_name = processed_photo.name
                sketch_name = processed_sketch.name
                pairing_map[photo_name] = sketch_name

print("Verifying first 5 pairs in train set:\n")

train_photos = sorted(TRAIN_PHOTO_DIR.glob("*.jpg"))[:5]
train_sketches = sorted(TRAIN_SKETCH_DIR.glob("*.jpg"))[:5]

for i, (train_photo, train_sketch) in enumerate(zip(train_photos, train_sketches), 1):
    print(f"Pair {i}: {train_photo.name}")
    
    # Read the actual image files to get their original content
    # We need to trace back which raw files were used
    # Since we copied them, we need to check the file content or metadata
    
    # Instead, let's check if the pairing file has the correct mapping
    # by looking at what SHOULD be paired
    
    print(f"  Train photo: {train_photo.name}")
    print(f"  Train sketch: {train_sketch.name}")
    
    # Check if we can find which raw files these came from
    # by checking file sizes or content
    train_photo_size = train_photo.stat().st_size
    train_sketch_size = train_sketch.stat().st_size
    
    # Try to find matching files in raw directory
    matching_photos = []
    matching_sketches = []
    
    for raw_photo in RAW_PHOTO_DIR.glob("*.jpg"):
        if raw_photo.stat().st_size == train_photo_size:
            matching_photos.append(raw_photo.name)
    
    for raw_sketch in RAW_SKETCH_DIR.glob("*.jpg"):
        if raw_sketch.stat().st_size == train_sketch_size:
            matching_sketches.append(raw_sketch.name)
    
    if matching_photos and matching_sketches:
        print(f"  Possible raw photo: {matching_photos[0]}")
        print(f"  Possible raw sketch: {matching_sketches[0]}")
        
        # Check if this is a correct pair according to pairing file
        if matching_photos[0] in pairing_map:
            expected_sketch = pairing_map[matching_photos[0]]
            if matching_sketches[0] == expected_sketch:
                print(f"  [OK] CORRECT PAIR (matches pairing file)")
            else:
                print(f"  [ERROR] MISMATCH! Expected: {expected_sketch}, Got: {matching_sketches[0]}")
        else:
            print(f"  ? Photo not found in pairing file")
    else:
        print(f"  ? Could not identify source files")
    
    print()
