"""
Show detailed information about the paired dataset.
Helps identify if pairs are correctly matched.
"""

from pathlib import Path

# Paths
PAIRS_FILE = Path("data/processed/pairs/cufs.txt")
RAW_PHOTO_DIR = Path("data/raw/cufs/photos")
RAW_SKETCH_DIR = Path("data/raw/cufs/sketches")
TRAIN_PHOTO_DIR = Path("data/pix2pix/train/photo")
TRAIN_SKETCH_DIR = Path("data/pix2pix/train/sketch")

print("=" * 60)
print("PAIRING VERIFICATION REPORT")
print("=" * 60)

# Load pairing file
pairing_map = {}
pairing_list = []
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
                pairing_list.append((photo_name, sketch_name))

print(f"\nTotal pairs in pairing file: {len(pairing_list)}")

# Check first 10 pairs
print("\nFirst 10 pairs from pairing file:")
print("-" * 60)
for i, (photo_name, sketch_name) in enumerate(pairing_list[:10], 1):
    raw_photo = RAW_PHOTO_DIR / photo_name
    raw_sketch = RAW_SKETCH_DIR / sketch_name
    
    photo_exists = raw_photo.exists()
    sketch_exists = raw_sketch.exists()
    
    status = "[OK]" if (photo_exists and sketch_exists) else "[MISSING]"
    print(f"{i:2d}. {status} {photo_name:20s} <-> {sketch_name:25s}")
    if not photo_exists:
        print(f"     Photo file not found!")
    if not sketch_exists:
        print(f"     Sketch file not found!")

# Check what's actually in train directories
print("\n" + "=" * 60)
print("What's in train directories (first 5):")
print("-" * 60)

train_photos = sorted(TRAIN_PHOTO_DIR.glob("*.jpg"))[:5]
train_sketches = sorted(TRAIN_SKETCH_DIR.glob("*.jpg"))[:5]

for i, (train_photo_path, train_sketch_path) in enumerate(zip(train_photos, train_sketches), 1):
    print(f"\nTrain pair {i}: {train_photo_path.name}")
    
    # Try to identify source by checking file sizes
    train_photo_size = train_photo_path.stat().st_size
    train_sketch_size = train_sketch_path.stat().st_size
    
    # Find matching files
    matching_photo = None
    matching_sketch = None
    
    for raw_photo in RAW_PHOTO_DIR.glob("*.jpg"):
        if abs(raw_photo.stat().st_size - train_photo_size) < 100:  # Allow small differences
            matching_photo = raw_photo.name
            break
    
    for raw_sketch in RAW_SKETCH_DIR.glob("*.jpg"):
        if abs(raw_sketch.stat().st_size - train_sketch_size) < 100:
            matching_sketch = raw_sketch.name
            break
    
    if matching_photo and matching_sketch:
        print(f"  Source photo: {matching_photo}")
        print(f"  Source sketch: {matching_sketch}")
        
        # Check if this matches pairing file
        if matching_photo in pairing_map:
            expected_sketch = pairing_map[matching_photo]
            if matching_sketch == expected_sketch:
                print(f"  Status: [CORRECT] Matches pairing file")
            else:
                print(f"  Status: [MISMATCH] Expected {expected_sketch}, got {matching_sketch}")
        else:
            print(f"  Status: [UNKNOWN] Photo not in pairing file")
    else:
        print(f"  Status: [ERROR] Could not identify source files")

print("\n" + "=" * 60)
print("To visually verify pairs, open:")
print("  data/pix2pix/train_AB/ - Combined side-by-side images")
print("  data/pix2pix/test_AB/ - Combined side-by-side images")
print("=" * 60)
