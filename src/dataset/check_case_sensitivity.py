"""
Check for case-sensitivity issues in pairing file.
Windows is case-insensitive but the pairing file might have case mismatches.
"""

from pathlib import Path

RAW_PHOTO_DIR = Path("data/raw/cufs/photos")
RAW_SKETCH_DIR = Path("data/raw/cufs/sketches")
PAIRS_FILE = Path("data/processed/pairs/cufs.txt")

print("Checking for case-sensitivity and file existence issues...\n")

issues = []
if PAIRS_FILE.exists():
    with open(PAIRS_FILE, "r") as f:
        for line_num, line in enumerate(f, 1):
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
                
                # Check if files exist (case-insensitive on Windows)
                photo_exists = raw_photo.exists()
                sketch_exists = raw_sketch.exists()
                
                # Also check case-sensitive match
                actual_photos = list(RAW_PHOTO_DIR.glob(photo_name.replace(".jpg", "*.jpg")))
                actual_sketches = list(RAW_SKETCH_DIR.glob(sketch_name.replace(".jpg", "*.jpg")))
                
                if not photo_exists:
                    issues.append(f"Line {line_num}: Photo not found: {photo_name}")
                elif actual_photos and actual_photos[0].name != photo_name:
                    issues.append(f"Line {line_num}: Photo case mismatch: expected {photo_name}, found {actual_photos[0].name}")
                
                if not sketch_exists:
                    issues.append(f"Line {line_num}: Sketch not found: {sketch_name}")
                elif actual_sketches and actual_sketches[0].name != sketch_name:
                    issues.append(f"Line {line_num}: Sketch case mismatch: expected {sketch_name}, found {actual_sketches[0].name}")

if issues:
    print(f"Found {len(issues)} issues:\n")
    for issue in issues[:20]:  # Show first 20
        print(f"  {issue}")
    if len(issues) > 20:
        print(f"  ... and {len(issues) - 20} more issues")
else:
    print("No case-sensitivity or file existence issues found!")
    print("All pairs in the pairing file have matching files in raw directories.")

print("\n" + "="*60)
print("If pairs still look wrong visually, the pairing file itself")
print("might have incorrect pairings. You may need to manually verify")
print("that each photo-sketch pair shows the same person.")
print("="*60)
