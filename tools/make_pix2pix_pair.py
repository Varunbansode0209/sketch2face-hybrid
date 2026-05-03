import cv2
from pathlib import Path

def create_paired_images(sketch_dir, photo_dir, out_dir):
    """Create side-by-side paired images (sketch|photo)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for sk in sketch_dir.glob("*.jpg"):
        ph = photo_dir / sk.name
        if not ph.exists():
            continue

        sketch = cv2.imread(str(sk))
        photo  = cv2.imread(str(ph))

        sketch = cv2.resize(sketch, (256, 256))
        photo  = cv2.resize(photo, (256, 256))

        ab = cv2.hconcat([sketch, photo])
        cv2.imwrite(str(out_dir / sk.name), ab)
        count += 1
    
    return count

# Create train pairs
train_sketch_dir = Path("data/pix2pix/train/sketch")
train_photo_dir  = Path("data/pix2pix/train/photo")
train_out_dir    = Path("data/pix2pix/train_AB")
train_count = create_paired_images(train_sketch_dir, train_photo_dir, train_out_dir)
print(f"Pix2Pix train pairs created: {train_count} images")

# Create test pairs
test_sketch_dir = Path("data/pix2pix/test/sketch")
test_photo_dir  = Path("data/pix2pix/test/photo")
test_out_dir    = Path("data/pix2pix/test_AB")
test_count = create_paired_images(test_sketch_dir, test_photo_dir, test_out_dir)
print(f"Pix2Pix test pairs created: {test_count} images")
