"""
Automatically fix CUFS pairing by matching sketches to photos using:
1. Pix2Pix to convert sketch to photo
2. ArcFace to get embeddings
3. Find best matching photo for each sketch
"""

from pathlib import Path
import cv2
import numpy as np
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple progress bar replacement
    def tqdm(iterable, desc=""):
        return iterable

from src.embedding.arcface_infer import get_embedding
from src.preprocess.detect_face import detect_faces
from src.generation.pix2pix_infer import generate_face_from_sketch

RAW_PHOTO_DIR = Path("data/raw/cufs/photos")
RAW_SKETCH_DIR = Path("data/raw/cufs/sketches")
OUT_PAIRS_FILE = Path("data/processed/pairs/cufs_fixed.txt")

def extract_face(img):
    """Extract face from image"""
    boxes = detect_faces(img)
    if not boxes:
        return None
    x1, y1, x2, y2 = boxes[0]
    return img[y1:y2, x1:x2]

def get_photo_embedding(photo_path):
    """Get embedding for a photo"""
    try:
        img = cv2.imread(str(photo_path))
        if img is None:
            return None
        face = extract_face(img)
        if face is None:
            return None
        return get_embedding(face)
    except Exception as e:
        return None

def get_sketch_embedding(sketch_path):
    """Get embedding for a sketch by converting to photo first"""
    try:
        sketch = cv2.imread(str(sketch_path))
        if sketch is None:
            return None
        
        # Convert sketch to photo using Pix2Pix
        generated_photo = generate_face_from_sketch(sketch)
        
        # Extract face and get embedding
        face = extract_face(generated_photo)
        if face is None:
            return None
        
        return get_embedding(face)
    except Exception as e:
        return None

print("=" * 60)
print("AUTOMATIC CUFS PAIRING FIX")
print("=" * 60)

# Load all photos and sketches
photos = sorted(RAW_PHOTO_DIR.glob("*.jpg"))
sketches = sorted(RAW_SKETCH_DIR.glob("*.jpg"))

print(f"\nFound {len(photos)} photos and {len(sketches)} sketches")

# Step 1: Generate embeddings for all photos
print("\nStep 1: Generating embeddings for all photos...")
photo_embeddings = {}
for photo_path in tqdm(photos, desc="Processing photos"):
    emb = get_photo_embedding(photo_path)
    if emb is not None:
        photo_embeddings[photo_path] = emb

print(f"Successfully generated {len(photo_embeddings)} photo embeddings")

# Step 2: For each sketch, find best matching photo
print("\nStep 2: Matching sketches to photos...")
print("(This will take a while - converting sketches to photos and computing similarities)")

pairs = []
photo_emb_array = np.array(list(photo_embeddings.values()))
photo_paths_list = list(photo_embeddings.keys())

for sketch_path in tqdm(sketches, desc="Matching sketches"):
    sketch_emb = get_sketch_embedding(sketch_path)
    if sketch_emb is None:
        continue
    
    # Compute similarities with all photos
    similarities = photo_emb_array @ sketch_emb
    
    # Find best match
    best_idx = np.argmax(similarities)
    best_photo_path = photo_paths_list[best_idx]
    best_similarity = similarities[best_idx]
    
    pairs.append((best_photo_path, sketch_path, best_similarity))

print(f"\nMatched {len(pairs)} sketch-photo pairs")

# Step 3: Save pairs to file
print(f"\nStep 3: Saving pairs to {OUT_PAIRS_FILE}")
OUT_PAIRS_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_PAIRS_FILE, "w") as f:
    for photo_path, sketch_path, similarity in pairs:
        f.write(f"{photo_path} {sketch_path}\n")

print(f"Saved {len(pairs)} pairs")
print(f"\nAverage similarity: {np.mean([s for _, _, s in pairs]):.4f}")
print(f"Min similarity: {np.min([s for _, _, s in pairs]):.4f}")
print(f"Max similarity: {np.max([s for _, _, s in pairs]):.4f}")

print("\n" + "=" * 60)
print("DONE! You can now use data/processed/pairs/cufs_fixed.txt")
print("Update prepare_pix2pix_cufs.py to use this file instead.")
print("=" * 60)
