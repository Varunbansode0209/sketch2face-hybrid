import cv2
from pathlib import Path
from src.generation.pix2pix_infer import generate_face_from_sketch

SKETCH_DIR = Path("datasetzipped/CUFSF/cropped_sketch/cropped_sketch")

sketches = list(SKETCH_DIR.glob("*.jpg"))
assert len(sketches) > 0, "No sketches found"

img_path = sketches[0]
print("Using sketch:", img_path)

img = cv2.imread(str(img_path))
assert img is not None, "Failed to read sketch"

out = generate_face_from_sketch(img)

out_path = Path("processed/generated_face.jpg")
out_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_path), out)

print("✅ Generated face saved at:", out_path)
