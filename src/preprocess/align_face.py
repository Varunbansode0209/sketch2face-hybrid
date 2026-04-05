import cv2
import numpy as np
from pathlib import Path
from detect_face import detect_faces

SRC_DIR = Path("data/test_images")
OUT_DIR = Path("data/aligned_faces")
OUT_DIR.mkdir(parents=True,exist_ok=True)

ARC_SIZE = (112,112)

def clamp_box(x1,y1,x2,y2,w,h):
    x1 = max(0, x1); y1 = max(0,y1)
    x2= min(w-1,x2); y2 = min(h-1,y2)
    return x1,y1,x2,y2


def crop_and_align(img, box):
    h, w = img.shape[:2]
    x1,y1,x2,y2 = clamp_box(*box,w, h)

    face = img[y1:y2, x1:x2]
    if face.size ==0:
        return None
    
    face = cv2.resize(face, ARC_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    return face

def main():
    img_paths = list(SRC_DIR.glob("*"))
    if not img_paths:
        print("No images found")
        return

    saved = 0
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        boxes = detect_faces(img)
        if not boxes:
            print(f"No face in {img_path.name}")
            continue

        # Save all detected faces (index them)
        for i, box in enumerate(boxes):
            face = crop_and_align(img, box)
            if face is None:
                continue

            out_name = f"{img_path.stem}_face{i}.png"
            out_path = OUT_DIR / out_name

            # save back as uint8 for inspection
            vis = ((face * 128.0) + 127.5).clip(0,255).astype(np.uint8)
            cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            saved += 1

    print(f"Saved {saved} aligned faces to {OUT_DIR}")

if __name__ == "__main__":
    main()