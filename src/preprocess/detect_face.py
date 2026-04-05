import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL_PATH = "models/onnx/yolov8n_face.onnx"
INPUT_SIZE = 640

from pathlib import Path
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "onnx" / "yolov8n_face.onnx"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLOv8 face model not found at: {MODEL_PATH}")

sess = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

input_name = sess.get_inputs()[0].name

def detect_faces(img, conf_thres=0.4, iou_thres=0.5):
    h0, w0 = img.shape[:2]

    # Preprocessing
    img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = np.transpose(img_norm, (2, 0, 1))[None]

    # Inference
    preds = sess.run(None, {input_name: img_norm})[0]
    
    # YOLOv8 output shape is usually (1, 84, 8400) or similar
    # We need to transpose it to (8400, 84) to iterate over detections
    preds = np.squeeze(preds).T 

    boxes = []
    scores = []

    for p in preds:
        # p[0:4] are cx, cy, w, h
        # p[4] is the score for 'face'
        score = p[4] 
        if score < conf_thres:
            continue

        cx, cy, w, h = p[:4]

        # Convert center-xywh to x1, y1, x2, y2
        # And scale to original image dimensions
        x1 = int((cx - w/2) * w0 / INPUT_SIZE)
        y1 = int((cy - h/2) * h0 / INPUT_SIZE)
        width = int(w * w0 / INPUT_SIZE)
        height = int(h * h0 / INPUT_SIZE)

        boxes.append([x1, y1, width, height])
        scores.append(float(score))

    if not boxes:
        return []

    # Non-Maximum Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            final_boxes.append((x, y, x + w, y + h))

    return final_boxes


def detect_single_face(img):
    boxes = detect_faces(img)

    if not boxes:
        return None

    # Take the largest face (best for gallery)
    x1, y1, x2, y2 = max(
        boxes,
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
    )

    # Clamp coordinates
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face = img[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return face


def main():
    img_paths = list(Path("data/test_images").glob("*"))

    if not img_paths:
        print("No images found")
        return

    for img_path in img_paths:
        img = cv2.imread(str(img_path))

        boxes = detect_faces(img)

        if not boxes:
            print(f"No face detected in {img_path.name}")
            continue

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

        out = Path("processed/detection_test") / img_path.name
        cv2.imwrite(str(out), img)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
