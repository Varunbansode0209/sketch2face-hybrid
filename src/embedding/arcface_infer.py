import numpy as np
import onnxruntime as ort
import cv2

MODEL_PATH = "models/onnx/arcface_r50.onnx"

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "onnx" / "arcface_r50.onnx"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"ArcFace model not found at: {MODEL_PATH}")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)


input_name = session.get_inputs()[0].name


def preprocess_face(face_img):
    """
    Model expects NHWC: (1, 112, 112, 3)
    """
    img = cv2.resize(face_img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0  # ArcFace normalization

    img = np.expand_dims(img, axis=0)  # (1, 112, 112, 3)
    return img


def get_embedding(face_img):
    emb = session.run(
        None,
        {input_name: preprocess_face(face_img)}
    )[0]

    emb = emb / np.linalg.norm(emb)
    return emb.squeeze()
