import cv2
import numpy as np

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def unsharp_mask(img, amount=1.2):
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=1.0)
    sharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def enhance_generated_face(img):
    img = apply_clahe(img)
    img = unsharp_mask(img)
    return img
