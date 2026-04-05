import cv2
import numpy as np

def is_forensic_style(sketch):
    gray = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)

    # Edge density check
    edges = cv2.Canny(gray, 80, 160)
    density = edges.mean()

    # Reject very noisy or very empty sketches
    if density < 8 or density > 65:
        return False

    # Contrast check
    contrast = gray.std()
    if contrast < 25:
        return False

    return True
