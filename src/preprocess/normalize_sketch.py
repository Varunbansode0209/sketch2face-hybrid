import cv2
import numpy as np

def normalize_to_cufs(sketch):
    # Convert to grayscale
    gray = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
    
    # Gently stretch contrast instead of harsh edge detection
    # CLAHE is usually better for sketches than global equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    
    # Convert back to BGR so Pix2Pix receives 3 channels
    return cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
