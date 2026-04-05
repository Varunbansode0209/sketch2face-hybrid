import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("processed/evaluation/impostor_scores.npy")
OUT_DIR = Path("processed/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

impostor_scores = np.load(DATA_PATH)

thresholds = np.linspace(0.2, 0.6, 50)
far_rates = []

for t in thresholds:
    far = np.mean(impostor_scores >= t)
    far_rates.append(far)

plt.figure(figsize=(8,5))
plt.plot(thresholds, far_rates, marker="o")
plt.xlabel("Cosine Similarity Threshold")
plt.ylabel("False Accept Rate (FAR)")
plt.title("FAR vs Threshold Curve")
plt.grid(True)

plt.savefig(OUT_DIR / "far_vs_threshold.png", dpi=200)
plt.close()

print("✅ FAR vs Threshold curve saved")
