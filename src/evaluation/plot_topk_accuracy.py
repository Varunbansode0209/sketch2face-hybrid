import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path("processed/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Example values (replace with actual if you test)
k = [1, 3, 5]
accuracy = [0.82, 0.91, 0.96]

plt.figure(figsize=(6,4))
plt.plot(k, accuracy, marker="o")
plt.xlabel("Top-K")
plt.ylabel("Identification Accuracy")
plt.title("Top-K Identification Accuracy")
plt.grid(True)

plt.savefig(OUT_DIR / "topk_accuracy.png", dpi=200)
plt.close()

print("✅ Top-K accuracy curve saved")
