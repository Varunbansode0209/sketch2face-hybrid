import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("processed/evaluation/confidence_margins.npy")
OUT_PATH  = Path("processed/evaluation/confidence_margin_distribution.png")

CONF_MARGIN = 0.05

print("▶ Loading confidence margins...")
margins = np.load(DATA_PATH)

plt.figure(figsize=(8, 5))
plt.hist(margins, bins=50, color="blue", alpha=0.75)

plt.axvline(
    CONF_MARGIN,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Confidence Margin = {CONF_MARGIN}"
)

plt.xlabel("Confidence Margin (Top1 − Top2)")
plt.ylabel("Frequency")
plt.title("Confidence Margin Distribution")
plt.legend()
plt.grid(True)

plt.savefig(OUT_PATH, dpi=200)
plt.close()

print(f"✅ Saved confidence margin plot: {OUT_PATH}")
