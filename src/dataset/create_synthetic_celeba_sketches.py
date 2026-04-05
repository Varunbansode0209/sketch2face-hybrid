"""
Create synthetic sketch–photo pairs from CelebA photos for Pix2Pix fine-tuning.

Why this exists
---------------
- The current Pix2Pix generator was trained only on CUFS forensic sketches.
- CelebA / internet-style sketches have a different appearance (domain gap),
  so the CUFS-only generator produces blurry / off-identity faces.
- To keep the CUFS pipeline untouched and still improve CelebA performance,
  we fine-tune a *copy* of the CUFS Pix2Pix generator on synthetic data.

What this script does
---------------------
- Takes cropped CelebA face photos and converts them to sketch-like images
  using Canny edges + light smoothing (CPU-only, OpenCV).
- Creates paired images in the same `A|B` format used for CUFS Pix2Pix
  training: left half = synthetic sketch, right half = original photo.
- Writes pairs into a new dataset directory inside the pix2pix repo:

    pytorch-CycleGAN-and-pix2pix/datasets/celeba_synth/train
    pytorch-CycleGAN-and-pix2pix/datasets/celeba_synth/test

- Optionally oversamples existing CUFS Pix2Pix pairs into this dataset so
  that the fine-tuning data is dominated by CelebA but still includes a
  non-trivial fraction of real CUFS pairs (~10% by count).

This synthetic dataset is then used to fine-tune a separate generator
checkpoint for CelebA while the original CUFS-trained checkpoint remains
unchanged. This preserves the existing CUFS pipeline exactly as-is, while
improving identity preservation for CelebA and internet sketches.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default input: cropped CelebA photos (already aligned to faces)
DEFAULT_CELEBA_PHOTO_DIR = PROJECT_ROOT / "data" / "raw" / "celeba" / "photos"

# Output: Pix2Pix-ready paired dataset inside the original pix2pix repo
PIX2PIX_DATASETS_ROOT = PROJECT_ROOT / "pytorch-CycleGAN-and-pix2pix" / "datasets"
DEFAULT_OUT_DATASET = PIX2PIX_DATASETS_ROOT / "celeba_synth"

# Original CUFS Pix2Pix dataset (already aligned pairs).
CUFS_PIX2PIX_ROOT = PIX2PIX_DATASETS_ROOT / "cufs_pix2pix"


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in root.glob("*") if p.suffix.lower() in exts])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def photo_to_synthetic_sketch(
    img_bgr: np.ndarray,
    img_size: int = 256,
    canny_low: int = 50,
    canny_high: int = 150,
) -> np.ndarray:
    """
    Convert a BGR face photo to a synthetic sketch.

    The goal is not photorealistic pencil art, but a stable, high-contrast
    sketch that roughly matches the CUFS-like sketches the generator expects.
    """
    # Resize and convert to grayscale
    img_resized = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Normalize contrast for more consistent edges
    gray = cv2.equalizeHist(gray)

    # Canny edges
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Slightly thicken and smooth edges
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)

    # Invert so that lines are dark on light background, similar to CUFS
    sketch = 255 - edges

    # Convert to 3-channel for consistency with CUFS pix2pix input
    sketch_3c = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    return sketch_3c


def build_celeba_pairs(
    photo_dir: Path,
    out_root: Path,
    img_size: int = 256,
    val_ratio: float = 0.05,
    canny_low: int = 50,
    canny_high: int = 150,
    max_images: int | None = None,
) -> None:
    """
    Create Pix2Pix paired images for CelebA:
    left = synthetic sketch, right = original photo (both resized to img_size).
    """
    photos = list_images(photo_dir)
    if max_images is not None:
        photos = photos[:max_images]

    if not photos:
        raise RuntimeError(f"No input images found in {photo_dir}")

    n_total = len(photos)
    n_val = max(1, int(n_total * val_ratio))

    train_paths = photos[n_val:]
    val_paths = photos[:n_val]

    train_dir = out_root / "train"
    test_dir = out_root / "test"
    ensure_dir(train_dir)
    ensure_dir(test_dir)

    def _process_split(
        paths: Iterable[Path],
        out_dir: Path,
        split_name: str,
    ) -> int:
        count = 0
        for i, photo_path in enumerate(paths, start=1):
            img = cv2.imread(str(photo_path))
            if img is None:
                continue

            sketch = photo_to_synthetic_sketch(
                img, img_size=img_size, canny_low=canny_low, canny_high=canny_high
            )
            photo_resized = cv2.resize(
                img, (img_size, img_size), interpolation=cv2.INTER_AREA
            )

            pair = cv2.hconcat([sketch, photo_resized])
            out_path = out_dir / photo_path.name
            cv2.imwrite(str(out_path), pair)

            count += 1
            if i % 500 == 0:
                print(f"[{split_name}] Processed {i}/{len(list(paths))} images...")
        return count

    print(f"➡ Creating CelebA synthetic pairs in: {out_root}")
    n_train = _process_split(train_paths, train_dir, "train")
    n_test = _process_split(val_paths, test_dir, "test")

    print(f"✅ CelebA synthetic train pairs: {n_train}")
    print(f"✅ CelebA synthetic test  pairs: {n_test}")


def oversample_cufs_into_celeba(out_root: Path, target_cufs_ratio: float = 0.10) -> None:
    """
    Oversample CUFS Pix2Pix pairs into the CelebA synthetic dataset.

    This is a purely offline data operation: we do NOT modify the CUFS
    training code or the existing CUFS generator. Instead, we copy CUFS
    paired images into the CelebA dataset (with suffixed filenames) so that
    the fine-tuning data remains mostly CelebA (~90%) but still includes
    a healthy amount of real CUFS sketches as regularization.
    """
    train_dir = out_root / "train"
    test_dir = out_root / "test"
    ensure_dir(train_dir)
    ensure_dir(test_dir)

    celeba_train = list_images(train_dir)
    celeba_test = list_images(test_dir)
    n_celeba_train = len(celeba_train)
    n_celeba_test = len(celeba_test)

    if n_celeba_train == 0:
        print("⚠️ No CelebA train pairs found; skipping CUFS oversampling.")
        return

    cufs_train_dir = CUFS_PIX2PIX_ROOT / "train"
    cufs_test_dir = CUFS_PIX2PIX_ROOT / "test"

    cufs_train = list_images(cufs_train_dir)
    cufs_test = list_images(cufs_test_dir)

    if not cufs_train:
        print("⚠️ No CUFS Pix2Pix pairs found; skipping CUFS oversampling.")
        return

    # Compute how many CUFS copies we need so that:
    #   effective_cufs / (celeba + effective_cufs) ~= target_cufs_ratio
    # Solve for effective_cufs:
    #   effective_cufs = target * (celeba + effective_cufs)
    #   effective_cufs * (1 - target) = target * celeba
    #   effective_cufs = target * celeba / (1 - target)
    desired_cufs_train = int(
        math.ceil(target_cufs_ratio * n_celeba_train / (1.0 - target_cufs_ratio))
    )
    copies_per_img = max(1, math.ceil(desired_cufs_train / max(1, len(cufs_train))))

    print(
        f"➡ Oversampling CUFS into CelebA train: "
        f"{len(cufs_train)} base images × {copies_per_img} copies each "
        f"→ target ≈ {target_cufs_ratio*100:.1f}% CUFS"
    )

    def _copy_with_suffix(
        src_list: List[Path],
        dst_dir: Path,
        split_name: str,
        copies: int,
    ) -> int:
        copied = 0
        for src in src_list:
            for k in range(copies):
                dst_name = f"{src.stem}_cufs_{k:02d}{src.suffix}"
                dst_path = dst_dir / dst_name
                img = cv2.imread(str(src))
                if img is None:
                    continue
                cv2.imwrite(str(dst_path), img)
                copied += 1
        print(f"✅ Copied {copied} CUFS pairs into {split_name}")
        return copied

    _copy_with_suffix(cufs_train, train_dir, "train", copies_per_img)

    if cufs_test and n_celeba_test > 0:
        # For test we do a single copy per image; exact ratio is less critical.
        _copy_with_suffix(cufs_test, test_dir, "test", copies=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create synthetic CelebA sketch–photo pairs for Pix2Pix fine-tuning "
            "and optionally oversample CUFS pairs into the same dataset."
        )
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=DEFAULT_CELEBA_PHOTO_DIR,
        help="Directory with cropped CelebA face photos.",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=DEFAULT_OUT_DATASET,
        help="Output Pix2Pix dataset root inside the pix2pix repo.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Target image size (both sketch and photo are resized to this).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Fraction of images reserved for test split.",
    )
    parser.add_argument(
        "--canny_low",
        type=int,
        default=50,
        help="Lower Canny edge threshold.",
    )
    parser.add_argument(
        "--canny_high",
        type=int,
        default=150,
        help="Upper Canny edge threshold.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional cap on number of CelebA photos to process (for quick tests).",
    )
    parser.add_argument(
        "--no_cufs_oversample",
        action="store_true",
        help="If set, do NOT copy CUFS Pix2Pix pairs into the CelebA dataset.",
    )
    parser.add_argument(
        "--cufs_ratio",
        type=float,
        default=0.10,
        help="Approximate target CUFS fraction in the mixed training set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_celeba_pairs(
        photo_dir=args.input_root,
        out_root=args.out_root,
        img_size=args.img_size,
        val_ratio=args.val_ratio,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        max_images=args.max_images,
    )

    if not args.no_cufs_oversample:
        oversample_cufs_into_celeba(out_root=args.out_root, target_cufs_ratio=args.cufs_ratio)
    else:
        print("ℹ️ Skipping CUFS oversampling (no_cufs_oversample flag set).")


if __name__ == "__main__":
    main()

