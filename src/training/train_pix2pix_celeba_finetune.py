"""
Fine-tune Pix2Pix generator for CelebA sketches.

This script:
1. Initializes a new checkpoint directory with the CUFS-trained generator weights
2. Runs fine-tuning on the mixed CelebA+CUFS dataset
3. Keeps the original CUFS checkpoint untouched

Why this approach:
- CUFS pipeline remains unchanged (original checkpoint never modified)
- Fine-tuning adapts the generator to CelebA-style sketches
- Mixed dataset (90% CelebA synthetic, 10% CUFS) prevents catastrophic forgetting
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIX2PIX_ROOT = PROJECT_ROOT / "pytorch-CycleGAN-and-pix2pix"

# Paths
CUFS_CHECKPOINT_DIR = PIX2PIX_ROOT / "checkpoints" / "cufs_pix2pix"
CELEBA_CHECKPOINT_DIR = PIX2PIX_ROOT / "checkpoints" / "cufs_celeba_finetune"
CELEBA_DATASET = PIX2PIX_ROOT / "datasets" / "celeba_synth"


def initialize_checkpoint_from_cufs():
    """
    Copy CUFS generator weights to the new CelebA checkpoint directory.
    
    This initializes the fine-tuning from the CUFS-trained generator,
    ensuring we start with a model that already understands sketch-to-photo
    translation, just adapted to CelebA-style sketches.
    """
    if not CUFS_CHECKPOINT_DIR.exists():
        raise RuntimeError(
            f"CUFS checkpoint not found at {CUFS_CHECKPOINT_DIR}. "
            "Please ensure CUFS Pix2Pix training is complete."
        )

    cufs_gen_path = CUFS_CHECKPOINT_DIR / "latest_net_G.pth"
    cufs_disc_path = CUFS_CHECKPOINT_DIR / "latest_net_D.pth"

    if not cufs_gen_path.exists():
        raise RuntimeError(f"CUFS generator checkpoint not found: {cufs_gen_path}")

    CELEBA_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy generator weights (required for fine-tuning)
    celeba_gen_path = CELEBA_CHECKPOINT_DIR / "latest_net_G.pth"
    shutil.copy2(cufs_gen_path, celeba_gen_path)
    print(f"✅ Copied CUFS generator: {cufs_gen_path} → {celeba_gen_path}")

    # Copy discriminator weights (optional but helps convergence)
    if cufs_disc_path.exists():
        celeba_disc_path = CELEBA_CHECKPOINT_DIR / "latest_net_D.pth"
        shutil.copy2(cufs_disc_path, celeba_disc_path)
        print(f"✅ Copied CUFS discriminator: {cufs_disc_path} → {celeba_disc_path}")

    # Copy training options for reference
    for opt_file in ["train_opt.txt", "test_opt.txt"]:
        src = CUFS_CHECKPOINT_DIR / opt_file
        if src.exists():
            dst = CELEBA_CHECKPOINT_DIR / opt_file
            shutil.copy2(src, dst)


def run_finetuning(
    n_epochs: int = 10,
    n_epochs_decay: int = 0,
    batch_size: int = 4,
    lr: float = 1e-4,
    load_size: int = 286,
    crop_size: int = 256,
    save_epoch_freq: int = 2,
):
    """
    Run Pix2Pix fine-tuning using the standard training script.
    
    Parameters match the CUFS training config but with:
    - Lower learning rate (1e-4 vs 2e-4) for fine-tuning
    - Fewer epochs (10-20) to avoid overfitting
    - Mixed CelebA+CUFS dataset (already prepared by create_synthetic_celeba_sketches.py)
    """
    if not CELEBA_DATASET.exists():
        raise RuntimeError(
            f"CelebA synthetic dataset not found at {CELEBA_DATASET}. "
            "Please run: python -m src.dataset.create_synthetic_celeba_sketches"
        )

    train_script = PIX2PIX_ROOT / "train.py"

    cmd = [
        sys.executable,
        str(train_script),
        "--model", "pix2pix",
        "--name", "cufs_celeba_finetune",
        "--dataroot", str(CELEBA_DATASET),
        "--dataset_mode", "aligned",
        "--direction", "AtoB",
        "--continue_train",  # Load from the checkpoint we just initialized
        "--epoch_count", "1",
        "--n_epochs", str(n_epochs),
        "--n_epochs_decay", str(n_epochs_decay),
        "--batch_size", str(batch_size),
        "--load_size", str(load_size),
        "--crop_size", str(crop_size),
        "--lr", str(lr),
        "--lambda_L1", "100",  # Match CUFS training
        "--netG", "unet_256",  # Match CUFS architecture
        "--norm", "batch",  # Match CUFS normalization
        "--no_dropout",  # Match CUFS config
        "--save_epoch_freq", str(save_epoch_freq),
        "--print_freq", "50",
        "--display_freq", "400",
    ]

    print("\n" + "=" * 60)
    print("STARTING PIX2PIX FINE-TUNING")
    print("=" * 60)
    print(f"Dataset: {CELEBA_DATASET}")
    print(f"Checkpoint: {CELEBA_CHECKPOINT_DIR}")
    print(f"Epochs: {n_epochs} (+ {n_epochs_decay} decay)")
    print(f"Learning rate: {lr}")
    print("=" * 60 + "\n")

    result = subprocess.run(cmd, cwd=str(PIX2PIX_ROOT))
    return result.returncode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Pix2Pix generator for CelebA sketches from CUFS checkpoint."
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--n_epochs_decay",
        type=int,
        default=0,
        help="Number of epochs for learning rate decay (default: 0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate (default: 1e-4, lower than CUFS training)",
    )
    parser.add_argument(
        "--load_size",
        type=int,
        default=286,
        help="Load size before cropping (default: 286)",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=256,
        help="Crop size (default: 256)",
    )
    parser.add_argument(
        "--skip_init",
        action="store_true",
        help="Skip checkpoint initialization (use if already initialized)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_init:
        print("▶ Initializing checkpoint from CUFS generator...")
        initialize_checkpoint_from_cufs()
        print("✅ Checkpoint initialization complete\n")
    else:
        print("ℹ️ Skipping checkpoint initialization (--skip_init flag set)\n")

    exit_code = run_finetuning(
        n_epochs=args.n_epochs,
        n_epochs_decay=args.n_epochs_decay,
        batch_size=args.batch_size,
        lr=args.lr,
        load_size=args.load_size,
        crop_size=args.crop_size,
    )

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ FINE-TUNING COMPLETE")
        print("=" * 60)
        print(f"Fine-tuned generator saved at:")
        print(f"  {CELEBA_CHECKPOINT_DIR / 'latest_net_G.pth'}")
        print("\nNext steps:")
        print("  1. Test the fine-tuned generator on CelebA sketches")
        print("  2. Update pix2pix_infer.py to use this checkpoint for CelebA")
        print("=" * 60)
    else:
        print("\n❌ Fine-tuning failed. Check logs above for errors.")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
