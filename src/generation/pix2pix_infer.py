"""
Pix2Pix inference module with dual generator support.

This module supports two generators:
1. CUFS generator: Trained on CUFS forensic sketches (original, unchanged)
2. CelebA generator: Fine-tuned on CelebA synthetic sketches (new)

The appropriate generator is selected based on the gallery type to ensure
optimal identity preservation for each domain.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
from typing import Literal

# Add pix2pix repo to path (robust to current working directory)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIX2PIX_ROOT = PROJECT_ROOT / "pytorch-CycleGAN-and-pix2pix"
sys.path.append(str(PIX2PIX_ROOT))

from models.networks import define_G

DEVICE = torch.device("cpu") # Updated to torch.device for better compatibility

# Paths to trained generators (relative to project root)
CUFS_MODEL_PATH = PIX2PIX_ROOT / "checkpoints" / "cufs_pix2pix" / "latest_net_G.pth"
CELEBA_MODEL_PATH = PIX2PIX_ROOT / "checkpoints" / "cufs_celeba_finetune" / "latest_net_G.pth"

# Generator instances (lazy-loaded)
_cufs_generator: torch.nn.Module | None = None
_celeba_generator: torch.nn.Module | None = None


def _load_generator(checkpoint_path: Path) -> torch.nn.Module:
    """Load a Pix2Pix generator from checkpoint."""
    netG = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="unet_256",
        norm="batch",
        use_dropout=False,
        init_type="normal",
        init_gain=0.02
    )
    netG.to(DEVICE)
    
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    
    # Fix for potential __patch_instance_norm issues in some pix2pix versions
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    
    netG.load_state_dict(state_dict)
    netG.eval()
    return netG


def _get_generator(gallery_type: Literal["cufs", "celeba"]) -> torch.nn.Module:
    """
    Get the appropriate generator for the given gallery type.
    
    Args:
        gallery_type: "cufs" for CUFS gallery, "celeba" for CelebA gallery
        
    Returns:
        Loaded and initialized generator model
    """
    global _cufs_generator, _celeba_generator
    
    if gallery_type == "cufs":
        if _cufs_generator is None:
            if not CUFS_MODEL_PATH.exists():
                raise RuntimeError(
                    f"CUFS generator checkpoint not found: {CUFS_MODEL_PATH}\n"
                    "Please ensure CUFS Pix2Pix training is complete."
                )
            _cufs_generator = _load_generator(CUFS_MODEL_PATH)
        return _cufs_generator
    
    elif gallery_type == "celeba":
        if _celeba_generator is None:
            # Fallback to CUFS generator if CelebA checkpoint doesn't exist
            if not CELEBA_MODEL_PATH.exists():
                print(
                    f"⚠️ CelebA generator checkpoint not found: {CELEBA_MODEL_PATH}\n"
                    "Falling back to CUFS generator. For better CelebA results, "
                    "please run fine-tuning:\n"
                    "  python -m src.training.train_pix2pix_celeba_finetune"
                )
                if _cufs_generator is None:
                    _cufs_generator = _load_generator(CUFS_MODEL_PATH)
                return _cufs_generator
            _celeba_generator = _load_generator(CELEBA_MODEL_PATH)
        return _celeba_generator
    
    else:
        raise ValueError(f"Invalid gallery_type: {gallery_type}. Must be 'cufs' or 'celeba'")


# ---------------- Pre/Post ----------------
def preprocess(img):
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(DEVICE) # Ensure input is on the same device as model


def postprocess(tensor):
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# ---------------- API ----------------
def generate_face_from_sketch(
    sketch_img: np.ndarray,
    gallery_type: Literal["cufs", "celeba"] = "cufs"
) -> np.ndarray:
    """
    Generate a face photo from a sketch using the appropriate generator.
    
    Args:
        sketch_img: Input sketch image (BGR format, OpenCV)
        gallery_type: "cufs" for CUFS gallery, "celeba" for CelebA gallery
        
    Returns:
        Generated face photo (BGR format, OpenCV)
    """
    netG = _get_generator(gallery_type)
    with torch.no_grad():
        inp = preprocess(sketch_img)
        out = netG(inp)
    return postprocess(out)