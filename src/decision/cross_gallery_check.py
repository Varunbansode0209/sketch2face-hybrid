"""
Cross-Gallery Consistency Check (CGCC).

Ensures identity consistency across CUFS and CelebA galleries to detect
potential false positives or identity mismatches.
"""

import numpy as np
from typing import Tuple, Optional


def cross_gallery_consistency(
    query_emb: np.ndarray,
    celeba_gallery: Optional[np.ndarray],
    cufs_gallery: Optional[np.ndarray],
) -> dict:
    """
    Check consistency of query identity across both galleries.
    
    Args:
        query_emb: Query embedding vector (normalized)
        celeba_gallery: CelebA gallery embeddings (N_celeba, D), normalized, or None
        cufs_gallery: CUFS gallery embeddings (N_cufs, D), normalized, or None
        
    Returns:
        Dictionary with:
            - consistent: bool or None (None if unavailable)
            - score_gap: float or None
            - verdict: "CONSISTENT" | "INCONSISTENT" | "UNAVAILABLE"
            - celeba_top1: float or None
            - cufs_top1: float or None
    """
    # Check if both galleries are available
    if celeba_gallery is None or cufs_gallery is None:
        return {
            "consistent": None,
            "score_gap": None,
            "verdict": "UNAVAILABLE",
            "celeba_top1": None,
            "cufs_top1": None,
            "message": "Cross-gallery check unavailable (only one gallery loaded)",
        }
    
    # Ensure query is normalized
    query_emb = query_emb / np.linalg.norm(query_emb)
    
    # Ensure galleries are normalized
    celeba_norms = np.linalg.norm(celeba_gallery, axis=1, keepdims=True)
    celeba_gallery = celeba_gallery / (celeba_norms + 1e-8)
    
    cufs_norms = np.linalg.norm(cufs_gallery, axis=1, keepdims=True)
    cufs_gallery = cufs_gallery / (cufs_norms + 1e-8)
    
    # Compute top-1 similarities in both galleries
    celeba_similarities = celeba_gallery @ query_emb
    cufs_similarities = cufs_gallery @ query_emb
    
    celeba_top1 = float(np.max(celeba_similarities))
    cufs_top1 = float(np.max(cufs_similarities))
    
    # Compute gap
    score_gap = abs(celeba_top1 - cufs_top1)
    
    # Threshold for consistency (allow some variance due to different distributions)
    consistency_threshold = 0.15
    
    if score_gap <= consistency_threshold:
        consistent = True
        verdict = "CONSISTENT"
        message = (
            f"Identity consistent across galleries. "
            f"CUFS: {cufs_top1:.3f} | CelebA: {celeba_top1:.3f} | Gap: {score_gap:.3f}"
        )
    else:
        consistent = False
        verdict = "INCONSISTENT"
        message = (
            f"Identity inconsistent across galleries. "
            f"CUFS: {cufs_top1:.3f} | CelebA: {celeba_top1:.3f} | Gap: {score_gap:.3f} > {consistency_threshold:.2f}"
        )
    
    return {
        "consistent": consistent,
        "score_gap": float(score_gap),
        "verdict": verdict,
        "celeba_top1": celeba_top1,
        "cufs_top1": cufs_top1,
        "message": message,
    }
