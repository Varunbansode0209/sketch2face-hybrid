"""
Match Reliability Scoring (MRS).

Converts raw similarity scores into human-interpretable reliability scores (0-100)
with clear explanations for decision-making.
"""

import numpy as np
from typing import List


def compute_match_reliability(
    top1_score: float,
    top2_score: float,
    top_k_scores: List[float],
    gallery_name: str,
) -> dict:
    """
    Compute reliability score (0-100) from similarity scores.
    
    Args:
        top1_score: Top-1 similarity score
        top2_score: Top-2 similarity score
        top_k_scores: List of top-K similarity scores (at least top-2, ideally top-5)
        gallery_name: "cufs" or "celeba"
        
    Returns:
        Dictionary with:
            - reliability_score: int (0-100)
            - level: "LOW" | "MEDIUM" | "HIGH"
            - explanation: list of explanation strings
    """
    explanations = []
    
    # Base score from top1_score (0.0-1.0 -> 0-100)
    base_score = int(top1_score * 100)
    explanations.append(f"Base score from top-1 similarity ({top1_score:.3f})")
    
    # Confidence margin bonus
    margin = top1_score - top2_score
    margin_threshold = 0.05 if gallery_name == "cufs" else 0.02
    
    margin_bonus = 0
    if margin >= margin_threshold:
        margin_bonus = min(15, int((margin / margin_threshold) * 10))
        explanations.append(
            f"Strong margin bonus (+{margin_bonus}): {margin:.3f} > {margin_threshold:.2f}"
        )
    else:
        explanations.append(
            f"Weak margin penalty: {margin:.3f} < {margin_threshold:.2f}"
        )
        margin_bonus = -5  # Small penalty
    
    # Top-K consistency bonus
    if len(top_k_scores) >= 3:
        top_k_std = np.std(top_k_scores[:5]) if len(top_k_scores) >= 5 else np.std(top_k_scores)
        consistency_bonus = 0
        
        if top_k_std < 0.01:  # Very tight cluster
            consistency_bonus = 10
            explanations.append(f"Excellent top-K consistency (std: {top_k_std:.4f})")
        elif top_k_std < 0.03:  # Good consistency
            consistency_bonus = 5
            explanations.append(f"Good top-K consistency (std: {top_k_std:.4f})")
        else:
            explanations.append(f"Lower top-K consistency (std: {top_k_std:.4f})")
    else:
        consistency_bonus = 0
        explanations.append("Insufficient top-K scores for consistency check")
    
    # Gallery-specific penalty (CelebA has larger population = more confusion risk)
    gallery_penalty = 0
    if gallery_name == "celeba":
        gallery_penalty = -5
        explanations.append("CelebA gallery penalty (large population)")
    else:
        explanations.append("CUFS gallery (small population, lower risk)")
    
    # Compute final score
    reliability_score = base_score + margin_bonus + consistency_bonus + gallery_penalty
    
    # Clip to valid range
    reliability_score = max(0, min(100, reliability_score))
    
    # Determine level
    if reliability_score >= 80:
        level = "HIGH"
    elif reliability_score >= 50:
        level = "MEDIUM"
    else:
        level = "LOW"
    
    return {
        "reliability_score": reliability_score,
        "level": level,
        "explanation": explanations,
        "margin": margin,
        "margin_threshold": margin_threshold,
    }
