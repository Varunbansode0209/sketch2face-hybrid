"""
Gallery Density Awareness (GDA).

Detects if a query lies in a dense identity cluster, which increases
the risk of false positive matches in large galleries like CelebA.
"""

import numpy as np
from typing import Tuple


def compute_gallery_density(
    query_emb: np.ndarray,
    gallery_embs: np.ndarray,
    k: int = 50,
) -> dict:
    """
    Compute density score for query embedding in gallery.
    
    Args:
        query_emb: Query embedding vector (normalized)
        gallery_embs: Gallery embeddings array (N, D), already normalized
        k: Number of nearest neighbors to consider
        
    Returns:
        Dictionary with:
            - density_score: float (0-1), where 1 = very dense
            - risk_level: "LOW" | "MEDIUM" | "HIGH"
            - message: Human-readable explanation
    """
    if gallery_embs.shape[0] < k:
        k = gallery_embs.shape[0]
    
    # Ensure query is normalized
    query_emb = query_emb / np.linalg.norm(query_emb)
    
    # Ensure gallery embeddings are normalized
    gallery_norms = np.linalg.norm(gallery_embs, axis=1, keepdims=True)
    gallery_embs = gallery_embs / (gallery_norms + 1e-8)
    
    # Compute cosine similarities to all gallery embeddings
    similarities = gallery_embs @ query_emb
    
    # Get top-k similarities
    top_k_indices = np.argsort(-similarities)[:k]
    top_k_similarities = similarities[top_k_indices]
    
    # Density metrics
    mean_similarity = np.mean(top_k_similarities)
    std_similarity = np.std(top_k_similarities)
    
    # Dense cluster = high mean + low variance (many faces are similar)
    density_score = mean_similarity
    
    # Risk assessment
    if density_score < 0.6 or std_similarity > 0.1:
        risk_level = "LOW"
        message = (
            f"Query lies in sparse region (mean similarity to {k} neighbors: {mean_similarity:.3f}). "
            "Low identity confusion risk."
        )
    elif density_score < 0.75 and std_similarity <= 0.1:
        risk_level = "MEDIUM"
        message = (
            f"Query lies in moderately dense cluster (mean similarity: {mean_similarity:.3f}). "
            "Similar faces exist in gallery."
        )
    else:  # density_score >= 0.75 and std <= 0.1
        risk_level = "HIGH"
        message = (
            f"Query lies in very dense identity cluster (mean similarity: {mean_similarity:.3f}). "
            "High risk of confusion with similar identities."
        )
    
    return {
        "density_score": float(density_score),
        "risk_level": risk_level,
        "message": message,
        "mean_similarity": float(mean_similarity),
        "std_similarity": float(std_similarity),
        "k": k,
    }
