"""
Decision Intelligence Layer for Sketch2Face-Hybrid.

This module provides post-matching decision features that improve
trustworthiness and explainability without modifying core matching logic.
"""

from src.decision.reliability_score import compute_match_reliability
from src.decision.gallery_density import compute_gallery_density
from src.decision.cross_gallery_check import cross_gallery_consistency

__all__ = [
    "compute_match_reliability",
    "compute_gallery_density",
    "cross_gallery_consistency",
]
