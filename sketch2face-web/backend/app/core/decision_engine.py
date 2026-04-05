from typing import Dict, List, Optional
from app.models.match_log import DecisionIntelligence
import sys
from pathlib import Path

# Import existing Decision Intelligence modules
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.reliability_score import compute_match_reliability
from src.decision.gallery_density import compute_gallery_density
from src.decision.cross_gallery_check import cross_gallery_consistency
import numpy as np

class DecisionEngine:
    """
    Decision Intelligence Engine
    
    Uses existing src.decision modules (MRS, GDA, CGCC)
    instead of duplicating logic.
    """
    
    def analyze(
        self,
        top_matches: List[Dict],
        gallery: str,
        query_embedding: Optional[np.ndarray] = None,
        gallery_embeddings: Optional[np.ndarray] = None,
        cross_gallery_data: Optional[Dict] = None
    ) -> DecisionIntelligence:
        """
        Run full decision intelligence analysis using existing modules.
        
        Args:
            top_matches: List of top K matches with scores
            gallery: Gallery used (cufs/celeba)
            query_embedding: Query embedding vector (for density check)
            gallery_embeddings: Gallery embeddings (for density check)
            cross_gallery_data: Dict with celeba_gallery and cufs_gallery embeddings
        """
        
        if not top_matches or len(top_matches) < 1:
            return DecisionIntelligence(
                reliability_score=0.0,
                density_risk="UNKNOWN",
                consistency_verdict="UNAVAILABLE",
                final_decision="NO_MATCHES"
            )
        
        # Extract scores
        top1_score = top_matches[0].get("similarity_score", 0.0)
        top2_score = top_matches[1].get("similarity_score", 0.0) if len(top_matches) > 1 else 0.0
        top_k_scores = [m.get("similarity_score", 0.0) for m in top_matches[:5]]
        
        # 1. Match Reliability Score (MRS) - use existing module
        reliability = compute_match_reliability(
            top1_score=top1_score,
            top2_score=top2_score,
            top_k_scores=top_k_scores,
            gallery_name=gallery
        )
        
        # 2. Gallery Density Awareness (GDA) - use existing module
        if query_embedding is not None and gallery_embeddings is not None:
            density = compute_gallery_density(
                query_emb=query_embedding,
                gallery_embs=gallery_embeddings,
                k=50
            )
            density_risk = density["risk_level"]
        else:
            # Fallback: simple density check from scores
            density_risk = self._simple_density_check(top_matches)
        
        # 3. Cross-Gallery Consistency Check (CGCC) - use existing module
        if cross_gallery_data:
            consistency = cross_gallery_consistency(
                query_emb=query_embedding if query_embedding is not None else np.array([]),
                celeba_gallery=cross_gallery_data.get("celeba_gallery"),
                cufs_gallery=cross_gallery_data.get("cufs_gallery")
            )
            consistency_verdict = consistency["verdict"]
        else:
            consistency_verdict = "UNAVAILABLE"
        
        # 4. Final Decision
        final_decision = self._make_final_decision(
            reliability["reliability_score"],
            density_risk,
            consistency_verdict
        )
        
        return DecisionIntelligence(
            reliability_score=reliability["reliability_score"],
            density_risk=density_risk,
            consistency_verdict=consistency_verdict,
            final_decision=final_decision
        )
    
    def _simple_density_check(self, top_matches: List[Dict]) -> str:
        """Fallback density check if embeddings not available"""
        if len(top_matches) < 5:
            return "LOW"
        
        top1_score = top_matches[0].get("similarity_score", 0.0)
        close_matches = sum(
            1 for m in top_matches[1:5]
            if abs(m.get("similarity_score", 0.0) - top1_score) < 0.05
        )
        
        if close_matches >= 3:
            return "HIGH"
        elif close_matches >= 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _make_final_decision(
        self,
        reliability_score: float,
        density_risk: str,
        consistency_verdict: str
    ) -> str:
        """Make final decision based on Decision Intelligence"""
        reliability_ok = reliability_score >= 70
        density_ok = density_risk in ["LOW", "MEDIUM"]
        consistency_ok = consistency_verdict != "INCONSISTENT"
        
        if reliability_ok and density_ok and consistency_ok:
            return "ACCEPTED"
        elif reliability_score >= 50 and density_ok:
            return "MEDIUM_CONFIDENCE"
        else:
            return "LOW_CONFIDENCE"

# Singleton instance
decision_engine = DecisionEngine()