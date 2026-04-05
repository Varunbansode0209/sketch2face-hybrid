"""
Test script to verify model imports work correctly
Run this from the backend directory to check if models can be imported
"""

import sys
from pathlib import Path

# Calculate project root (same as ai_pipeline_wrapper.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # backend/ -> sketch2face-web/ -> project_root
sys.path.insert(0, str(PROJECT_ROOT))

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"src exists: {(PROJECT_ROOT / 'src').exists()}")
print()

# Test imports
print("Testing model imports...")
print("-" * 50)

try:
    from src.generation.pix2pix_infer import generate_face_from_sketch
    print("✅ Pix2Pix import: SUCCESS")
except ImportError as e:
    print(f"❌ Pix2Pix import: FAILED - {e}")

try:
    from src.embedding.arcface_infer import get_embedding
    print("✅ ArcFace import: SUCCESS")
except ImportError as e:
    print(f"❌ ArcFace import: FAILED - {e}")

try:
    from src.preprocess.detect_face import detect_faces
    print("✅ Face Detection import: SUCCESS")
except ImportError as e:
    print(f"❌ Face Detection import: FAILED - {e}")

try:
    from src.preprocess.sketch_validator import is_forensic_style
    print("✅ Sketch Validator import: SUCCESS")
except ImportError as e:
    print(f"❌ Sketch Validator import: FAILED - {e}")

try:
    from src.decision.reliability_score import compute_match_reliability
    print("✅ Decision Intelligence import: SUCCESS")
except ImportError as e:
    print(f"❌ Decision Intelligence import: FAILED - {e}")

print("-" * 50)
print("\nIf all imports succeeded, models are properly integrated! ✅")
