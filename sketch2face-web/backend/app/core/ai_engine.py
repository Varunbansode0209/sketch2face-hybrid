from typing import Dict, List
from pathlib import Path
import sys
import numpy as np

# Adjust sys path so we can import src reliably
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# IMPORTANT: torch MUST be imported BEFORE onnxruntime on Windows to avoid CUDA WinError 127
import torch 
from app.core.ai_pipeline_wrapper import AIPipelineWrapper
from app.core.config import settings

import cv2
from src.embedding.arcface_infer import get_embedding
from src.preprocess.detect_face import detect_faces

class AIEngine:
    """Interface to the existing AI/ML pipeline"""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.results_dir = Path(settings.RESULTS_DIR)
        
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline wrapper
        self.pipeline = AIPipelineWrapper(
            upload_dir=self.upload_dir,
            results_dir=self.results_dir
        )
    
    async def run_matching(self, input_image_path: str, gallery: str) -> Dict:
        """
        Run the AI matching pipeline using wrapper function.
        
        This calls your existing src.run_full_system functions directly
        without CLI interaction.
        """
        try:
            # Use wrapper to call pipeline programmatically
            results = self.pipeline.run_matching_pipeline(
                input_image_path=input_image_path,
                gallery_name=gallery
            )
            
            return results
            
        except Exception as e:
            raise Exception(f"AI Engine error: {str(e)}")
    
    async def extract_features(self, image_path: str) -> List[float]:
        """Extract ArcFace embeddings (coordinates) for a single image"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise Exception("Failed to read image for coordinate extraction")
            
        boxes = detect_faces(img)
        if boxes:
            x1, y1, x2, y2 = boxes[0]
            face = img[y1:y2, x1:x2]
        else:
            # Fallback center crop if detector fails
            h, w, _ = img.shape
            min_dim = min(h, w)
            cx, cy = w // 2, h // 2
            half = min_dim // 2
            face = img[cy - half : cy + half, cx - half : cx + half]
            
        # 112x112 is required for ArcFace R50
        face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)
        
        # Extracted vector coordinates!
        emb = get_embedding(face)
        return emb.tolist()
    
    async def rebuild_gallery(self, gallery: str) -> Dict:
        """Rebuild embeddings for a gallery"""
        cmd = [
            "python", "-m", "src.rebuild_gallery",
            "--gallery", gallery
        ]
        
        result = subprocess.run(
            cmd,
            cwd=self.ai_core_path,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for rebuild
        )
        
        if result.returncode != 0:
            raise Exception(f"Gallery rebuild failed: {result.stderr}")
        
        return {
            "status": "success",
            "gallery": gallery,
            "message": "Gallery embeddings rebuilt successfully"
        }

# Singleton instance
ai_engine = AIEngine()