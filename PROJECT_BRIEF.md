# Sketch-to-Face Recognition Project - Brief Overview

## Project Purpose
A **hybrid deep learning system** that matches hand-drawn or digital facial sketches to real face images in a gallery. The system uses a two-stage approach: first converting sketches to photos using a GAN, then performing face recognition using deep embeddings.

## Core Architecture

### Main Pipeline (as seen in `match_query.py`)
1. **Sketch Input** → Load query sketch image
2. **Pix2Pix Generation** → Convert sketch to photo using GAN
3. **Face Detection** → Extract face region from generated photo (RetinaFace)
4. **ArcFace Embedding** → Generate 512-dimensional face embedding
5. **Similarity Matching** → Compare against pre-computed gallery embeddings
6. **Top-K Results** → Return best matches with similarity scores and visualizations

## Key Components

### Models Used
- **RetinaFace** (ONNX) - Face detection and alignment
- **ArcFace ResNet100** (ONNX) - Face embedding generation
- **Pix2Pix GAN** (PyTorch) - Sketch-to-photo translation
- Additional models (mentioned in design doc but may not be fully implemented):
  - Sketch Encoder (ResNet18)
  - Style Normalization (U-Net)
  - Quality Estimator (ResNet18)

### Source Code Structure
```
src/
├── api/              - FastAPI endpoint (basic setup)
├── dataset/          - Dataset indexing and splitting utilities
├── embedding/        - ArcFace inference and gallery building
├── generation/       - Pix2Pix sketch-to-photo generation
├── inference/        - End-to-end pipeline (placeholder)
├── matching/         - Query matching and visualization
├── preprocess/       - Face detection, alignment, gallery preprocessing
├── training/         - Training scripts for various models
└── utils/            - Configuration and helper functions
```

## Datasets
- **CUFS** - Sketch-photo pairs
- **CUFSF** - Forensic sketch-photo pairs  
- **FS2K** - Face sketch dataset (appears to be primary dataset)
- **CelebA** - Large-scale face dataset for gallery

## Current State
- ✅ Core matching pipeline implemented (`match_query.py`)
- ✅ Pix2Pix inference working
- ✅ ArcFace embedding extraction working
- ✅ Gallery embedding pre-computation
- ✅ Visualization of top-K matches
- ⚠️ API endpoint is basic (needs implementation)
- ⚠️ Full inference pipeline is placeholder
- ⚠️ Training scripts present but status unknown

## Workflow Example
1. User provides a sketch image (`data/query/f1-010-01-sz1.jpg`)
2. System generates a photo from the sketch using Pix2Pix
3. Face is detected and extracted from generated photo
4. ArcFace generates embedding vector
5. Cosine similarity computed against gallery embeddings
6. Top-K matches (default: 5) returned with scores above threshold (0.25)
7. Results visualized and saved to `processed/results/topk/`

## Key Files
- `src/matching/match_query.py` - Main matching script
- `src/generation/pix2pix_infer.py` - Sketch-to-photo generation
- `src/embedding/arcface_infer.py` - Face embedding extraction
- `src/preprocess/detect_face.py` - Face detection
- `embeddings/gallery/fs2k_gallery.npy` - Pre-computed gallery embeddings
- `embeddings/index.json` - Mapping of indices to image paths

## Technology Stack
- **PyTorch** - Deep learning framework (Pix2Pix)
- **ONNX Runtime** - Model inference (ArcFace, RetinaFace)
- **OpenCV** - Image processing
- **FastAPI** - Web API (basic setup)
- **NumPy** - Numerical computations
