\# HYBRID DEEP LEARNING MODEL FOR SKETCH TO FACE RECOGNITION



\## Objective

To design a hybrid deep learning system that matches a hand-drawn or digital facial sketch

to real face images using sketch-to-photo translation, deep embeddings, and explainable AI.



\## Datasets

\- CUFS (Sketch–Photo pairs)

\- CUFSF (Forensic sketch–photo pairs)

\- IIIT-D Sketch Dataset

\- CelebA (Face gallery and synthetic sketch generation)



\## Models Used

\- Face Detection: RetinaFace (ONNX, onnxruntime-gpu)

\- Face Embedding: ArcFace ResNet100 (ONNX)

\- Sketch-to-Photo Translation: Pix2Pix GAN (PyTorch)

\- Sketch Encoder: ResNet18 (Metric Learning)

\- Sketch Style Normalization: U-Net

\- Sketch Quality Estimation: ResNet18

\- Text Encoder (Optional): Sentence-BERT / CLIP



\## System Pipeline

Sketch Input →

Sketch Quality Estimation →

Style Normalization →

Face Detection \& Alignment →

Sketch Encoder OR Pix2Pix + ArcFace →

Embedding Generation →

Similarity Matching →

Top-K Results / No Match →

Grad-CAM Explainability



\## Output

\- Top-K matched face images with similarity scores

\- Confidence score for each match

\- Heatmap visualization for explainability

\- No-match decision when similarity is below threshold



