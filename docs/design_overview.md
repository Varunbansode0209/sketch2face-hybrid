\# HYBRID DEEP LEARNING MODEL FOR SKETCH TO FACE RECOGNITION



\## Objective

The objective of this project is to design a hybrid deep learning system that can take a hand-drawn

or digital facial sketch as input, convert it into a meaningful representation, and retrieve the

most similar real face images from a database. The system also provides explainability using heatmaps

to justify why a match was selected.



\## Datasets

\- CUFS (Sketch–Photo Pairs)

\- CUFSF (Forensic Sketch–Photo Pairs)

\- IIIT-D Sketch Dataset

\- CelebA (Face Gallery and Synthetic Sketch Generation)



\## Models Used

\- Face Detection: RetinaFace (ONNX, onnxruntime-gpu)

\- Face Embedding: ArcFace ResNet100 (ONNX)

\- Sketch-to-Photo Translation: Pix2Pix GAN (PyTorch)

\- Sketch Encoder: ResNet18 with Metric Learning

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

\- Heatmap visualization showing important facial regions

\- No-match response if similarity is below threshold



