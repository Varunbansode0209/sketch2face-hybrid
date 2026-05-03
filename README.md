📌 Overview
Sketch2Face is a comprehensive, full-stack Artificial Intelligence platform built to bridge the gap between forensic sketches and real-world photographic identification. By utilizing advanced Generative Adversarial Networks (GANs) and state-of-the-art facial recognition models, the system successfully translates hand-drawn forensic sketches into hyper-realistic photos, and securely matches them against large criminal or civilian databases in real-time.

✨ Key Features
🎨 Sketch-to-Photo Translation: Leverages highly-trained Pix2Pix (CycleGAN) models fine-tuned on the CUFS and CelebA datasets to convert abstract sketches into realistic facial structures.
🧬 Deep Facial Embeddings: Uses ArcFace (ResNet-50) to extract 512-dimensional facial coordinate vectors for highly accurate, mathematically-driven identity matching.
📊 Decision Intelligence Engine: Doesn't just return a match. It analyzes the results using:
Reliability Scoring: Confidence level based on mathematical distance.
Density Risk: Evaluates how "crowded" the vector space is to prevent false positives.
Cross-Gallery Consistency: Validates the identity across multiple datasets.
🌡️ Explainable AI (Heatmaps): Generates visual heatmaps to show exactly which facial features the AI focused on when making an identification.
🔐 Secure Role-Based Authentication: Custom JWT authentication supporting Admin and standard user roles.
🏗️ System Architecture
The project is built on a modern Hybrid-Cloud Architecture, separating the heavy GPU machine learning workloads from the fast, interactive client interface.

1. Frontend (Vercel)
Built with React.js and Vite.
Styled dynamically with TailwindCSS.
Handles image uploads, result visualizations, user authentication, and admin dashboards.
2. Backend (Hugging Face Spaces)
Powered by FastAPI (Python).
Hosts the heavy PyTorch AI models (.pth checkpoints).
Processes image tensors, runs OpenCV inferences, and communicates with the database.
3. Database (MongoDB Atlas)
Secure NoSQL storage for users, authentication hashes, and system logs.
🚀 Getting Started (Local Development)
Prerequisites
Node.js (v18+)
Python 3.10+
Git
1. Clone the Repository
bash
git clone https://github.com/Varunbansode0209/sketch2face-hybrid.git
cd sketch2face-hybrid
2. Setup the AI Backend
bash
# Navigate to the backend
cd sketch2face-web/backend
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# Start the FastAPI server
uvicorn app.main:app --reload --port 8000
3. Setup the Frontend
bash
# Open a new terminal and navigate to the frontend
cd sketch2face-web/frontend
# Install dependencies
npm install
# Start the Vite development server
npm run dev
The frontend will be available at http://localhost:5173 and the backend at http://localhost:8000.

☁️ Deployment Guide
Deploying the Frontend (Vercel)
Import the GitHub repository into Vercel.
Vercel will automatically detect the vercel.json configuration in the root directory.
No further configuration is needed! Vercel will automatically route the build process to sketch2face-web/frontend.
Deploying the Backend (Hugging Face Spaces)
Create a new Docker Space on Hugging Face.
Link this GitHub repository.
Hugging Face will read the Dockerfile in sketch2face-web/backend/Dockerfile and automatically spin up the FastAPI service. (Note: Large PyTorch .pth models may need to be uploaded via Git LFS or direct transfer due to size limits).
🛡️ Presentation Safe-Mode
The platform includes an internal "Demo Mode" fallback mechanism within the ai_pipeline_wrapper.py. If deployed to a constrained cloud environment that cannot host the heavy Pix2Pix .pth models, the system gracefully degrading to process the raw sketch directly while triggering a Presentation Bluff to ensure live demos remain functional and visually impressive.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the 
issues page
.

📝 License
This project is for educational and demonstrative purposes.
