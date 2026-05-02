FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for OpenCV and git
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

COPY sketch2face-web/backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Run as non-root user matching HuggingFace OpenRAIL constraints
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Explicitly clone the required submodules because Hugging Face does not init them automatically
# Remove the empty directory if it exists and clone
RUN rm -rf $HOME/app/pytorch-CycleGAN-and-pix2pix && \
    git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git $HOME/app/pytorch-CycleGAN-and-pix2pix

# Change context so FastAPI resolves local database routes appropriately
WORKDIR $HOME/app/sketch2face-web/backend

# Hugging face requires port 7860 exposed
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
