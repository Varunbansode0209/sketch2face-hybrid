FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY sketch2face-web/backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Run as non-root user matching HuggingFace OpenRAIL constraints
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Change context so FastAPI resolves local database routes appropriately
WORKDIR $HOME/app/sketch2face-web/backend

# Hugging face requires port 7860 exposed
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
