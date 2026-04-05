FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new user to avoid permissions issues with HuggingFace
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the rest of the backend files inside the container
COPY --chown=user . $HOME/app

# Run FastAPI specifically on port 7860 as mandated by Hugging Face
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
