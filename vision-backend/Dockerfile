# Vision Backend - production Docker image
# Build: docker build -t vision-backend:latest .
# Run:   docker run -p 8000:8000 -e MONGO_URI=mongodb://host:27017 vision-backend:latest

FROM python:3.11-slim

# System deps for OpenCV, FFmpeg, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (PyTorch CPU from official index)
COPY requirements.txt constraints.cpu.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt \
       -c constraints.cpu.txt

# Application code (do not copy .env — config via env at runtime)
COPY app ./app

# Writable dirs used by config (event_videos, uploads, gallery, chroma, models)
RUN mkdir -p /app/event_videos /app/static_video_uploads /app/Gallery \
             /app/static_video_chroma_db /app/models

# Run as non-root user
RUN useradd -m -u 1000 -s /bin/sh appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# All config (MONGO_URI, KAFKA, etc.) must be passed at run time via -e or --env-file
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
