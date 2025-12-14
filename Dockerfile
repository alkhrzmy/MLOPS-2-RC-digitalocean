# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (ffmpeg for librosa/audioread)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching (CPU-only torch wheel)
COPY requirements.txt ./
# Ensure numpy present before heavy deps
RUN pip install --no-cache-dir numpy==1.26.4 \
 && PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    pip install --no-cache-dir -r requirements.txt

# Copy project and install package
COPY . .
RUN PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
