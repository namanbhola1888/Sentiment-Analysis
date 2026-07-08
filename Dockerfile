# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — install all Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.11-slim AS builder

# System build deps (only what pip needs to compile wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Upgrade pip + wheel tooling
RUN pip install --upgrade pip setuptools wheel --no-cache-dir

# ── Layer A: CPU-only PyTorch (saves ~2 GB vs CUDA build) ─────────────────
# Render has no GPU — the full CUDA wheel is pure waste.
RUN pip install --no-cache-dir \
    "torch==2.9.1+cpu" \
    "torchvision==0.24.1+cpu" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── Layer B: TensorFlow ────────────────────────────────────────────────────
RUN pip install --no-cache-dir "tensorflow==2.20.0"

# ── Layer C: Headless OpenCV (no GUI libs needed in a container) ───────────
RUN pip install --no-cache-dir "opencv-python-headless==4.12.0.88"

# ── Layer D: Remaining app requirements ───────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK vader_lexicon at build time (avoids runtime download)
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime — slim final image with only what's needed
# ─────────────────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.11-slim

# Runtime system deps only (no build tools in the final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire installed Python site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy NLTK data downloaded during build
COPY --from=builder /root/nltk_data /root/nltk_data

# Copy application source (done last — most frequently changing layer)
COPY . .

RUN mkdir -p uploads

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV MPLBACKEND=Agg

# ── Force CPU-only TensorFlow / PyTorch ─────────────────────────────────────
# Render free tier has no GPU. Without these, TF probes for CUDA at startup,
# finds partial CUDA runtime libs on the host, and crashes with error 303.
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# ─────────────────────────────────────────────────────────────────────────────

HEALTHCHECK --interval=60s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fs http://localhost:5000/api/health || exit 1

CMD ["python", "app.py"]

