# Use Python 3.11 full image (more libs pre-installed)
FROM --platform=linux/amd64 python:3.11

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first (Docker caching)
COPY requirements.txt .

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install heavy packages separately for caching
RUN pip --default-timeout=300 install --no-cache-dir \
    tensorflow==2.20.0 \
    torch==2.9.1 torchvision==0.24.1 \
    opencv-python==4.12.0.88 opencv-contrib-python==4.12.0.88

# Install remaining packages
RUN pip --default-timeout=300 install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p uploads

# Expose port
EXPOSE 5000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Optional safe health check for Render
HEALTHCHECK --interval=60s --timeout=3s --start-period=30s --retries=5 \
    CMD curl -fs http://localhost:5000/api/health || echo "Health check failed"


# Run the Flask app
CMD ["python", "app.py"]
