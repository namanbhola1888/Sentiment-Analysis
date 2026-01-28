FROM python:3.11-slim AS build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install requirements in one go (better compatibility)
RUN pip install --default-timeout=300 --retries=10 --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

# Create directories
RUN mkdir -p uploads

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

CMD ["python", "app.py"]