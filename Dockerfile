FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for audio processing (libsndfile is required by librosa)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for model artifacts if it doesn't exist
RUN mkdir -p app/ml/artifacts

# Expose port
EXPOSE 8000

# Command to run the application (using shell form to allow environment variable expansion)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
