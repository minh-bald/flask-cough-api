# Use Python 3.10 instead of 3.9 (better compatibility for TFLite)
FROM python:3.10-slim

# Set working directory
WORKDIR /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt || true

# Manually install TFLite runtime (since pip can’t find it)
RUN pip install --no-cache-dir \
    https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.14.0-cp310-cp310-manylinux_2_17_x86_64.whl

# Expose Flask port
EXPOSE 8080

# Environment vars for Flask
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Run Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]

