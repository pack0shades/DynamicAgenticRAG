FROM python:3.10.15-slim

# Install system dependencies and clean up in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies with cache optimization
# COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    gradio \
    loguru \
    openai \
    exa_py \
    python-dotenv

# Copy application code
COPY . .

# Create log directory (will be overlain by Docker volume)
RUN mkdir -p /app/shared_logs/logs && touch /app/shared_logs/logs/log_text.txt

# Runtime configuration
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# API keys will be injected via environment variables
CMD ["python", "ui.py", "--share"]