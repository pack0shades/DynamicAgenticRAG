FROM python:3.10.15-slim

WORKDIR /app

COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libleptonica-dev \
    tesseract-ocr \
    libtesseract-dev \
    python3-pil \
    tesseract-ocr-eng \
    tesseract-ocr-script-latn \
    libmagic1 \
    libmagic-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN pip install uv
RUN uv venv /opt/backend_env --python 3.10
ENV PATH="/opt/backend_env/bin:$PATH"

# Install Python dependencies
RUN uv pip sync requirements.txt
RUN uv pip install "pymilvus[model]" "pathway[xpack-llm-docs]" python-magic Pillow pi-heif \
    unstructured-inference pdf2image python-dateutil unstructured.pytesseract \
    && python -m nltk.downloader popular

RUN export TESSDATA_PREFIX=/usr/share/tesseract-orc

# Setup guardrails (moved to entrypoint)
RUN git clone https://github.com/guardrails-ai/toxic_language.git \
    && git clone https://github.com/guardrails-ai/regex_match.git \
    && mkdir -p /opt/backend_env/lib/python3.10/site-packages/guardrails/hub/guardrails \
    && mv toxic_language/ regex_match/ /opt/backend_env/lib/python3.10/site-packages/guardrails/hub/guardrails/

# make a log directory and create a log file
RUN mkdir -p /app/shared_logs/logs && touch /app/shared_logs/logs/log_text.txt

# Runtime configuration
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8001 8765 8766

ENTRYPOINT ["/app/entrypoint.sh"]