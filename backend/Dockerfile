FROM python:3.10.15-slim

WORKDIR /app

COPY . .

ARG GUARDRAILS_API_KEY

ENV GUARDRAILS_API_KEY=${GUARDRAILS_API_KEY}

ARG OPENAI_API_KEY

ENV OPENAI_API_KEY=${OPENAI_API_KEY}

ARG JINA_API_KEY

ENV JINA_API_KEY=${JINA_API_KEY}


# Install git and other required utilities (optional)
RUN apt-get update && apt-get install -y git

RUN apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn -y
# RUN apt-get tesseract-ocr-dev
        
RUN apt install libmagic1 libmagic-dev -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install libgl1 -y
RUN apt-get install poppler-utils -y

RUN pip install uv

RUN uv venv /opt/backend_env --python 3.10

ENV PATH="/opt/backend_env/bin:$PATH"
# Install dependencies using the virtual environment's pip
RUN uv pip sync requirements.txt
RUN uv pip install "pymilvus[model]"
RUN uv pip install "pathway[xpack-llm-docs]"
RUN uv pip install python-magic
RUN uv pip install Pillow
RUN uv pip install pi-heif
RUN uv pip install unstructured-inference
RUN uv pip install pdf2image
RUN uv pip install python-dateutil
RUN uv pip install unstructured.pytesseract
RUN uv pip install --upgrade nltk
RUN python setup.py

RUN apt-get update

RUN apt-get update && apt-get install -y \
    libleptonica-dev \
    tesseract-ocr \
    libtesseract-dev \
    python3-pil \
    tesseract-ocr-eng \
    tesseract-ocr-script-latn \
    && rm -rf /var/lib/apt/lists/*

RUN export TESSDATA_PREFIX=/usr/share/tesseract-orc

RUN guardrails configure --enable-metrics --enable-remote-inferencing --token $GUARDRAILS_API_KEY

RUN git clone https://github.com/guardrails-ai/toxic_language.git

RUN mkdir /opt/backend_env/lib/python3.10/site-packages/guardrails/hub/guardrails

RUN mv toxic_language/ /opt/backend_env/lib/python3.10/site-packages/guardrails/hub/guardrails/

RUN git clone https://github.com/guardrails-ai/regex_match.git

RUN mv regex_match/ /opt/backend_env/lib/python3.10/site-packages/guardrails/hub/guardrails/

EXPOSE 8001
EXPOSE 8765
EXPOSE 8766

CMD ["uvicorn", "fast_api_server:app", "--port", "8001", "--host", "0.0.0.0"]