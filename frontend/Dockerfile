FROM python:3.10.15-slim
RUN apt-get update
RUN apt-get -f install
RUN pip install --upgrade pip
RUN pip install gradio loguru openai exa_py python-dotenv
RUN pip install --upgrade --force-reinstall gradio
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV EXA_API_KEY=${EXA_API_KEY}
ENV JINA_API_KEY=${JINA_API_KEY}

WORKDIR /app
COPY . .
RUN mkdir -p /app/shared_logs/logs && touch /app/shared_logs/logs/log_text.md
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "ui.py", "--share"]