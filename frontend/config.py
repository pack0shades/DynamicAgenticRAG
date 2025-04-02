# LOG_FILE = "/app/shared_logs/logs/log_text.md"
LOG_FILE = "logs/log_text.txt"
HOST_NAME = "0.0.0.0"
GRADIO_PORT = 7860
FAST_API_PORT = 8001
MODEL = "gpt-4o-mini"

WEB_SEARCH_PROMPT = [
                    {"role": "system", "content": "You are a helpful AI assistant. Summarize the given search results about user_message"},
                    {"role": "user", "content": """Search results: {result}\n"""}]

JINA_URL = "https://s.jina.ai/"