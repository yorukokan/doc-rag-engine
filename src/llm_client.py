import os
from dotenv import load_dotenv
from openai import OpenAI

# .env dosyasındaki LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME bilgilerini yükle
load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)
MODEL_NAME = os.getenv("LLM_MODEL_NAME")
