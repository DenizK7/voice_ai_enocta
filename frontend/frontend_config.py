# frontend_config.py
import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "deniz-ai.up.railway.app/api/chatbot")
