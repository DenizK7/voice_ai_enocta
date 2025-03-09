# frontend_config.py
import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000/api/chatbot")
