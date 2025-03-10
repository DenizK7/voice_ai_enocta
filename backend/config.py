# config.py
import os
from dotenv import load_dotenv

load_dotenv()
class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
    
    DEBUG = os.getenv("DEBUG", "False") == "True"
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    
    RAG_AGENT_API_KEY = os.getenv("RAG_AGENT_API_KEY")
    
   
    
    VIDEO_UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    TRANSCRIPT_UPLOAD_FOLDER= os.getenv("TRANSCRIPT_UPLOAD_FOLDER", "transcripts")
    FAISS_SAVE_FOLDER= os.getenv("FAISS_SAVE_FOLDER", "faiss")
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
    ELEVENLABS_VOICE_ID =os.getenv("ELEVENLABS_VOICE_ID")

    DENIZ_OPENAI_API_KEY = os.getenv("DENIZ_OPENAI_API_KEY", OPENAI_API_KEY)
    DENIZ_ELEVENLABS_API_KEY = os.getenv("DENIZ_ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)
    DENIZ_BRAVE_API_KEY = os.getenv("DENIZ_BRAVE_API_KEY", BRAVE_API_KEY)