import requests
from config import Config
ELEVENLABS_API_KEY = Config.ELEVENLABS_API_KEY  
ELEVENLABS_VOICE_ID = Config.ELEVENLABS_VOICE_ID  
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech"

def text_to_speech(text: str) -> bytes:
    
    headers = {
        "Accept": "audio/mpeg",
        "xi-api-key": Config.ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8
        }
    }

    response = requests.post(f"{ELEVENLABS_URL}/{ELEVENLABS_VOICE_ID}", json=payload, headers=headers)

    if response.status_code == 200:
        return response.content 
    else:
        raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")
