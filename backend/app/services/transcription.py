import os
import re
import time
import unicodedata
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from io import BytesIO
from flask import json
import requests
from elevenlabs.client import ElevenLabs
from pytubefix import YouTube
from config import Config

config = Config()

def normalize_filename(filename: str) -> str:
    """
    Converts the file name to a safe format.
    - Replaces Turkish characters with their English equivalents.
    - Replaces spaces with underscores.
    - Removes invalid characters.
    
    Args:
        filename (str): The original file name.
    
    Returns:
        str: The normalized file name.
    """
    nfkd_form = unicodedata.normalize('NFKD', filename)
    only_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    safe_name = secure_filename(only_ascii)  
    safe_name = re.sub(r'[\/:*?"<>|]', '', str(safe_name))  # Remove forbidden characters for Windows
    return safe_name

def upload_mp3(video_url: str) -> str:
    """
    Downloads the audio from YouTube and converts it to MP3 format.
    
    Args:
        video_url (str): The URL of the YouTube video.
    
    Returns:
        str: The path to the downloaded MP3 file and its title.
    """
    yt = YouTube(video_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    if not audio_stream:
        raise Exception("No audio stream available from the provided URL.")

    destination = Config.VIDEO_UPLOAD_FOLDER
    os.makedirs(destination, exist_ok=True)

    # Create a safe file name
    title = normalize_filename(yt.title)
    new_mp3_file = os.path.join(destination, f"{title}.mp3")

    if os.path.exists(new_mp3_file):
        print(f"{yt.title} is already downloaded as MP3.")
        return new_mp3_file, yt.title

    out_file = audio_stream.download(output_path=destination)
    os.replace(out_file, new_mp3_file)

    print(f"{yt.title} has been successfully downloaded as MP3.")
    return new_mp3_file, yt.title

def speech_to_text(mp3_file: str,video_url, title) -> str:
    """
    Converts the MP3 file to text and saves it as a JSON file.
    
    Args:
        mp3_file (str): The path to the MP3 file.
    
    Returns:
        dict: The transcription in JSON format.
    """
    base, _ = os.path.splitext(os.path.basename(mp3_file))
    safe_filename = normalize_filename(base)
    transcript_filename = f"{safe_filename}.json"
    transcript_path = os.path.join(Config.TRANSCRIPT_UPLOAD_FOLDER, transcript_filename)

    os.makedirs(Config.TRANSCRIPT_UPLOAD_FOLDER, exist_ok=True)

    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)

    client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)

    with open(mp3_file, "rb") as f:
        audio_data = BytesIO(f.read())

    transcription = client.speech_to_text.convert(
        file=audio_data,
        model_id="scribe_v1",
        tag_audio_events=True,
        language_code="tur",
        diarize=True,
    )
    data = transcription.__dict__
    data["metada"] = {"video_url": video_url, "title": title}
    with open(transcript_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False, default=lambda o: str(o))
    try:
        os.remove(mp3_file)
        print(f"✅ {mp3_file} deleted, it no longer takes up space!")
    except Exception as e:
        print(f"Could not delete MP3 file: {e}")
        
    time.sleep(1)
    
    return data


