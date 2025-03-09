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

def speech_to_text(mp3_file: str) -> str:
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
    time.sleep(1)
    with open(transcript_path, "w", encoding="utf-8") as json_file:
        json.dump(transcription.__dict__, json_file, indent=4, ensure_ascii=False, default=lambda o: str(o))
    try:
        os.remove(mp3_file)
        print(f"✅ {mp3_file} deleted, it no longer takes up space!")
    except Exception as e:
        print(f"Could not delete MP3 file: {e}")
    return transcription.__dict__

def prepare_ai_transcript(transcript_json: dict, video_title: str, video_url: str) -> dict:
    """
    Processes the transcript data and saves it as JSON.
    - Uses a safe file name for saving.
    
    Args:
        transcript_json (dict): The transcription data in JSON format.
        video_title (str): The title of the video.
        video_url (str): The URL of the video.
    
    Returns:
        dict: The processed transcript data.
    """
    time.sleep(1)
    words = transcript_json.get("words", [])
    segments = []
    current_segment = None

    for entry in words:
        data = parse_word_entry(entry)
        if not data["text"]:
            data["text"] = " "
        cleaned_text = remove_fillers(data["text"])
        data["text"] = cleaned_text

        if current_segment is None or data["speaker"] != current_segment["speaker"]:
            if current_segment:
                current_segment["end"] = data["start"]
                current_segment["text"] = re.sub(r'\s+', ' ', str(current_segment["text"])).strip()
                segments.append(current_segment)
            current_segment = {
                "speaker": data["speaker"],
                "start": data["start"],
                "end": data["end"],
                "text": data["text"]
            }
        else:
            current_segment["text"] += " " + data["text"]
            current_segment["end"] = data["end"]

    if current_segment:
        current_segment["text"] = re.sub(r'\s+', ' ', str(current_segment["text"])).strip()
        segments.append(current_segment)

    revised = {
        "video_metadata": {
            "title": video_title,
            "url": video_url
        },
        "segments": segments
    }

    safe_filename = normalize_filename(video_title)
    transcript_filename = f"{safe_filename}.json"
    transcript_path = os.path.join(Config.TRANSCRIPT_UPLOAD_FOLDER, transcript_filename)

    os.makedirs(Config.TRANSCRIPT_UPLOAD_FOLDER, exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as json_file:
        json.dump(revised, json_file, indent=4, ensure_ascii=False)

    return revised

def remove_fillers(text) -> str:
    """
    Removes unnecessary filler words from the text.
    
    Args:
        text (str): The text to be cleaned.
    
    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""
    fillers = {"eee", "hımm", "uh", "umm", "mmm", "ııı"}
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in fillers]
    return " ".join(filtered_words)

def parse_word_entry(entry) -> dict:
    """
    Extracts text, start, end, and speaker information from each entry in the 'words' list.
    
    Args:
        entry (str): The entry string to be parsed.
    
    Returns:
        dict: A dictionary containing 'text', 'start', 'end', and 'speaker'.
    """
    pattern_text = r"text='([^']*)'"
    pattern_start = r"start=([\d\.]+)"
    pattern_end = r"end=([\d\.]+)"
    pattern_speaker = r"speaker_id='([^']*)'"

    text_match = re.search(pattern_text, str(entry))
    start_match = re.search(pattern_start, str(entry))
    end_match = re.search(pattern_end, str(entry))
    speaker_match = re.search(pattern_speaker,str(entry))

    return {
        "text": text_match.group(1) if text_match else "",
        "start": float(start_match.group(1)) if start_match else 0.0,
        "end": float(end_match.group(1)) if end_match else 0.0,
        "speaker": speaker_match.group(1) if speaker_match else "unknown"
    }
