import os
import io
import json
import logging
from flask import Blueprint, Response, request, jsonify, send_file
from app.services import llm, transcription, tts
from config import Config
from app.services.prompt_manager import PromptManager
from app.services.vector_db import add_transcript_to_db, delete_by_video_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

prompt_manager = PromptManager()
chatbot_bp = Blueprint('chatbot', __name__)
TRANSCRIPT_FOLDER = Config.TRANSCRIPT_UPLOAD_FOLDER

@chatbot_bp.route('/list-transcripts', methods=['GET'])
def list_transcripts():
    """
    Endpoint to list uploaded transcripts.
    Lists .json files in the transcripts folder.
    """
    logging.info("Listing transcripts from folder: %s", TRANSCRIPT_FOLDER)
    try:
        if not os.path.exists(TRANSCRIPT_FOLDER):
            logging.info("Transcript folder does not exist.")
            return jsonify({"transcripts": []})
        files = [f for f in os.listdir(TRANSCRIPT_FOLDER) if f.endswith(".json")]
        logging.info("Found %d transcript files.", len(files))
        return jsonify({"transcripts": files})
    except Exception as e:
        logging.error("Error listing transcripts: %s", str(e))
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route('/delete-transcript', methods=['POST'])
def delete_transcript():
    """
    Delete transcript and related FAISS records based on video URL.
    Reads the JSON file specified by "filename", extracts the video URL,
    removes all embeddings related to this video URL from FAISS, and deletes the JSON file.
    """
    logging.info("Delete transcript endpoint called.")
    data = request.get_json()
    if not data or "filename" not in data:
        logging.error("Filename missing in request.")
        return jsonify({"error": "Missing filename in request"}), 400

    filename = data["filename"]
    file_path = os.path.join(TRANSCRIPT_FOLDER, filename)

    try:
        if not os.path.exists(file_path):
            logging.error("File not found: %s", file_path)
            return jsonify({"error": "File not found"}), 404

        with open(file_path, "r", encoding="utf-8") as json_file:
            transcript_data = json.load(json_file)

        video_url = transcript_data.get("video_metadata", {}).get("url", None)
        if not video_url:
            logging.error("Video URL not found in transcript.")
            return jsonify({"error": "Video URL not found in transcript"}), 400

        deleted = delete_by_video_url(video_url)
        if not deleted:
            logging.error("Failed to delete from vector DB for video URL: %s", video_url)
            return jsonify({"error": "Could not remove from vector DB"}), 200

        os.remove(file_path)
        logging.info("Deleted transcript and vector DB records for video URL: %s", video_url)
        return jsonify({"success": True, "message": f"Video ({video_url}) and related embeddings deleted successfully"})
    except Exception as e:
        logging.error("Error deleting transcript: %s", str(e))
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route('/upload-youtube', methods=['POST'])
def upload_youtube_video():
    """
    Endpoint for uploading video from a YouTube URL.
    Expects "video_url" in the JSON payload. Downloads the video,
    converts it to mp3, generates a transcript, and adds it to the vector database.
    """
    logging.info("Upload YouTube video endpoint called.")
    data = request.get_json()
    if not data or "video_url" not in data:
        logging.error("Missing video_url in request.")
        return jsonify({"error": "Missing video_url in request"}), 400

    video_url = data["video_url"]
    try:
        mp3_file, title = transcription.upload_mp3(video_url)
        transcript_json = transcription.speech_to_text(mp3_file, video_url=video_url, title=title)
        vector_add = add_transcript_to_db(transcript_json)
        logging.info("Uploaded and processed video from URL: %s", video_url)
        return f"{vector_add}"
    except Exception as e:
        logging.error("Error uploading YouTube video: %s", str(e))
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint.
    Expects "question", and optionally "top_k" and "temperature" in the JSON payload.
    Generates an answer and references using the llm.generate_answer function.
    """
    logging.info("Chat endpoint called.")
    data = request.get_json()
    if not data or "question" not in data:
        logging.error("Missing question in request.")
        return jsonify({"error": "Missing question in request"}), 400

    question = data["question"]
    top_k = data.get("top_k", 5)
    temperature = data.get("temperature", 0.7)
    try:
        result = llm.generate_answer(question, top_k=top_k, temperature=temperature)
        logging.info("Chat generated an answer for question: %s", question)
        return jsonify(result)
    except Exception as e:
        logging.error("Error in chat endpoint: %s", str(e))
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route('/tts-chat', methods=['POST'])
def tts_chat():
    """
    Chat + TTS endpoint.
    Generates an answer using the LLM model, converts the answer text to speech using TTS,
    and returns MP3 audio data along with JSON metadata in the response header.
    """
    logging.info("TTS Chat endpoint called.")
    data = request.get_json()
    if not data or "question" not in data:
        logging.error("Missing question in TTS request.")
        return jsonify({"error": "Missing question in request"}), 400

    question = data["question"]
    top_k = data.get("top_k", 5)
    temperature = data.get("temperature", 0.7)
    similarity_threshold = data.get("similarity_threshold", 0.77)

    try:
        result = llm.generate_answer(question, top_k=top_k, temperature=temperature, similarity_threshold=similarity_threshold)
        answer_text = result["answer"]
        references = result.get("references", [])
        audio_bytes = tts.text_to_speech(answer_text)
        audio_stream = io.BytesIO(audio_bytes)
        audio_stream.seek(0)
        metadata = json.dumps({
            "answer": answer_text,
            "references": references
        })
        response = Response(audio_stream, mimetype="audio/mpeg")
        response.headers["Metadata"] = metadata
        response.headers["Content-Disposition"] = "attachment; filename=response.mp3"
        logging.info("TTS Chat processed successfully.")
        return response
    except Exception as e:
        logging.error("Error in TTS Chat endpoint: %s", str(e))
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route('/get-prompts', methods=['GET'])
def get_prompts():
    """
    Returns the current prompt settings.
    """
    logging.info("Getting prompt settings.")
    return jsonify(prompt_manager.load_prompts())

@chatbot_bp.route('/update-prompts', methods=['POST'])
def update_prompts():
    """
    Updates the prompt settings.
    """
    logging.info("Updating prompt settings.")
    data = request.get_json()
    if "adjustment_instructions" not in data:
        logging.error("Missing adjustment_instructions in prompt update.")
        return jsonify({"error": "Missing adjustment_instructions"}), 400
    prompt_manager.save_prompts(data)
    logging.info("Prompt settings updated successfully.")
    return jsonify({"success": True, "message": "Prompt updated successfully!"})

@chatbot_bp.route('/upload-json', methods=['POST'])
def upload_json():
    """
    Upload JSON endpoint.
    Expects a JSON file as a form-data parameter named "file".
    Loads the JSON data and adds it to the vector database.
    """
    logging.info("Upload JSON endpoint called.")
    if 'file' not in request.files:
        logging.error("No file found in request.")
        return jsonify({"error": "Missing file in request"}), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file.")
        return jsonify({"error": "No selected file"}), 400

    try:
        json_data = json.load(file)
        vector_add = add_transcript_to_db(json_data)
        logging.info("JSON file uploaded and added to vector database successfully.")
        return jsonify({"success": True, "message": "JSON uploaded and added to vector database successfully", "result": vector_add})
    except Exception as e:
        logging.error("Error uploading JSON: %s", str(e))
        return jsonify({"error": str(e)}), 500
