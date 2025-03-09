import os
from flask import Blueprint, Response, json, request, jsonify, send_file
from app.services import llm, transcription, tts
from config import Config
import io
from app.services.prompt_manager import PromptManager
from app.services.vector_db import add_transcript_to_db, delete_by_video_url

prompt_manager = PromptManager()
EMBEDDING_DIM = 1536
chatbot_bp = Blueprint('chatbot', __name__)
TRANSCRIPT_FOLDER = Config.TRANSCRIPT_UPLOAD_FOLDER  # Folder where transcripts are stored

@chatbot_bp.route('/list-transcripts', methods=['GET'])
def list_transcripts():
    """
     **Endpoint to list uploaded transcripts**
    - Called with a `GET` request.
    - Lists `.json` files in the folder.
    """
    try:
        if not os.path.exists(TRANSCRIPT_FOLDER):
            return jsonify({"transcripts": []})  # Return empty list if folder doesn't exist

        files = [f for f in os.listdir(TRANSCRIPT_FOLDER) if f.endswith(".json")]
        return jsonify({"transcripts": files})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chatbot_bp.route('/delete-transcript', methods=['POST'])
def delete_transcript():
    """
    🗑 **Delete transcript and related FAISS records based on video URL**
    - Finds the file using `"filename"` in the JSON request.
    - Extracts `"video_url"` from the JSON.
    - Removes all embeddings related to this `video_url` from FAISS.
    """
    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "Missing filename in request"}), 400

    filename = data["filename"]
    file_path = os.path.join(TRANSCRIPT_FOLDER, filename)

    try:
        # **1️⃣ Read JSON file and get video URL**
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        with open(file_path, "r", encoding="utf-8") as json_file:
            transcript_data = json.load(json_file)
        
        video_url = transcript_data.get("video_metadata", {}).get("url", None)
        if not video_url:
            return jsonify({"error": "Video URL not found in transcript"}), 400

        # **2️⃣ Delete all embeddings from FAISS for this video URL**
        deleted = delete_by_video_url(video_url)
        if not deleted:
            return jsonify({"error": "Could not remove from vector DB"}), 200

        # **3️⃣ Delete JSON file**
        os.remove(file_path)

        return jsonify({"success": True, "message": f"Video ({video_url}) and related embeddings deleted successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route('/upload-youtube', methods=['POST'])
def upload_youtube_video():
    """
    Endpoint for uploading video from YouTube URL:
    - Expects "video_url" in the JSON payload.
    - Downloads video from the URL, converts to mp3, and generates a transcript.
    """
    data = request.get_json()
    if not data or "video_url" not in data:
        return jsonify({"error": "Missing video_url in request"}), 400

    video_url = data["video_url"]
    try:
        mp3_file, title = transcription.upload_mp3(video_url)   
        clean_json = transcription.prepare_ai_transcript(transcription.speech_to_text(mp3_file), video_title=title, video_url=video_url)
        vector_add = add_transcript_to_db(clean_json)
        return f"{vector_add}"
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint:
    - Expects "question", optionally "top_k" and "temperature" in the JSON payload.
    - Generates answer and references using the llm.generate_answer function.
    """
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing question in request"}), 400
    
    question = data["question"]
    top_k = data.get("top_k", 5)  # Default value is 5
    temperature = data.get("temperature", 0.7)  # Default value is 0.7

    try:
        # Process the question using llm service
        result = llm.generate_answer(question, top_k=top_k, temperature=temperature)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route('/tts-chat', methods=['POST'])
def tts_chat():
    """
    **Chat + TTS Endpoint**
    1️ Generates answer using LLM model.
    2️ Converts text to speech using ElevenLabs API.
    3️ Returns MP3 audio data.
    4️ Sends answer and references in JSON format in the `Response` header.
    """
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing question in request"}), 400

    question = data["question"]
    top_k = data.get("top_k", 5)  # Default value is 5
    temperature = data.get("temperature", 0.7)  # Default value is 0.7
    similarity_threshold = data.get("similarity_threshold", 0.85)

    try:
        #  **1. Generate answer with LLM**
        result = llm.generate_answer(question, top_k=top_k, temperature=temperature, similarity_threshold=similarity_threshold)
        answer_text = result["answer"]
        references = result.get("references", [])

        #  **2. Generate speech using TTS**
        audio_bytes = tts.text_to_speech(answer_text)

        #  **3. Prepare MP3 file using BytesIO**
        audio_stream = io.BytesIO(audio_bytes)
        audio_stream.seek(0)  # **Must seek to the beginning!**

        #  **4. Create JSON metadata**
        metadata = json.dumps({
            "answer": answer_text,
            "references": references
        })

        # 📡 **5. Send MP3 and JSON via Response**
        response = Response(audio_stream, mimetype="audio/mpeg")
        response.headers["Metadata"] = metadata  # Add JSON to header
        response.headers["Content-Disposition"] = "attachment; filename=response.mp3"

        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@chatbot_bp.route('/get-prompts', methods=['GET'])
def get_prompts():
    """ Returns the current prompt settings. """
    return jsonify(prompt_manager.load_prompts())

@chatbot_bp.route('/update-prompts', methods=['POST'])
def update_prompts():
    """ Updates the prompt settings. """
    data = request.get_json()
    if "adjustment_instructions" not in data:
        return jsonify({"error": "Missing adjustment_instructions"}), 400
    
    prompt_manager.save_prompts(data)
    return jsonify({"success": True, "message": "Prompt updated successfully!"})
