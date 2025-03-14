import streamlit as st
import requests
import json
from frontend_config import BACKEND_URL

# BACKEND_URL: Backend API address
BACKEND_URL = BACKEND_URL +("/api/chatbot")

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🎙️ Video AI Chatbot")

# Navigation options: Two Chat pages, Uploaded Files, Edit Prompts, and Upload Video.
page = st.sidebar.radio(
    "Navigation", 
    ["💬 Chatbot (TTS)", "💬 Chatbot (Text)", "📂 Uploaded Files", "📝 Edit Prompts", "📹 Upload Video"]
)

# ----------------------------
# 💬 Chatbot (TTS) Page
# ----------------------------
if page == "💬 Chatbot (TTS)":
    st.header("💬 AI Chatbot (TTS)")
    st.write("Type your question. The LLM will answer by utilizing relevant transcripts (audio response).")

    with st.sidebar.expander("⚙️ Chat Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
        top_k = st.slider("Top-K (Number of results)", 1, 10, 3)
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.01)
    question = st.chat_input("Enter your question...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        try:
            chat_endpoint = f"{BACKEND_URL}/tts-chat"
            payload = {
                "question": question,
                "temperature": temperature,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold
            }
            response = requests.post(chat_endpoint, json=payload)
            if response.status_code == 200:
                audio_bytes = response.content
                metadata = response.headers.get("Metadata", "{}")
                metadata = json.loads(metadata)
                answer = metadata.get("answer", "No answer returned.")
                references = metadata.get("references", [])
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    if references:
                        st.markdown("### 🔗 References:")
                        for ref in references:
                            st.markdown(f"- {ref}")
            else:
                st.error(f"⚠️ Backend error: {response.status_code} {response.text}")
        except Exception as e:
            st.error(f"❌ Error occurred: {e}")

# ----------------------------
# 💬 Chatbot (Text) Page
# ----------------------------
elif page == "💬 Chatbot (Text)":
    st.header("💬 AI Chatbot (Text)")
    st.write("Type your question. The LLM will answer by utilizing relevant transcripts (text-only response).")

    with st.sidebar.expander("⚙️ Chat Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05, key="text_temp")
        top_k = st.slider("Top-K (Number of results)", 1, 10, 3, key="text_topk")
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.01, key="text_thresh")
    question = st.text_input("Enter your question...", key="text_question")

    if question:
        with st.spinner("Generating answer..."):
            try:
                chat_endpoint = f"{BACKEND_URL}/chat"
                payload = {
                    "question": question,
                    "temperature": temperature,
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold
                }
                response = requests.post(chat_endpoint, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "No answer returned.")
                    references = result.get("references", [])
                    st.markdown("**Answer:**")
                    st.write(answer)
                    if references:
                        st.markdown("### References:")
                        for ref in references:
                            st.markdown(f"- {ref}")
                else:
                    st.error(f"⚠️ Backend error: {response.status_code} {response.text}")
            except Exception as e:
                st.error(f"❌ Error occurred: {e}")

# ----------------------------
# 📂 Uploaded Files Page
# ----------------------------
elif page == "📂 Uploaded Files":
    st.header("📂 Uploaded Transcripts")
    try:
        list_endpoint = f"{BACKEND_URL}/list-transcripts"
        response = requests.get(list_endpoint)
        if response.status_code == 200:
            transcripts = response.json().get("transcripts", [])
        else:
            transcripts = []
        if not transcripts:
            st.info("⚠️ No transcripts found.")
        else:
            for transcript in transcripts:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 {transcript}")
                with col2:
                    if st.button("🗑 Delete", key=transcript):
                        delete_endpoint = f"{BACKEND_URL}/delete-transcript"
                        delete_payload = {"filename": transcript}
                        delete_response = requests.post(delete_endpoint, json=delete_payload)
                        if delete_response.status_code == 200:
                            st.success(f"✅ {transcript} deleted successfully!")
                        else:
                            st.error(f"⚠️ Could not delete {transcript}")
    except Exception as e:
        st.error(f"❌ Error occurred: {e}")

# ----------------------------
# 📝 Edit Prompts Page
# ----------------------------
elif page == "📝 Edit Prompts":
    st.header("📝 Prompt Settings")

    def fetch_prompt():
        """
        Fetches the current prompt adjustment instructions from the backend.
        
        Returns:
            str: The current prompt adjustment instructions.
        """
        response = requests.get(f"{BACKEND_URL}/get-prompts")
        if response.status_code == 200:
            return response.json().get("adjustment_instructions", "")
        else:
            return "⚠️ Error fetching prompt from backend."

    default_prompt = fetch_prompt()
    updated_prompt = st.text_area("Edit Prompt Adjustments", value=default_prompt, height=200)
    if st.button("Update Prompt"):
        update_payload = {"adjustment_instructions": updated_prompt}
        update_response = requests.post(f"{BACKEND_URL}/update-prompts", json=update_payload)
        if update_response.status_code == 200:
            st.success("✅ Prompt updated successfully!")
        else:
            st.error(f"⚠️ Error updating prompt: {update_response.text}")

# ----------------------------
# 📹 Upload Video Page
# ----------------------------
elif page == "📹 Upload Video":
    st.header("🎥 Video Upload & Transcription")
    st.write("Enter the YouTube video URL and get the audio transcription.")
    video_url = st.text_input("Enter the YouTube video URL:")
    if st.button("Upload Video"):
        if video_url:
            try:
                st.write("📤 Sending URL to backend for transcription...")
                endpoint = f"{BACKEND_URL}/upload-youtube"
                payload = {"video_url": video_url}
                response = requests.post(endpoint, json=payload)
                if response.status_code == 200:
                    st.success("✅ Transcript received successfully!")
                else:
                    st.error(f"⚠️ Backend error: {response.status_code} {response.text}")
            except Exception as e:
                st.error(f"❌ Error occurred: {e}")
        else:
            st.warning("⚠️ Please enter a valid video URL.")
