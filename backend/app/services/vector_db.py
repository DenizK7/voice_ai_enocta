import os
import json
import numpy as np
import faiss
import logging
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from config import Config  
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

client = OpenAI()

save_folder = Config.FAISS_SAVE_FOLDER
os.makedirs(save_folder, exist_ok=True)

INDEX_FILE = os.path.join(save_folder, "faiss_index.index")
METADATA_FILE = os.path.join(save_folder, "metadata.json")

EMBEDDING_MODEL = "text-embedding-3-large"
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
semantic_chunker = SemanticChunker(embedding_model)

def get_embedding_dim():
    """
    Get the embedding dimension automatically from the OpenAI API.
    """
    logging.info("Getting embedding dimension from OpenAI API.")
    example_text = "Test"
    response = client.embeddings.create(input=[example_text], model=EMBEDDING_MODEL)
    dim = len(response.data[0].embedding)
    logging.info(f"Embedding dimension is {dim}.")
    return dim

EMBEDDING_DIM = get_embedding_dim()

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector using its L2 norm.
    """
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

class VectorDB:
    """
    A vector database using FAISS for inner product similarity search.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index_file = INDEX_FILE
        self.metadata_file = METADATA_FILE
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

        logging.info("Initializing VectorDB.")
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            logging.info("Loading existing FAISS index and metadata from disk.")
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            if self.index.ntotal != len(self.metadata):
                logging.warning("FAISS index and metadata mismatch! Recreating index and metadata.")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = []
        else:
            logging.info("No existing index found. Creating new FAISS index.")

    def add_transcript(self, embedding: np.ndarray, meta: dict):
        """
        Add a transcript's embedding and metadata to the FAISS index.
        """
        logging.info("Adding transcript to FAISS index.")
        embedding = normalize_vector(embedding.astype(np.float32))
        embedding = np.array([embedding], dtype=np.float32)
        self.index.add(embedding)
        self.metadata.append(meta)
        self.save()
        logging.info("Transcript added successfully.")

    def search(self, query_embedding: np.ndarray, k: int = 3) -> dict:
        """
        Search for transcripts in the FAISS index using cosine similarity and logistic transformation.
        """
        logging.info("Performing search in FAISS index.")
        if self.index.ntotal == 0:
            logging.warning("FAISS index is empty! Search cannot be performed.")
            return {"results": []}
        query_embedding = normalize_vector(query_embedding.astype(np.float32))
        query_embedding = np.array([query_embedding], dtype=np.float32)
        similarities, indices = self.index.search(query_embedding, k)
        logging.info(f"Raw similarities: {similarities[0]}")
        
        def logistic_transform(x, k=10, x0=0.3):
            return 1 / (1 + np.exp(-k * (x - x0)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.metadata):
                raw_similarity = similarities[0][i]
                transformed_similarity = logistic_transform(raw_similarity, k=10, x0=0.3)
                meta = self.metadata[idx]
                meta["similarity"] = transformed_similarity
                results.append(meta)
                logging.info(f"Result {i}: Raw similarity: {raw_similarity}, Transformed similarity: {transformed_similarity}")
        return {"results": results}

    def save(self):
        """
        Save the FAISS index and metadata to disk.
        """
        logging.info("Saving FAISS index and metadata to disk.")
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logging.info("Save completed.")

vector_db_instance = VectorDB(EMBEDDING_DIM)

def get_embedding(text: str) -> np.ndarray:
    """
    Retrieve and normalize the embedding for a given text using the OpenAI API.
    """
    logging.info(f"Getting embedding for text: {text[:30]}...")
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    emb = np.array(response.data[0].embedding, dtype=np.float32)
    normalized_emb = normalize_vector(emb)
    logging.info("Embedding retrieved and normalized.")
    return normalized_emb

def extract_word_data(word_str):
    """
    Parse the string-formatted word data to extract speaker and text.
    """
    word_str = str(word_str)
    text_match = re.search(r"text='(.*?)'", word_str)
    speaker_match = re.search(r"speaker_id='(.*?)'", word_str)
    return speaker_match.group(1) if speaker_match else "unknown", text_match.group(1) if text_match else ""

def add_transcript_to_db(transcript_data: dict) -> bool:
    """
    Add transcript data to the FAISS index.
    """
    logging.info("Adding transcript data to database.")
    video_url = transcript_data.get("metada", {}).get("video_url", "")
    title = transcript_data.get("metada", {}).get("title", "")
    words = transcript_data.get("words", [])
    formatted_text = ""
    current_speaker = None
    current_speaker_text = []

    for word in words:
        speaker, text = extract_word_data(word)
        if speaker != current_speaker:
            if current_speaker is not None:
                formatted_text += f"{current_speaker}: {' '.join(current_speaker_text)} "
            current_speaker = speaker
            current_speaker_text = []
        if text.strip():
            current_speaker_text.append(text)

    if current_speaker_text:
        formatted_text += f"{current_speaker}: {' '.join(current_speaker_text)} "

    logging.info("Creating documents using semantic chunker.")
    documents = semantic_chunker.create_documents([formatted_text])

    for doc in documents:
        chunk_text = doc.page_content
        logging.info(f"Processing document chunk: {chunk_text[:30]}...")
        embedding = get_embedding(chunk_text)
        meta = {
            "text": chunk_text,
            "video_title": title,
            "video_url": video_url,
            "embedding": embedding.tolist()
        }
        vector_db_instance.add_transcript(embedding, meta)

    mp3_path = os.path.join(os.path.dirname(os.path.dirname(save_folder)), "uploads", f"{title}.mp3")
    if os.path.exists(mp3_path):
        os.remove(mp3_path)
        logging.info(f"Removed mp3 file: {mp3_path}")
    logging.info("Transcript added to database successfully.")
    return True

def search_transcripts(query: str, k: int = 3) -> dict:
    """
    Search for transcripts in the FAISS index using a query.
    """
    logging.info(f"Searching transcripts for query: {query}")
    query_embedding = get_embedding(query)
    results = vector_db_instance.search(query_embedding, k=k)
    logging.info(f"Search completed. Found {len(results.get('results', []))} results.")
    return results

def delete_by_video_url(video_url: str) -> bool:
    """
    Delete transcripts from the FAISS index that match the given video URL.
    """
    logging.info(f"Deleting transcripts with video URL: {video_url}")
    vector_db_instance.metadata = [meta for meta in vector_db_instance.metadata if meta.get("video_url") != video_url]
    vector_db_instance.save()
    logging.info("Deletion completed.")
    return True
