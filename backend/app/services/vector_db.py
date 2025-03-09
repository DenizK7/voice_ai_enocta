import os
import json
import numpy as np
import faiss
from openai import OpenAI
from config import Config  # FAISS_SAVE_FOLDER should be defined in Config

# Create the OpenAI client.
client = OpenAI()

# The OpenAI embedding model "text-embedding-ada-002" produces 1536-dimensional embeddings.
EMBEDDING_DIM = 1536

# Check if the save folder exists, if not, create it.
save_folder = Config.FAISS_SAVE_FOLDER
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

INDEX_FILE = os.path.join(save_folder, "faiss_index.index")
METADATA_FILE = os.path.join(save_folder, "metadata.json")

class VectorDB:
    def __init__(self, dimension: int, index_file=INDEX_FILE, metadata_file=METADATA_FILE):
        """
        Initializes the VectorDB with given parameters.
        
        Args:
            dimension (int): The dimension of the embeddings.
            index_file (str): The path to the FAISS index file.
            metadata_file (str): The path to the metadata JSON file.
        """
        self.dimension = dimension
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

        # Load existing files if they exist
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            # Ensure that the number of metadata matches the number of vectors in the index.
            if self.index.ntotal != len(self.metadata):
                print("Warning: Index and metadata mismatch. Rebuilding...")
                self.index = faiss.IndexFlatL2(dimension)
                self.metadata = []

    def add_transcript(self, embedding: np.ndarray, meta: dict):
        """
        Adds a new embedding and its metadata to the FAISS database.
        
        Args:
            embedding (np.ndarray): The embedding to be added.
            meta (dict): The metadata associated with the embedding.
        """
        embedding = np.array([embedding], dtype=np.float32)
        self.index.add(embedding)
        self.metadata.append(meta)
        self.save()  # Save data after adding.

    def search(self, query_embedding: np.ndarray, k: int = 3) -> dict:
        """
        Searches the FAISS database for the k most similar embeddings to the provided query.
        
        Args:
            query_embedding (np.ndarray): The query embedding to search for.
            k (int): The number of results to return.
        
        Returns:
            dict: A dictionary containing the search results.
        """
        if self.index.ntotal == 0:
            print("⚠️ FAISS index is empty, search cannot be performed.")
            return {"results": []}

        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, k)

        if len(indices) == 0 or len(indices[0]) == 0:
            print("⚠️ No matching results found in FAISS index.")
            return {"results": []}

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):  # Avoid out-of-range errors
                results.append(self.metadata[idx])

        return {"results": results}


    def _delete_by_video_url(self, video_url: str) -> bool:
        """
        Deletes all embeddings and metadata related to a specific video URL from the FAISS index.
        
        Args:
            video_url (str): The URL of the video to delete.
        
        Returns:
            bool: Whether the deletion was successful.
        """
        indexes_to_remove = [
            idx for idx, meta in enumerate(self.metadata) if meta.get("video_url") == video_url
        ]

        if not indexes_to_remove:
            return True

        remaining_embeddings = [
            np.array(meta["embedding"], dtype=np.float32)
            for idx, meta in enumerate(self.metadata) if idx not in indexes_to_remove
        ]

        self.index = faiss.IndexFlatL2(self.dimension)  
        if remaining_embeddings:
            self.index.add(np.array(remaining_embeddings, dtype=np.float32))  # Create a new index

        self.metadata = [meta for idx, meta in enumerate(self.metadata) if idx not in indexes_to_remove]

        self.save()

        return True  


    def save(self):
        """
        Saves the current state of the FAISS index and metadata to files.
        """
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

vector_db_instance = VectorDB(EMBEDDING_DIM)

def get_all_transcripts() -> dict:
    """
    Returns all transcripts stored in the vector database.
    
    Returns:
        dict: A dictionary containing all the stored transcripts.
    """
    return {"results": vector_db_instance.metadata}

def get_embedding(text: str) -> np.ndarray:
    """
    Generates an embedding for the given text using the OpenAI embedding model.
    
    Args:
        text (str): The text to generate an embedding for.
    
    Returns:
        np.ndarray: The generated embedding, normalized.
    """
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    embedding = np.array(response.data[0].embedding, dtype=np.float32)
    return embedding / np.linalg.norm(embedding)  

def add_transcript_to_db(transcript_data: dict) -> bool:
    """
    Adds a transcript (with segments) to the vector database.
    
    Args:
        transcript_data (dict): The transcript data to be added.
    
    Returns:
        bool: Whether the addition was successful.
    """
    video_meta = transcript_data.get("video_metadata", {})
    segments = transcript_data.get("segments", [])
    
    for seg in segments:
        emb = get_embedding(seg["text"])
        meta = {
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "video_title": video_meta.get("title", ""),
            "video_url": video_meta.get("url", ""),
            "embedding": emb.tolist()
        }
        vector_db_instance.add_transcript(emb, meta)
    return True

def search_transcripts(query: str, k: int = 3) -> dict:
    """
    Searches the vector database for the k most similar transcripts to the provided query.
    
    Args:
        query (str): The query to search for.
        k (int): The number of results to return.
    
    Returns:
        dict: A dictionary containing the search results.
    """
    query_embedding = get_embedding(query)
    return vector_db_instance.search(query_embedding, k=k)

def delete_by_video_url(video_url: str) -> bool:
    """
    Deletes a transcript and its associated embeddings by video URL.
    
    Args:
        video_url (str): The URL of the video to delete.
    
    Returns:
        bool: Whether the deletion was successful.
    """
    return vector_db_instance._delete_by_video_url(video_url)
