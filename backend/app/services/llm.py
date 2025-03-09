import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from app.services.vector_db import search_transcripts, get_embedding, vector_db_instance
from app.services.web_search import web_search  # Web search using Brave API
from app.services.prompt_manager import PromptManager  # New prompt management
from config import Config

prompt_manager = PromptManager()
client = OpenAI()

def format_time(seconds: float) -> str:
    """
    Converts seconds to a "minute:second" format.
    
    Args:
        seconds (float): The time in seconds.
    
    Returns:
        str: The formatted time as "MM:SS".
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculates the normalized cosine similarity between two vectors.
    
    Args:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.
    
    Returns:
        float: The cosine similarity between the two vectors.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1 / norm1, vec2 / norm2)

def get_all_transcripts() -> dict:
    """
    Returns all transcripts in the FAISS database.
    This is used to create the BM25 model.
    
    Returns:
        dict: A dictionary containing the transcripts.
    """
    return {"results": vector_db_instance.metadata}

def hybrid_search(question: str, top_k: int = 3, similarity_threshold: float = 0.85, max_results: int = 3) -> dict:
    """
    Hybrid Search: Uses FAISS (Dense) + BM25 (Sparse).
    - Filters FAISS results using cosine similarity.
    - If there are not enough FAISS results, a BM25 search is performed.
    - Results are weighted 70% for FAISS and 30% for BM25.
    - Returns up to max_results results.
    
    Args:
        question (str): The question to search for.
        top_k (int): The number of top results to fetch from FAISS.
        similarity_threshold (float): The similarity threshold for FAISS results.
        max_results (int): The maximum number of results to return.
    
    Returns:
        dict: A dictionary containing the hybrid search results.
    """
    # FAISS search
    try:
        faiss_results = search_transcripts(question, k=top_k)
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        faiss_results = {"results": []}

    faiss_ranked = []
    if faiss_results["results"]:
        question_embedding = get_embedding(question)
        for res in faiss_results["results"]:
            try:
                segment_embedding = np.array(res["embedding"])
                score = cosine_similarity(question_embedding, segment_embedding)
                res["similarity"] = score
                if score >= similarity_threshold:
                    faiss_ranked.append({
                        "text": res["text"],
                        "similarity": score,
                        "video_title": res.get("video_title", ""),
                        "video_url": res.get("video_url", ""),
                        "start": res.get("start", 0),
                        "source": "faiss"
                    })
            except Exception as e:
                print(f"Error calculating cosine similarity: {e}")
        faiss_ranked = sorted(faiss_ranked, key=lambda x: x["similarity"], reverse=True)[:max_results]

    # BM25 search: If FAISS results are insufficient
    bm25_ranked = []
    transcripts = get_all_transcripts()["results"]
    if transcripts:
        bm25_corpus = [doc["text"] for doc in transcripts]
        tokenized_corpus = [doc.split() for doc in bm25_corpus]
        bm25_index = BM25Okapi(tokenized_corpus)
        tokenized_query = question.split()
        bm25_scores = bm25_index.get_scores(tokenized_query)
        bm25_threshold = 4  # Only include scores above a certain threshold
        bm25_top_indices = np.argsort(bm25_scores)[::-1]
        for i in bm25_top_indices:
            if bm25_scores[i] >= bm25_threshold:
                bm25_ranked.append({
                    "text": bm25_corpus[i],
                    "similarity": bm25_scores[i],
                    "source": "bm25"
                })
        num_needed = max_results - len(faiss_ranked)
        if num_needed > 0:
            bm25_ranked = bm25_ranked[:num_needed]
        else:
            bm25_ranked = []

    # Combine hybrid results: 70% from FAISS, 30% from BM25
    hybrid_results = []
    for item in faiss_ranked:
        hybrid_results.append({
            "text": item["text"],
            "weighted_score": item["similarity"] * 0.7,
            "video_title": item.get("video_title", ""),
            "video_url": item.get("video_url", ""),
            "start": item.get("start", 0),
            "source": item["source"]
        })
    for item in bm25_ranked:
        hybrid_results.append({
            "text": item["text"],
            "weighted_score": item["similarity"] * 0.3,
            "source": item["source"]
        })
    hybrid_results = sorted(hybrid_results, key=lambda x: x["weighted_score"], reverse=True)[:max_results]

    # If hybrid results are empty, fallback to web search
    if not hybrid_results:
        web_results = web_search(question, max_results=max_results)
        hybrid_results = []
        for res in web_results.get("results", []):
            hybrid_results.append({
                "text": res["title"],
                "video_url": res["url"],
                "source": "web"
            })

    return {"results": hybrid_results}

def generate_answer(
    question: str, 
    max_results: int = 3, 
    similarity_threshold: float = 0.85, 
    temperature: float = 0.7, 
    top_k: int = 3
) -> dict:
    """
    1. Uses hybrid search (FAISS + BM25) to gather context.
    2. Optimizes the prompt with PromptManager.
    3. Generates the answer using the GPT-4o mini model.
    4. Returns the answer and references.
    
    Args:
        question (str): The question to be answered.
        max_results (int): The maximum number of results to return.
        similarity_threshold (float): The threshold for similarity filtering.
        temperature (float): The temperature for GPT-4o mini model.
        top_k (int): The number of top results to fetch from FAISS.
    
    Returns:
        dict: A dictionary containing the answer and references.
    """
    # Get hybrid search results
    hybrid_results = hybrid_search(
        question, top_k=top_k, 
        similarity_threshold=similarity_threshold, 
        max_results=max_results
    )
    # Combine the texts of the hybrid results as context
    context = "\n".join([res["text"] for res in hybrid_results["results"]])
    
    
    base_prompt = prompt_manager.create_base_prompt(context, question)
    final_prompt = prompt_manager.optimize_prompt(base_prompt)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[ 
            {"role": "system", "content": "You are a helpful assistant that provides answers based on the provided context."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=temperature,
        top_p=0.9
    )
    
    answer = response.choices[0].message.content

    # Format references
    references = []
    for res in hybrid_results["results"]:
        if res.get("source") == "faiss":
            # If from FAISS, use video info if available
            if res.get("video_title") and res.get("video_url"):
                start_time = res.get("start", 0)
                time_str = format_time(start_time)
                ref = f'{res["video_title"]} ({res["video_url"]}) at {time_str} '
                references.append(ref)
            else:
                references.append(res["text"])
        elif res.get("source") == "bm25":
            # For BM25 results, use text and video URL if available
            if res.get("video_url"):
                references.append(f'{res["text"]} ({res["video_url"]})')
            else:
                references.append(res["text"])
        else:
            references.append(res["text"])

    if answer.strip().lower().startswith("üzgünüm"):
        print("LLM response started with 'Sorry'; resetting context and retrying with web search results...")
        web_results = web_search(question, max_results=max_results)
        web_context_lines = [res.get("title", "") for res in web_results.get("results", [])]
        new_context = "\n".join(web_context_lines)
        new_base_prompt = prompt_manager.create_base_prompt(new_context, question)
        new_final_prompt = prompt_manager.optimize_prompt(new_base_prompt)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides answers based on the provided context."},
                {"role": "user", "content": new_final_prompt}
            ],
            temperature=temperature,
            top_p=0.9
        )
        answer = response.choices[0].message.content
        references = [f'{res.get("title", "")} ({res.get("url", "No URL")})' for res in web_results.get("results", [])]

    return {"answer": answer, "references": references}
