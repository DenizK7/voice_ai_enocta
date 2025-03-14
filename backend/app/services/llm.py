import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from app.services.vector_db import search_transcripts, get_embedding, vector_db_instance
from app.services.web_search import web_search
from app.services.prompt_manager import PromptManager
from config import Config
from langchain.utils.math import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

prompt_manager = PromptManager()
client = OpenAI()

def format_time(seconds: float) -> str:
    """
    Convert seconds to a "MM:SS" format.
    
    Args:
        seconds (float): The time in seconds.
    
    Returns:
        str: The formatted time.
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def get_all_transcripts() -> dict:
    """
    Retrieve all transcripts from the FAISS database.
    
    Returns:
        dict: A dictionary containing the transcripts.
    """
    logging.info("Fetching all transcripts from FAISS database.")
    return {"results": vector_db_instance.metadata}

def hybrid_search(question: str, top_k: int = 3, similarity_threshold: float = 0.78, max_results: int = 3) -> dict:
    """
    Perform a hybrid search using FAISS (dense) and BM25 (sparse).
    
    Uses FAISS to get dense embeddings and filters results based on cosine similarity.
    If insufficient FAISS results are found, performs a BM25 search.
    Results are weighted 70% for FAISS and 30% for BM25 and up to max_results are returned.
    
    Args:
        question (str): The search query.
        top_k (int): Number of top FAISS results to retrieve.
        similarity_threshold (float): Minimum similarity score for FAISS results.
        max_results (int): Maximum number of results to return.
    
    Returns:
        dict: A dictionary containing the hybrid search results.
    """
    logging.info("Starting hybrid search.")
    try:
        results = search_transcripts(question, k=top_k)
        logging.info("FAISS search completed successfully.")
    except Exception as e:
        logging.error(f"Error during transcript search: {e}")
        results = {"results": []}

    faiss_ranked = []
    if results["results"]:
        for res in results["results"]:
            similarity_score = res["similarity"]
            if similarity_score >= similarity_threshold:
                faiss_ranked.append(res)
        faiss_ranked = sorted(faiss_ranked, key=lambda x: x["similarity"], reverse=True)[:max_results]
        logging.info(f"FAISS returned {len(faiss_ranked)} results above the threshold.")
    else:
        logging.info("No results found from FAISS search.")

    bm25_ranked = []
    transcripts = get_all_transcripts()["results"]
    if transcripts:
        bm25_corpus = [doc["text"] for doc in transcripts]
        tokenized_corpus = [doc.split() for doc in bm25_corpus]
        bm25_index = BM25Okapi(tokenized_corpus)
        tokenized_query = question.split()
        bm25_scores = bm25_index.get_scores(tokenized_query)
        bm25_threshold = 8
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
        logging.info(f"BM25 returned {len(bm25_ranked)} results above the threshold.")
    else:
        logging.info("No transcripts available for BM25 search.")

    hybrid_results = []
    for item in faiss_ranked:
        hybrid_results.append({
            "text": item["text"],
            "video_title": item.get("video_title", ""),
            "video_url": item.get("video_url", "")
        })
    for item in bm25_ranked:
        hybrid_results.append({
            "text": item["text"]
        })

    if not hybrid_results:
        logging.info("Hybrid search produced no results; falling back to web search.")
        return False

    logging.info(f"Hybrid search produced a total of {len(hybrid_results)} results.")
    return {"results": hybrid_results}

def generate_answer(
    question: str, 
    max_results: int = 3, 
    similarity_threshold: float = 0.77, 
    temperature: float = 0.7, 
    top_k: int = 3
) -> dict:
    """
    Generate an answer by combining hybrid search results with a prompt optimization process.
    
    1. Perform hybrid search (FAISS + BM25) to collect context.
    2. Optimize the prompt using the PromptManager.
    3. Generate the answer using the GPT-4o mini model.
    
    Args:
        question (str): The query to answer.
        max_results (int): Maximum number of context results.
        similarity_threshold (float): Similarity threshold for filtering results.
        temperature (float): Temperature setting for the language model.
        top_k (int): Number of top FAISS results to retrieve.
    
    Returns:
        dict: A dictionary containing the answer and references.
    """
    logging.info("Generating answer for the question.")
    hybrid_results = hybrid_search(
        question, top_k=top_k, 
        similarity_threshold=similarity_threshold, 
        max_results=max_results,
    )
    if hybrid_results is False:
        logging.info("Falling back to web search.")
        web_results = web_search(question, max_results=max_results)
        web_context_lines = []
        for res in web_results.get("results", []):
            snippet = res.get("snippet", "")
            web_context_lines.append(snippet)
        new_base_prompt = prompt_manager.create_base_prompt(web_context_lines, question)
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
        logging.info("Answer generated using web search fallback.")
    else:
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
        references = []
        for res in hybrid_results["results"]:
            if res.get("video_title") and res.get("video_url"):
                ref = f'{res["video_title"]} ({res["video_url"]})'
                references.append(ref)
            else:
                references.append(res["text"])
        logging.info("Answer generated using hybrid search context.")
    
    deduped_references = list(dict.fromkeys(references))
    return {"answer": answer, "references": deduped_references}
