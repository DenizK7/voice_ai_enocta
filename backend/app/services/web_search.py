import requests
from brave import Brave
from brave.goggles import thought_leadership
from config import Config  # API_KEY is retrieved from config

# Initialize the Brave API Client
brave = Brave()

def web_search(query: str, max_results: int = 3) -> dict:
    """
    Performs a search using the Brave Search API with the `thought_leadership` Goggle.
    
    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return (default is 3).
    
    Returns:
        dict: A dictionary containing the search results with titles, URLs, and snippets.
    """
    try:
        # Perform search with Brave Search
        search_results = brave.search(q=query, count=max_results)

        # Retrieve web results
        web_results = search_results.web_results

        if not web_results:
            return {"results": []}  # Return an empty list if no results

        # Return formatted results
        return {
            "results": [
                {
                    "title": result["title"],  # Access title
                    "url": result["url"],  # Access URL
                    "snippet": result.get("description", "No description available")  # Fallback if no description
                }
                for result in web_results
            ]
        }

    except Exception as e:
        print(f"⚠️ Brave Search API Error: {e}")
        return {"results": []}
