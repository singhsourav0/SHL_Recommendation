import faiss
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List

from rag_recommender.modules.generate_embeddings import generate_embedding

# ------------------ Setup ------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

INDEX_PATH = Path("vector.index")
TEXTS_PATH = Path("vector_texts.pkl")


# ------------------ Loader ------------------
def load_index_and_texts():
    """
    Load the FAISS index and corresponding texts.
    """
    if not INDEX_PATH.exists() or not TEXTS_PATH.exists():
        raise FileNotFoundError("Index or text file not found. Please build the index first.")

    index = faiss.read_index(str(INDEX_PATH))
    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)
    return index, texts


# ------------------ Search ------------------
def search_assessments(user_query: str, top_k: int = 10) -> List[str]:
    """
    Embed the user query and search top_k relevant assessments.
    """
    query_vector = np.array([generate_embedding(user_query)], dtype="float32")
    index, texts = load_index_and_texts()
    _, indices = index.search(query_vector, top_k)
    results = [texts[i] for i in indices[0]]
    return results


# ------------------ Debug/Test ------------------
if __name__ == "__main__":
    sample_query = (
        "I am hiring for Java developers who can also collaborate effectively with my "
        "business teams. Looking for an assessment(s) that can be completed in 40 minutes."
    )

    logging.info("Performing semantic search for assessments...")
    try:
        recommendations = search_assessments(sample_query)
        print("\nTop Recommendations:\n")
        for i, res in enumerate(recommendations, 1):
            print(f"{i}. {res}")
    except FileNotFoundError as e:
        logging.error(e)
