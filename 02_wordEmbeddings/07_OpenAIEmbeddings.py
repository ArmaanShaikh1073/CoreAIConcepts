
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── 3. OpenAI embeddings (what you use for GPT-based RAG) ────────────────────
from openai import OpenAI

client = OpenAI()  # needs OPENAI_API_KEY env var

def get_openai_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    """
    Production tip: batch your requests — don't call this in a loop one-by-one.
    The API accepts up to 2048 inputs per request.
    """
    text = text.replace("\n", " ")  # newlines can affect embeddings
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Batch version (always use this in production)
def get_openai_embeddings_batch(texts: list[str], model="text-embedding-3-small"):
    texts = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)] 

# code intention - this is the function you'll actually use at work, not the single-text version
# what it does - it takes a list of strings and returns a list of embeddings, one per string, in the same order
