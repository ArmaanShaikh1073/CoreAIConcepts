# naive_rag_complete.py
# The baseline system — understand this before anything advanced
# We'll build every improvement on top of this foundation

import os
#from openai import OpenAI
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import tiktoken
from dotenv import load_dotenv

# Automatically look for a .env file and load variables into os.environ
load_dotenv()

# Safe initialization fallback
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("CRITICAL: GROQ_API_KEY is missing! Check your .env file or environment variables.")

client_groq = Groq(api_key=api_key)


embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant      = QdrantClient(":memory:")
tokenizer   = tiktoken.get_encoding("cl100k_base")

COLLECTION = "naive_rag"
DIM        = 384

# ── INGESTION PIPELINE ────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Naive fixed-size chunking. Works for demos. Breaks for real docs."""
    tokens = tokenizer.encode(text)
    chunks = []
    start  = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end]))
        start = end - overlap
        if end == len(tokens):
            break
    return chunks

def ingest(documents: list[dict]):
    """documents = [{"text": ..., "source": ...}]"""
    # qdrant.recreate_collection(
    #     collection_name=COLLECTION,
    #     vectors_config=VectorParams(size=DIM, distance=Distance.DOT)
    # )

    # 1. Check if the collection already exists in memory/disk
    if qdrant.collection_exists(collection_name=COLLECTION):
        # 2. Hard delete it to clear out legacy distance metrics (like COSINE)
        qdrant.delete_collection(collection_name=COLLECTION)
        print(f"Cleared existing legacy collection: '{COLLECTION}'")

    # 3. Create a pristine collection using the correct DOT distance metric
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT)
    )


    points = []
    point_id = 0
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            vec = embed_model.encode(chunk, normalize_embeddings=True).tolist()
            points.append(PointStruct(
                id=point_id,
                vector=vec,
                payload={"text": chunk, "source": doc["source"]}
            ))
            point_id += 1
    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"Indexed {point_id} chunks from {len(documents)} documents.")

# ── RETRIEVAL ─────────────────────────────────────────────────────────────────

# def retrieve(query: str, top_k: int = 5) -> list[dict]:
#     """
#     Embed the query, search the vector DB, return top-k chunks.
#     This is where naive RAG most commonly fails:
#       - Query too short → vague embedding → wrong chunks
#       - top_k too small → miss the relevant chunk
#       - top_k too large → noisy context → LLM gets confused
#     """
#     q_vec = embed_model.encode(query, normalize_embeddings=True).tolist()
#     results = qdrant.query_points(
#         collection_name=COLLECTION,
#         query=q_vec,
#         limit=top_k,
#         with_payload=True
#     )
#     return [{"text": r.payload["text"],
#              "source": r.payload["source"],
#              "score": r.score} for r in results]

    # ── REWRITTEN RETRIEVAL FUNCTION ─────────────────────────────────────────────

# def retrieve(query: str, top_k: int = 5) -> list[dict]:
#     """
#     Embed the query, search the vector DB, return top-k chunks.
#     Fixed to prevent tuple unpacking AttributeError.
#     """
#     q_vec = embed_model.encode(query, normalize_embeddings=True).tolist()
#     results = qdrant.query_points(
#         collection_name=COLLECTION,
#         query=q_vec,
#         limit=top_k,
#         with_payload=True
#     )
    
#     cleaned_results = []
#     for r in results:
#         # If Qdrant returned a named tuple / ScoredPoint object
#         if hasattr(r, 'payload') and r.payload:
#             cleaned_results.append({
#                 "text": r.payload.get("text", ""),
#                 "source": r.payload.get("source", "unknown"),
#                 "score": r.score
#             })
#         # Fallback: If Qdrant packed it as a vanilla tuple (id, score, payload, etc.)
#         else:
#             # Safely check if r behaves like a tuple/list index matching Qdrant schema
#             try:
#                 # In standard tuple fallbacks, payload is often at index 2 or 3 
#                 # Let's inspect or safely extract if it's an unmapped structural tuple
#                 payload = r[2] if len(r) > 2 else {}
#                 score = r[1] if len(r) > 1 else 0.0
#                 cleaned_results.append({
#                     "text": payload.get("text", "") if isinstance(payload, dict) else str(payload),
#                     "source": payload.get("source", "unknown") if isinstance(payload, dict) else "unknown",
#                     "score": score
#                 })
#             except Exception:
#                 # Ultimate fallback if it's a completely flat tuple structure
#                 cleaned_results.append({
#                     "text": str(r),
#                     "source": "unknown",
#                     "score": 0.0
#                 })
#                 print(f"Warning: Unrecognized result format from Qdrant: {r}")
                
#     return cleaned_results



def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the query, search the vector DB, and safely extract payloads.
    Guarantees text string extraction to prevent empty context fields.
    """
    q_vec = embed_model.encode(query, normalize_embeddings=True).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=q_vec,
        limit=top_k,
        with_payload=True
    )
    
    cleaned_results = []
    for r in results:
        # 1. Extract the score safely
        score = getattr(r, 'score', 0.0)
        if isinstance(r, tuple) and len(r) > 1:
            score = r[1]

        # 2. Extract the payload container safely
        payload = None
        if hasattr(r, 'payload'):
            payload = r.payload
        elif isinstance(r, tuple) and len(r) > 2:
            payload = r[2]
            
        # 3. Extract strings from the payload container
        if payload:
            # Handle both dict lookups and object attribute lookups safely
            if isinstance(payload, dict):
                text_content = payload.get("text", "")
                source_content = payload.get("source", "unknown")
            else:
                text_content = getattr(payload, "text", "")
                source_content = getattr(payload, "source", "unknown")
        else:
            # If the entire record fell back to a raw string representation
            text_content = str(r)
            source_content = "unknown"

        # Only append if we actually captured text context
        if text_content.strip():
            cleaned_results.append({
                "text": text_content,
                "source": source_content,
                "score": score
            })
            
    return cleaned_results

# ── GENERATION ────────────────────────────────────────────────────────────────

def generate(query: str, context_chunks: list[dict]) -> dict:
    """
    Pack retrieved chunks into context, send to LLM.
    Failure mode: context ordering matters. LLMs pay most
    attention to beginning and end — put most relevant chunks there.
    """
    context = "\n\n---\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in context_chunks
    ])

    system_prompt = """You are a helpful assistant. Answer the user's question
using ONLY the provided context. If the context doesn't contain enough
information to answer, say "I don't have enough information to answer this."
Do not use any prior knowledge outside the context."""

    user_message = f"""Context:
{context}

Question: {query}

Answer:"""

    response = client_groq.chat.completions.create(
        model    = "llama-3.3-70b-versatile",
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature = 0,   # 0 = deterministic, best for factual RAG
    )
    return {
        "answer"       : response.choices[0].message.content,
        "tokens_used"  : response.usage.total_tokens,
        "chunks_used"  : len(context_chunks),
    }

# ── FULL PIPELINE ─────────────────────────────────────────────────────────────

def rag_query(query: str, top_k: int = 5) -> dict:
    chunks = retrieve(query, top_k=top_k)
    print(f"DEBUG: Retrieved {len(chunks)} chunks from DB for query '{query}'.")

    result = generate(query, chunks)
    result["retrieved_chunks"] = chunks
    return result

# ── TEST ──────────────────────────────────────────────────────────────────────
documents = [
    {"source": "product_docs.md",
     "text": """Our enterprise plan includes unlimited API calls, dedicated support,
     custom SLAs, and SSO integration. Pricing starts at $2000/month.
     Refunds are available within 30 days for annual plans."""},
    {"source": "tech_guide.md",
     "text": """RAG pipelines combine vector search with LLM generation.
     The retrieval quality directly determines answer quality.
     Common improvements include query rewriting, reranking, and hybrid search."""},
     {"source": "cooking_blog.md",
     "text": """For a quick and delicious meal, try our 20-minute chicken stir-fry recipe.
      It's packed with veggies and a savory sauce. Perfect for busy weeknights!"""},
    {
        "source": "travel_guide.md",
        "text": """The best time to visit Paris is from April to June or October to early November.
        Avoid the peak summer months for a more enjoyable experience with fewer crowds."""
    }, 
    {
        "source": "finance_report.md",
        "text": """The global economy is expected to grow by 3% in 2024, with technology and healthcare sectors leading the way. Inflation rates are projected to stabilize around 2% by the end of the year."""
    }
]

ingest(documents)
result = rag_query("What is the enterprise plan price?")
print(f"Answer: {result['answer']}")
print(f"Tokens used: {result['tokens_used']}")

# Now let's see it FAIL — expose the core weakness
bad_query = "tell me about the honey"  # vague query → vague retrieval
bad_result = rag_query(bad_query)
print(f"\nBad query answer: {bad_result['answer']}")
# It might still answer, but the retrieved chunks will be wrong
# because "money" matches too many things semantically


'''
____ Output ____________________________________________________________________________________________________

Indexed 2 chunks from 2 documents.
DEBUG: Retrieved 1 chunks from DB for query 'What is the enterprise plan price?'.
Answer: The enterprise plan price starts at $2000/month.
Tokens used: 294
DEBUG: Retrieved 1 chunks from DB for query 'tell me about the money'.

Bad query answer: The text mentions that the pricing for the enterprise plan starts at $2000/month, and refunds are available within 30 days for annual plans.

DEBUG: Retrieved 1 chunks from DB for query 'tell me about the honey'.

Bad query answer: I don't have enough information to answer this.
'''
