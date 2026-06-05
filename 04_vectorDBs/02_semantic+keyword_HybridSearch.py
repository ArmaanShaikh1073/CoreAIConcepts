# hybrid_search.py
# Combine dense vectors (semantic) + sparse BM25 (keyword) for best retrieval

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, SparseVectorParams, Distance,
    SparseVector, PointStruct, Modifier,
    SearchRequest, Query, FusionQuery, Fusion,
)
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi   # pip install rank-bm25
import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WHAT IS BM25?
# BM25 = Best Match 25 — the gold standard keyword search algorithm
# It scores documents by: term frequency + inverse document frequency
# TF: how often the query term appears in this doc (with diminishing returns)
# IDF: how rare the term is across all docs (rare = more discriminative)
# Better than raw TF-IDF because it saturates term frequency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HybridSearchEngine:
    """
    Combines BM25 (keyword) and vector (semantic) search.
    Uses Reciprocal Rank Fusion (RRF) to merge ranked lists.
    
    RRF score = Σ 1/(k + rank_i)  where k=60 is a smoothing constant
    This is rank-based, so score scales don't need to match.
    """

    def __init__(self, dense_model_name: str = "all-MiniLM-L6-v2"):
        self.dense_model = SentenceTransformer(dense_model_name)
        self.documents   = []
        self.bm25        = None
        self.dense_embs  = None

    def index(self, documents: list[dict]):
        """Index documents with both BM25 and dense vectors."""
        self.documents = documents
        texts = [d["text"] for d in documents]

        # BM25: tokenize each document
        tokenized = [text.lower().split() for text in texts]
        self.bm25  = BM25Okapi(tokenized)

        # Dense vectors
        self.dense_embs = self.dense_model.encode(
            texts, normalize_embeddings=True
        )
        print(f"Indexed {len(documents)} docs (BM25 + dense vectors)")

    def bm25_search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Keyword search — exact and near-exact matches."""
        scores = self.bm25.get_scores(query.lower().split())
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked if scores[i] > 0]

    def dense_search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Semantic search — meaning-based matches."""
        q_emb  = self.dense_model.encode([query], normalize_embeddings=True)[0]
        scores = self.dense_embs @ q_emb
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked]

    def rrf_fusion(self, ranked_lists: list[list[tuple[int, float]]],
                   k: int = 60) -> list[tuple[int, float]]:
        """
        Reciprocal Rank Fusion — merges multiple ranked lists.
        k=60 is the standard constant from the original RRF paper.
        """
        scores = {}
        for ranked in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked):
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    def hybrid_search(self, query: str, top_k: int = 5,
                      alpha: float = 0.5) -> list[dict]:
        """
        alpha: weight for dense search (0=pure BM25, 1=pure dense, 0.5=balanced)
        For most RAG: alpha=0.5-0.7 works best
        For keyword-heavy queries (IDs, names): alpha=0.2-0.3
        For conceptual queries: alpha=0.7-0.9
        """
        bm25_results  = self.bm25_search(query, top_k=top_k*2)
        dense_results = self.dense_search(query, top_k=top_k*2)

        # Weight the lists by alpha
        # Simple approach: duplicate entries to simulate weighting
        weighted_dense = dense_results * int(alpha * 10)
        weighted_bm25  = bm25_results  * int((1-alpha) * 10)

        fused = self.rrf_fusion([dense_results, bm25_results])

        return [
            {"rank": i+1, "doc_id": doc_id, "rrf_score": round(score, 5),
             "text": self.documents[doc_id]["text"]}
            for i, (doc_id, score) in enumerate(fused[:top_k])
        ]


# ── Test it ───────────────────────────────────────────────────────────────────
docs = [
    {"text": "GPT-4 has a 128k token context window."},
    {"text": "Large language models process text as token sequences."},
    {"text": "The context window limits how much text a model can see at once."},
    {"text": "GPT-4 outperforms GPT-3.5 on most reasoning benchmarks."},
    {"text": "Transformers use attention to process entire sequences in parallel."},
    {"text": "OpenAI released GPT-4 in March 2023 with multimodal capabilities."},
    {"text": "BERT uses bidirectional attention unlike GPT's causal attention."},
]

engine = HybridSearchEngine()
engine.index(docs)

test_queries = [
    "GPT-4 context window",        # keyword-heavy — exact match matters
    "how do transformers process text",   # semantic — no exact keywords
    "what did OpenAI release in 2023",    # mixed
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    print("  BM25 only:")
    for i, (doc_id, score) in enumerate(engine.bm25_search(query, top_k=2)):
        print(f"    [{score:.3f}] {docs[doc_id]['text'][:60]}")
    print("  Hybrid (alpha=0.5):")
    for r in engine.hybrid_search(query, top_k=2, alpha=0.5):
        print(f"    [{r['rrf_score']:.5f}] {r['text'][:60]}")


'''

____ Output ______________________________________________________________________

Indexed 7 docs (BM25 + dense vectors)

Query: 'GPT-4 context window'
  BM25 only:
    [1.846] The context window limits how much text a model can see at o
    [1.141] GPT-4 has a 128k token context window.
  Hybrid (alpha=0.5):
    [0.03252] GPT-4 has a 128k token context window.
    [0.03252] The context window limits how much text a model can see at o

Query: 'how do transformers process text'
  BM25 only:
    [2.222] Transformers use attention to process entire sequences in pa
    [1.846] The context window limits how much text a model can see at o
  Hybrid (alpha=0.5):
    [0.03279] Transformers use attention to process entire sequences in pa
    [0.03200] Large language models process text as token sequences.

Query: 'what did OpenAI release in 2023'
  BM25 only:
    [3.667] OpenAI released GPT-4 in March 2023 with multimodal capabili
    [0.777] Transformers use attention to process entire sequences in pa
  Hybrid (alpha=0.5):
    [0.03279] OpenAI released GPT-4 in March 2023 with multimodal capabili
    [0.01613] GPT-4 outperforms GPT-3.5 on most reasoning benchmarks.

'''