import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── 2. Real-time semantic search (what RAG does) ─────────────────────────────
class SimpleSemanticSearch:
    """Minimal semantic search — this is the core of RAG retrieval."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
    
    def index(self, documents: list[str]):
        """Encode and store all documents."""
        self.documents = documents
        # IMPORTANT: normalize=True makes cosine similarity = dot product
        # dot product is much faster, especially with vector databases
        self.embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,  # L2 normalize each vector
            show_progress_bar=True,
        )
        print(f"Indexed {len(documents)} documents, shape: {self.embeddings.shape}")
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search for documents most similar to the query."""
        query_emb = self.model.encode([query], normalize_embeddings=True)
        
        # With normalized vectors: cosine_sim = dot product
        scores = (query_emb @ self.embeddings.T)[0]
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [
            {"rank": i+1, "score": float(scores[idx]), "doc": self.documents[idx]}
            for i, idx in enumerate(top_indices)
        ]


# Test it
knowledge_base = [
    "Python is a high-level programming language known for its simplicity.",
    "Neural networks are inspired by the human brain's structure.",
    "Transformers use attention mechanisms to process sequences.",
    "RAG stands for Retrieval-Augmented Generation.",
    "Embeddings convert words into numerical vectors.",
    "Gradient descent is an optimization algorithm for training models.",
    "LangChain is a framework for building LLM applications.",
    "Vector databases store and search high-dimensional embeddings efficiently.",
]

searcher = SimpleSemanticSearch()
searcher.index(knowledge_base)

queries = [
    "What is a vector database?",
    "How do language models work?",
    "Show me something about Python",
    "What does RAG mean?",
    "Tell me about recursion in java",
]

for q in queries:
    print(f"\nQuery: '{q}'")
    results = searcher.search(q, top_k=2)
    for r in results:
        print(f"  [{r['rank']}] {r['score']:.3f}  {r['doc'][:60]}")


'''

___ Output ______________________________

Indexed 8 documents, shape: (8, 384)

Query: 'What is a vector database?'
  [1] 0.633  Vector databases store and search high-dimensional embedding
  [2] 0.385  Embeddings convert words into numerical vectors.

Query: 'How do language models work?'
  [1] 0.349  Embeddings convert words into numerical vectors.
  [2] 0.326  LangChain is a framework for building LLM applications.

Query: 'Show me something about Python'
  [1] 0.784  Python is a high-level programming language known for its si
  [2] 0.185  Embeddings convert words into numerical vectors.

Query: 'What does RAG mean?'
  [1] 0.767  RAG stands for Retrieval-Augmented Generation.
  [2] 0.094  LangChain is a framework for building LLM applications.

Query: 'Tell me about recursion in java'
  [1] 0.241  Transformers use attention mechanisms to process sequences.
  [2] 0.229  Python is a high-level programming language known for its si

'''