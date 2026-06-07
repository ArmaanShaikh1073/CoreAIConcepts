# mini_project_semantic_search_system.py
# A production-grade semantic search system in ~150 lines
# Covers: embed → chunk → index → hybrid search → ranked results

from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
import numpy as np
import uuid
import tiktoken

@dataclass
class Document:
    text: str
    source: str
    category: str = "general"
    doc_id: str   = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class SearchResult:
    text:     str
    source:   str
    category: str
    score:    float
    rank:     int

class SemanticSearchSystem:
    """
    Complete semantic search system.
    Usage:
      sys = SemanticSearchSystem()
      sys.ingest([Document(...), ...])
      results = sys.search("my query", category="tech")
    """

    COLLECTION = "knowledge_base"
    CHUNK_SIZE  = 256   # tokens
    CHUNK_OVERLAP = 32

    def __init__(self, embed_model: str = "all-MiniLM-L6-v2"):
        self.model   = SentenceTransformer(embed_model)
        self.dim     = self.model.get_sentence_embedding_dimension()
        self.enc     = tiktoken.get_encoding("cl100k_base")
        self.client  = QdrantClient(":memory:")
        self.chunks  = []    # store text for BM25
        self.bm25    = None
        self._init_collection()

    def _init_collection(self):
        self.client.create_collection(
            collection_name = self.COLLECTION,
            vectors_config  = VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    def _chunk(self, text: str, source: str, category: str) -> list[dict]:
        """Token-aware chunking with overlap."""
        tokens = self.enc.encode(text)
        chunks = []
        start  = 0
        while start < len(tokens):
            end        = min(start + self.CHUNK_SIZE, len(tokens))
            chunk_text = self.enc.decode(tokens[start:end])
            chunks.append({"text": chunk_text, "source": source, "category": category})
            start = end - self.CHUNK_OVERLAP
            if end == len(tokens):
                break
        return chunks

    def ingest(self, documents: list[Document]):
        """Chunk, embed, and index all documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self._chunk(doc.text, doc.source, doc.category))

        texts   = [c["text"] for c in all_chunks]
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

        # Upsert to Qdrant
        points = [
            PointStruct(
                id      = i,
                vector  = vectors[i].tolist(),
                payload = all_chunks[i],
            )
            for i in range(len(all_chunks))
        ]
        self.client.upsert(collection_name=self.COLLECTION, points=points)

        # Build BM25 index
        self.chunks = all_chunks
        self.bm25   = BM25Okapi([t.lower().split() for t in texts])
        print(f"Ingested {len(documents)} docs → {len(all_chunks)} chunks")

    def search(self, query: str, top_k: int = 5,
               category: str = None, mode: str = "hybrid") -> list[SearchResult]:
        """
        mode: "vector" | "bm25" | "hybrid"
        """
        q_vec   = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        filter_ = None
        if category:
            filter_ = Filter(must=[FieldCondition(
                key="category", match=MatchValue(value=category)
            )])


        # # OLD SYNTAX (client.search) — replaced by new Unified Query API
        # # Vector search
        # vec_results = self.client.search(
        #     collection_name = self.COLLECTION,
        #     query_vector    = q_vec,
        #     query_filter    = filter_,
        #     limit           = top_k * 3,
        #     with_payload    = True,
        # )
        # vec_ranked = {r.id: rank for rank, r in enumerate(vec_results)}


        response = self.client.query_points(
            collection_name=self.COLLECTION,
            query=q_vec,
            query_filter=filter_,
            limit=top_k * 3,
            with_payload=True,
        )

        vec_results = response.points

        vec_ranked = {
            point.id: rank
            for rank, point in enumerate(vec_results)
        }
        

        # BM25 search
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranked_ids = np.argsort(bm25_scores)[::-1][:top_k * 3].tolist()
        bm25_ranked = {doc_id: rank for rank, doc_id in enumerate(bm25_ranked_ids)}

        # RRF fusion
        all_ids = set(vec_ranked) | set(bm25_ranked)
        K = 60
        rrf_scores = {
            doc_id: (1/(K + vec_ranked.get(doc_id, 9999)) +
                     1/(K + bm25_ranked.get(doc_id, 9999)))
            for doc_id in all_ids
        }

        top_ids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])[:top_k]

        return [
            SearchResult(
                text     = self.chunks[doc_id]["text"],
                source   = self.chunks[doc_id]["source"],
                category = self.chunks[doc_id]["category"],
                score    = round(rrf_scores[doc_id], 5),
                rank     = i + 1,
            )
            for i, doc_id in enumerate(top_ids)
        ]


# ── Run it ────────────────────────────────────────────────────────────────────
docs = [
    Document(
        text="Retrieval-Augmented Generation (RAG) is a technique that combines "
             "a retrieval system with a generative model. Instead of relying purely "
             "on the model's internal knowledge, RAG first retrieves relevant documents "
             "from an external corpus, then uses them as context for generation. "
             "This dramatically reduces hallucination and allows the model to cite sources.",
        source="rag_paper.pdf",
        category="tech"
    ),
    Document(
        text="Vector databases store embeddings and enable approximate nearest neighbor "
             "search. Popular options include Qdrant, Pinecone, Weaviate, and Chroma. "
             "They differ in hosting options, performance, and feature sets. Qdrant is "
             "open-source and written in Rust for maximum performance.",
        source="vector_db_guide.md",
        category="tech"
    ),
    Document(
        text="Fine-tuning a language model adapts it to a specific domain or task. "
             "Techniques include LoRA, QLoRA, and full fine-tuning. LoRA adds small "
             "trainable matrices to frozen model layers, requiring much less GPU memory "
             "than full fine-tuning while achieving comparable results.",
        source="finetune_guide.md",
        category="tech"
    ),
    Document(
        text="Pizza originated in Naples, Italy. It typically consists of a flat round "
             "base of dough baked with a topping of tomato sauce and cheese, often "
             "with added meat or vegetables. Today, pizza is popular worldwide with "
             "countless regional variations.",
        source="pizza_wiki.html",
        category="food"
    ),
    Document(
        text="Sushi is a traditional Japanese dish with vinegared rice, usually "
             "accompanied by raw fish or seafood. It has become popular globally, "
             "with many variations like nigiri, maki, and sashimi. Sushi is often "
             "served with soy sauce, wasabi, and pickled ginger.",
        source="sushi_wiki.html",
        category="food"
    ),
    Document(
        text="Cricket is the most popular sport in India. It is a bat-and-ball game "
             "played between two teams of eleven players. The game has a rich history "
             "in India, with legendary players like Sachin Tendulkar and Virat Kohli. "
             "Cricket matches can last from a few hours to several days.",
        source="cricket_blog.html",
        category="sports"
    ),
    Document(
        text="The RAG pipeline combines retrieval with generation. The retriever "
             "fetches relevant documents based on the query, and the generator "
             "produces a response using both the query and retrieved context. This "
             "approach allows for more accurate and grounded responses, especially "
             "for knowledge-intensive tasks.",
        source="rag_blog.html",
        category="tech"
     ),
    Document(
        text="Vector embeddings encode semantic meaning as numbers. They allow us to "
             "represent text, images, and other data in a way that captures their "
             "meaning and relationships. Embeddings are the foundation of semantic "
             "search, recommendation systems, and many other AI applications.",
        source="embeddings_blog.html",
        category="tech"
     ),
     Document(
        text="Biryani is a delicious Indian dish. It consists of spiced rice cooked with "
             "meat, vegetables, or both. Biryani has many regional variations across India, "
             "each with its own unique blend of spices and cooking techniques.",
        source="biryani_blog.html",
        category="food"
     ),
        Document(
            text="Football is the most popular sport worldwide. It is played between two teams "
                "of eleven players with a spherical ball. The game is known as soccer in the "
                "United States and Canada. Football has a massive global following, with major "
                "events like the FIFA World Cup drawing billions of viewers.",
            source="football_wiki.html",
            category="sports"
        ),
        Document(
            text="The RAG pipeline combines retrieval with generation. The retriever fetches "
                "relevant documents based on the query, and the generator produces a response "
                "using both the query and retrieved context. This approach allows for more "
                "accurate and grounded responses, especially for knowledge-intensive tasks.",
            source="rag_blog.html",
            category="tech"
         ),
]

system = SemanticSearchSystem()
system.ingest(docs)

queries = [
    ("What is RAG and how does it work?",    None),
    ("Which vector database should I use?",  "tech"),
    ("How do I fine-tune with less GPU?",    "tech"),
    ("suggest me best food to eat today",    "food"),
    ("What is the most popular sport?",       None),
]

for query, cat in queries:
    print(f"\nQuery: '{query}'" + (f" [filter: {cat}]" if cat else ""))
    for r in system.search(query, top_k=2, category=cat, mode="hybrid"):
        print(f"  [{r.rank}] ({r.score:.5f}) {r.text[:80]}...")



'''
____ Output ______________________________________________________________________

Ingested 11 docs → 11 chunks

Query: 'What is RAG and how does it work?'
  [1] (0.03279) The RAG pipeline combines retrieval with generation. The retriever fetches relev...
  [2] (0.03254) Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval sy...

Query: 'Which vector database should I use?' [filter: tech]
  [1] (0.03333) Vector databases store embeddings and enable approximate nearest neighbor search...
  [2] (0.03279) Vector embeddings encode semantic meaning as numbers. They allow us to represent...

Query: 'How do I fine-tune with less GPU?' [filter: tech]
  [1] (0.03333) Fine-tuning a language model adapts it to a specific domain or task. Techniques ...
  [2] (0.03126) The RAG pipeline combines retrieval with generation. The retriever fetches relev...

Query: 'suggest me best food to eat today' [filter: food]
  [1] (0.03229) Biryani is a delicious Indian dish. It consists of spiced rice cooked with meat,...
  [2] (0.01677) Fine-tuning a language model adapts it to a specific domain or task. Techniques ...

Query: 'What is the most popular sport?'
  [1] (0.03333) Football is the most popular sport worldwide. It is played between two teams of ...
  [2] (0.03279) Cricket is the most popular sport in India. It is a bat-and-ball game played bet...

'''