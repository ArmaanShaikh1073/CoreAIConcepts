# qdrant_complete.py
# Full Qdrant integration — everything you need for production

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter, FieldCondition,
    MatchValue, Range, SearchRequest,
    SparseVector, SparseVectorParams, Modifier,
    CreateAliasOperation, CreateAlias,
)
from sentence_transformers import SentenceTransformer
import uuid

# ── 1. Connect ────────────────────────────────────────────────────────────────
# Local (Docker):  docker run -p 6333:6333 qdrant/qdrant
# In-memory (dev): QdrantClient(":memory:")
# Cloud:           QdrantClient(url="https://xyz.cloud.qdrant.io", api_key="...")

client = QdrantClient(":memory:")   # in-memory for this demo
model  = SentenceTransformer("all-MiniLM-L6-v2")
DIM    = 384

# ── 2. Create collection ──────────────────────────────────────────────────────
COLLECTION = "documents"

client.create_collection(
    collection_name = COLLECTION,
    vectors_config  = VectorParams(
        size     = DIM,
        distance = Distance.COSINE,  # or DOT, EUCLID
    ),
    # HNSW index config — tune for your recall/memory/speed target
    hnsw_config = {
        "m"               : 16,    # connections per node (16 = balanced)
        "ef_construct"    : 200,   # build quality (higher = better, slower build)
        "full_scan_threshold": 10000,  # switch to flat search below this size
    },
    # Optional: quantization for memory savings
    # quantization_config = ScalarQuantizationConfig(type=ScalarType.INT8)
)

print(f"Collection '{COLLECTION}' created.")

# ── 3. Prepare documents with metadata (payload) ──────────────────────────────
documents = [
    {"id": 1, "text": "Python is used for machine learning and data science.",
     "category": "tech", "year": 2024, "source": "blog"},
    {"id": 2, "text": "LangChain simplifies building LLM applications.",
     "category": "tech", "year": 2024, "source": "docs"},
    {"id": 3, "text": "Qdrant is a high-performance vector database.",
     "category": "tech", "year": 2023, "source": "docs"},
    {"id": 4, "text": "Pizza originated in Naples, Italy.",
     "category": "food", "year": 2022, "source": "wiki"},
    {"id": 5, "text": "Sushi is a traditional Japanese dish with rice.",
     "category": "food", "year": 2022, "source": "wiki"},
    {"id": 6, "text": "Cricket is the most popular sport in India.",
     "category": "sports", "year": 2023, "source": "blog"},
    {"id": 7, "text": "The RAG pipeline combines retrieval with generation.",
     "category": "tech", "year": 2024, "source": "paper"},
    {"id": 8, "text": "Vector embeddings encode semantic meaning as numbers.",
     "category": "tech", "year": 2024, "source": "blog"},
    {"id": 9, "text": "Biryani is a delicious Indian dish.",
     "category": "food", "year": 2025, "source": "blog"},
    {"id": 10, "text": "Football is the most popular sport worldwide.",
     "category": "sports", "year": 2022, "source": "wiki"}
]

# ── 4. Embed and upsert ───────────────────────────────────────────────────────
texts   = [d["text"] for d in documents]
vectors = model.encode(texts, normalize_embeddings=True).tolist()

points = [
    PointStruct(
        id      = doc["id"],          # must be int or UUID string
        vector  = vectors[i],
        payload = {                   # any JSON-serializable metadata
            "text"    : doc["text"],
            "category": doc["category"],
            "year"    : doc["year"],
            "source"  : doc["source"],
        }
    )
    for i, doc in enumerate(documents)
]

client.upsert(collection_name=COLLECTION, points=points)
print(f"Upserted {len(points)} documents.")

# ── 5. Basic vector search ────────────────────────────────────────────────────
def search(query: str, top_k: int = 3, **filters):
    q_vec = model.encode([query], normalize_embeddings=True)[0].tolist()

    # Old: client.search(collection_name=COLLECTION, query_vector=q_vec, ...)
    # New Unified Query API syntax:
    response = client.query_points(
        collection_name = COLLECTION,
        query    = q_vec,
        limit           = top_k,
        with_payload    = True,   # return metadata alongside vectors
        with_vectors    = False,  # don't return the full vector (saves bandwidth)
    )

    return response.points

prompt = "suggest me best food for today"
results = search(prompt)
print("\nBasic search results:")
print(f"Query: {prompt}")
for r in results:
    print(f"  [{r.score:.4f}] {r.payload['text']}")

# ── 6. Filtered search — the killer feature ───────────────────────────────────
# Filter FIRST, then search within filtered subset
# Much more efficient than post-filtering

def filtered_search(query: str, category: str = None,
                    year_min: int = None, top_k: int = 3):
    q_vec = model.encode([query], normalize_embeddings=True)[0].tolist()

    # Build filter conditions
    conditions = []
    if category:
        conditions.append(FieldCondition(
            key   = "category",
            match = MatchValue(value=category)
        ))
    if year_min:
        conditions.append(FieldCondition(
            key   = "year",
            range = Range(gte=year_min)    # gte=greater-than-or-equal
        ))

    filter_ = Filter(must=conditions) if conditions else None

    response = client.query_points(
        collection_name = COLLECTION,
        query    = q_vec,
        query_filter    = filter_,
        limit           = top_k,
        with_payload    = True,
    )
    return response.points

# Only search tech docs from 2024
results = filtered_search("AI tools", category="tech", year_min=2024)
print("\nFiltered search (tech, 2024+):")
for r in results:
    print(f"  [{r.score:.4f}] [{r.payload['year']}] {r.payload['text']}")

# ── 7. CRUD operations ────────────────────────────────────────────────────────

# UPDATE: re-upsert with same ID — Qdrant overwrites automatically
updated_doc = {"text": "Python is the dominant language for AI and ML.", "category": "tech", "year": 2024, "source": "blog"}
new_vec     = model.encode([updated_doc["text"]], normalize_embeddings=True)[0].tolist()
client.upsert(collection_name=COLLECTION, points=[
    PointStruct(id=1, vector=new_vec, payload=updated_doc)
])

# UPDATE payload only (without re-embedding — useful for metadata changes)
client.set_payload(
    collection_name = COLLECTION,
    payload         = {"source": "updated_blog"},
    points          = [1],
)

# DELETE
client.delete(
    collection_name = COLLECTION,
    points_selector = [4],   # delete pizza document
)

# GET a specific point
point = client.retrieve(
    collection_name = COLLECTION,
    ids             = [1],
    with_payload    = True,
)
print(f"\nRetrieved point 1: {point[0].payload['text']}")

# ── 8. Collection info and index stats ────────────────────────────────────────
# info = client.get_collection(COLLECTION)
# print(f"\nCollection stats:")
# print(f"  Vectors count: {info.vectors_count}")
# print(f"  Index status:  {info.status}")
# print(f"  Disk usage:    {info.disk_data_size} bytes")

# ── 8. Collection info and index stats ────────────────────────────────────────
info = client.get_collection(COLLECTION)
print(f"\nCollection stats:")

# Change 'info.vectors_count' to 'info.points_count'
print(f"  Points count:  {info.points_count}") 

# You can also fetch the total indexed vectors explicitly
print(f"  Indexed vectors: {info.indexed_vectors_count}")

print(f"  Index status:  {info.status}")


'''

_____ Output _________________________________________________________

Collection 'documents' created.
Upserted 10 documents.

Basic search results:
Query: suggest me best food for today
  [0.3145] Biryani is a delicious Indian dish.
  [0.2112] Sushi is a traditional Japanese dish with rice.
  [0.2087] Pizza originated in Naples, Italy.

Filtered search (tech, 2024+):
  [0.3557] [2024] Python is used for machine learning and data science.
  [0.2722] [2024] The RAG pipeline combines retrieval with generation.
  [0.2067] [2024] Vector embeddings encode semantic meaning as numbers.

Retrieved point 1: Python is the dominant language for AI and ML.

Collection stats:
  Points count:  9
  Indexed vectors: 0
  Index status:  green

'''