# similarity_metrics.py
import numpy as np

a = np.array([0.2, 0.8, 0.1, 0.9])
b = np.array([0.3, 0.7, 0.2, 0.8])
c = np.array([0.9, 0.1, 0.8, 0.2])  # opposite direction

# ── 1. Cosine similarity — the one you'll use 95% of the time ────────────────
# Measures angle between vectors, ignores magnitude
# Range: [-1, 1],  1=identical direction, 0=orthogonal, -1=opposite
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"cosine(a,b) = {cosine_sim(a,b):.4f}")  # high, similar direction
print(f"cosine(a,c) = {cosine_sim(a,c):.4f}")  # low, different direction

# Why cosine and not dot product?
# Dot product is affected by vector magnitude — a 10× longer vector of the
# same direction would score 10× higher. Cosine normalizes this out.
# Exception: if you L2-normalize your embeddings first,
# cosine_sim = dot_product (and dot product is faster!)

# ── 2. Euclidean distance — distance in embedding space ──────────────────────
# Range: [0, ∞),  0 = identical
def euclidean_dist(a, b):
    return np.linalg.norm(a - b)

print(f"\neuclidean(a,b) = {euclidean_dist(a,b):.4f}")  # small
print(f"euclidean(a,c) = {euclidean_dist(a,c):.4f}")  # large

# Use Euclidean when: you've done dimensionality reduction (UMAP, PCA)
# Avoid when: vectors have different magnitudes (long docs vs short ones)

# ── 3. Dot product — fast, used inside transformers ──────────────────────────
# = cosine_sim × magnitude_a × magnitude_b
# Range: (-∞, ∞)
print(f"\ndot(a,b) = {np.dot(a,b):.4f}")
print(f"dot(a,c) = {np.dot(a,c):.4f}")

# Used in: attention mechanism (Q·K), softmax scaling, vector DB HNSW indexes

# ── Decision guide ────────────────────────────────────────────────────────────
"""
Use cosine similarity when:
  - Comparing document embeddings (default choice)
  - Semantic search and RAG retrieval  
  - Clustering by topic
  - The vectors may have varying magnitudes

Use dot product when:
  - Vectors are L2-normalized (it equals cosine but faster)
  - Inside transformers (attention scores)
  - Your vector DB uses inner product index (Faiss, Pinecone)

Use Euclidean when:
  - Working in reduced 2D/3D space for visualization
  - k-NN classification after dimensionality reduction
  - The absolute position in space matters, not just direction
"""


'''
___ Output ______________________________

cosine(a,b) = 0.9893
cosine(a,c) = 0.3467

euclidean(a,b) = 0.2000
euclidean(a,c) = 1.4000

dot(a,b) = 1.3600
dot(a,c) = 0.5200

'''