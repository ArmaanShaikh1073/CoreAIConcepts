# hnsw_production.py
import hnswlib
import numpy as np

# 1. DEFINE MATRIX DIMENSIONS
N = 10_000   # Total number of documents to index
D = 384      # Dimensionality of vectors (e.g., MiniLM embedding output size)

# 2. INITIALIZE THE INDEX
# Space can be 'cosine', 'l2', or 'ip' (Inner Product/Dot Product)
index = hnswlib.Index(space='cosine', dim=D)

# 3. CONFIGURE GRAPH ARCHITECTURE (The critical step)
# This allocates memory and locks in the build parameters.
index.init_index(
    max_elements=N,         # CRITICAL: You must declare max capacity upfront. Hard to resize later.
    M=16,                   # Max bidirectional links per node. 16 is the industry sweet spot.
    ef_construction=200     # Beam width size used ONLY during build time. Higher = higher accuracy graph.
)

# 4. INGEST DATA GENERATION (Numeric Example Simulation)
# Generating 10,000 random vectors as mock text embeddings
vectors = np.random.randn(N, D).astype('float32')

# HNSWLib requires integer IDs to track graph nodes (unlike strings like 'doc_1')
# You must map these integers back to your database primary keys separately.
labels = np.arange(N) # [0, 1, 2, ... 9999]

# 5. BUILD THE GRAPH (The expensive O(N log N) operation)
# This drops vectors into layers and builds the entry-point connections.
index.add_items(vectors, labels)
print(f"Index built completely with {index.element_count} elements.")

# 6. TUNE RUNTIME RUN-TIME PERFORMANCE
# You can adjust 'ef_search' at runtime dynamically without rebuilding the graph!
index.set_ef(50) # Controls query beam width. 50 = balanced; 100-200 = near 100% recall.

# 7. SEARCH THE GRAPH
query = np.random.randn(D).astype('float32') # Single incoming user query vector

# Query returns two arrays: nearest integer IDs and their geometric distances
# k=5 means we want the top 5 closest documents
retrieved_ids, distances = index.knn_query(query, k=5)

# 8. OUTPUT PRODUCTION INTERPRETATION
print("\nTop 5 Results:")
for node_id, dist in zip(retrieved_ids[0], distances[0]):
    # Note: In 'cosine' space, HNSWLib calculates Cosine Distance (1.0 - Cosine Similarity).
    # To present a similarity score to a user: Score = 1.0 - Distance
    similarity_score = 1.0 - dist
    print(f"Node ID: {node_id} | Cosine Similarity Score: {similarity_score:.4f}")

# 9. SERIALIZATION (Save & Load for production serving)
index.save_index("hnsw_production_matrix.bin")

# To load on a serving instance or lambda function later:
new_index = hnswlib.Index(space='cosine', dim=D)
new_index.load_index("hnsw_production_matrix.bin", max_elements=N)


'''
Direct Punchlines for your Interviewer
If an interviewer asks: "How do you implement and tune HNSW in production using native libraries?"

Your Answer:

In production, we use hnswlib and instantiate it by setting three core hyper-parameters: max_elements, M, and ef_construction.

The major structural constraint is that max_elements must be declared upfront because the memory buffer for the graph allocations is fixed.

For tuning, we balance the performance profile using ef_search. This parameter can be modified on the fly at runtime without recreating the index. 
If our query metrics show a drop in accuracy (Recall), we crank up ef_search to expand our beam search window on Layer 0. 
If our application hits a strict SLA latency ceiling, we dial ef_search down to drop the overall CPU execution cycles per query.

'''


'''
___ OUTPUT_____________________________________

Index built completely with 10000 elements.

Top 5 Results:
Node ID: 2051 | Cosine Similarity Score: 0.1807
Node ID: 9949 | Cosine Similarity Score: 0.1729
Node ID: 6547 | Cosine Similarity Score: 0.1595
Node ID: 1812 | Cosine Similarity Score: 0.1573
Node ID: 949 | Cosine Similarity Score: 0.1553
'''