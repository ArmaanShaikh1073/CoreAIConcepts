# ivf_from_scratch.py
# Build intuition for how IVF works internally

import numpy as np
from sklearn.cluster import KMeans

class IVFIndex:
    """
    IVF: train k-means → assign vectors to cells → search only nearby cells.
    """
    def __init__(self, dim: int, n_lists: int = 100):
        self.dim     = dim
        self.n_lists = n_lists
        self.kmeans  = None
        # NUMERIC EXAMPLE: Creates a dictionary with 316 keys (0 to 315)
        # self.cells = {0: [], 1: [], ... 315: []}
        self.cells: dict[int, list] = {i: [] for i in range(n_lists)}

    def train(self, training_vectors: np.ndarray):
        """
        Train step: fit k-means to learn cell centroids.
        """
        # NUMERIC EXAMPLE: n_lists = 316. training_vectors shape = (12324, 128)
        print(f"Training k-means with {self.n_lists} clusters...")
        self.kmeans = KMeans(n_clusters=self.n_lists, n_init=10)
        self.kmeans.fit(training_vectors)
        # After this, self.kmeans.cluster_centers_ holds a matrix of shape (316, 128)
        # representing the center coordinates of all 316 buckets.
        print("Training complete.")

    def add(self, vectors: np.ndarray, ids: list[str]):
        """Assign each vector to its nearest centroid cell."""
        # NUMERIC EXAMPLE: vectors shape = (100000, 128)
        # cell_assignments becomes an array of 100,000 integers between 0 and 315
        # e.g., [5, 312, 14, 5, ...]
        cell_assignments = self.kmeans.predict(vectors)
        
        for idx, (vec, cell_id) in enumerate(zip(vectors, cell_assignments)):
            # Say vec = [2.0, 2.0, ... 128 dims], cell_id = 5, ids[idx] = "doc_0"
            # L2 norm calculation: length = sqrt(2^2 + 2^2 + ...) = say, 4.0
            # norm_vec becomes: [2/4, 2/4, ...] -> [0.5, 0.5, ...] (Length is now exactly 1.0)
            norm_vec = vec / (np.linalg.norm(vec) + 1e-10)
            
            # Appends ("doc_0", [0.5, 0.5, ...]) into self.cells[5]
            self.cells[cell_id].append((ids[idx], norm_vec))
            
        print(f"Added {len(vectors)} vectors across {self.n_lists} cells")
        avg_per_cell = len(vectors) / self.n_lists
        print(f"Average vectors per cell: {avg_per_cell:.0f}")

    def search(self, query: np.ndarray, top_k: int = 5, n_probe: int = 10):
        """
        1. Find the n_probe nearest centroids to the query
        2. Exhaustively search only those cells
        3. Return top_k results across all probed cells
        """
        # NUMERIC EXAMPLE: Incoming raw query vector is normalized to a length of 1.0
        q = query / (np.linalg.norm(query) + 1e-10)
        
        # Step 1: Matrix multiplication between cluster centers (316, 128) and query (128,)
        # Result 'centroid_scores' is an array of 316 similarity scores.
        centroid_scores = self.kmeans.cluster_centers_ @ q
        
        # Sorts the 316 scores descending and slices the top 'n_probe'.
        # If n_probe=5, nearest_cells might be array: [42, 115, 7, 201, 88]
        nearest_cells   = np.argsort(centroid_scores)[::-1][:n_probe]
        
        # Step 2: Exhaustive search inside only those 5 chosen buckets
        candidates = []
        for cell_id in nearest_cells:
            # Loops only through the ~316 documents stored inside cell_id 42, then 115, etc.
            # Total vectors checked = 5 cells * 316 vectors = 1,580 vectors (instead of 100,000!)
            for doc_id, vec in self.cells[cell_id]:
                # Dot product yields Cosine Similarity score directly because both are normalized
                score = float(vec @ q) # e.g., 0.2663
                candidates.append((doc_id, score))
        
        # Step 3: Sorts the 1,580 collected candidates by score descending
        candidates.sort(key=lambda x: -x[1])
        # Returns the top 5 absolute highest matching documents
        return candidates[:top_k]


# Build and test
N, D = 100_000, 128
n_lists = int(np.sqrt(N))  # 316 clusters

index = IVFIndex(dim=D, n_lists=n_lists)

# Training data (using the FAISS minimum multiplier of 39)
# 316 clusters * 39 vectors = 12,324 training vectors
training_data = np.random.randn(n_lists * 39, D).astype('float32')
index.train(training_data)

# Add all vectors
vectors = np.random.randn(N, D).astype('float32')
ids = [f"doc_{i}" for i in range(N)]
index.add(vectors, ids)

# Search with different n_probe values — see the recall/speed tradeoff
query = np.random.randn(D).astype('float32')
for n_probe in [1, 5, 10, 20, n_lists]:
    results = index.search(query, top_k=5, n_probe=n_probe)
    cells_searched_pct = n_probe / n_lists * 100
    print(f"n_probe={n_probe:4d} ({cells_searched_pct:5.1f}% of cells): "
          f"top result score = {results[0][1]:.4f}")

'''
IVF working explaination: 


'''


'''
___ Output ______________________________

Training k-means with 316 clusters...
Training complete.
Added 100000 vectors across 316 cells
Average vectors per cell: 316
n_probe=   1 (  0.3% of cells): top result score = 0.2663
n_probe=   5 (  1.6% of cells): top result score = 0.2663
n_probe=  10 (  3.2% of cells): top result score = 0.2663
n_probe=  20 (  6.3% of cells): top result score = 0.3119
n_probe= 316 (100.0% of cells): top result score = 0.3794

'''