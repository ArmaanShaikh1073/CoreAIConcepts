# flat_search.py
import numpy as np
from typing import List, Tuple

class FlatIndex:
    """
    Exact nearest neighbor — O(N·D) per query.
    Use when: N < 100k, highest recall required, index rarely queried.
    Never use when: N > 500k, latency < 50ms required.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: np.ndarray = None   # Stores database matrix, e.g., shape (10000, 384)
        self.ids: List[str] = []          # Stores matching IDs, e.g., ["doc_0", "doc_1", ...]

    def add(self, vectors: np.ndarray, ids: List[str]):
        """Add vectors to index. Normalize for cosine similarity."""
        # NUMERIC EXAMPLE: Say vectors has 2 documents, 3 dimensions: [[3.0, 4.0, 0.0], [0.0, 1.0, 0.0]]
        
        # Calculates geometric length of each row. 
        # Example: Row 1 length = sqrt(3^2 + 4^2 + 0^2) = 5.0. Row 2 length = 1.0.
        # norms shape becomes: [[5.0], [1.0]]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Divides every coordinate by its row length to force total length to 1.0.
        # Example: Row 1 becomes [3/5, 4/5, 0/5] -> [0.6, 0.8, 0.0] (Length is now 1.0)
        # 1e-10 is a safety buffer added to prevent crashing if a vector is all zeros (0 / 0).
        self.vectors = vectors / (norms + 1e-10)
        self.ids = ids

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Dot product = cosine similarity on L2-normalized vectors."""
        # NUMERIC EXAMPLE: Incoming raw query vector is [0.0, 3.0, 4.0]
        
        # Normalizes the single query vector to a length of 1.0.
        # Example: Length = sqrt(0^2 + 3^2 + 4^2) = 5.0. Normalized query 'q' = [0.0, 0.6, 0.8]
        q = query / (np.linalg.norm(query) + 1e-10)
        
        # Matrix Multiplication: Multiplies the normalized database matrix by the query vector.
        # Example: [0.6, 0.8, 0.0] (Doc 0) DOT [0.0, 0.6, 0.8] (Query) 
        # Score Doc 0 = (0.6*0.0) + (0.8*0.6) + (0.0*0.8) = 0.48
        # This gives an array of scores for all N documents simultaneously.
        scores = self.vectors @ q
        
        # np.argsort returns the positions that would sort the array from LOWEST to HIGHEST.
        # Example: If scores are [0.48, 0.91, 0.12], argsort gives indices: [2, 0, 1]
        # [::-1] reverses it to HIGHEST to LOWEST: [1, 0, 2]
        # [:top_k] slices the top requested items. If top_k=2, we get indices: [1, 0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Loops through the top indices to map them back to their original text IDs and scores.
        # Example: Returns [("doc_1", 0.91), ("doc_0", 0.48)]
        return [(self.ids[i], float(scores[i])) for i in top_indices]

# Usage
dim = 384
index = FlatIndex(dim)
vectors = np.random.randn(10_000, dim).astype('float32')
ids = [f"doc_{i}" for i in range(10_000)]
index.add(vectors, ids)

query = np.random.randn(dim).astype('float32')
results = index.search(query, top_k=5)

print("Top 5 results:")
for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")
# Works great at 10k. At 10M? ~3 seconds per query. Unusable.

'''
add function explained in detail:

The Goal: Ingestion & NormalizationThe main job of this function is to save your raw document vectors and their corresponding string IDs into the class instance (self). However, it performs a crucial mathematical optimization first: L2 Normalization.

Step-by-Step Execution:np.linalg.norm(..., axis=1, keepdims=True): 
This calculates the geometric length (magnitude) of each vector in the dataset. If your vector matrix has a shape of (10000, 384), norms will be a column vector of shape (10000, 1).

vectors / (norms + 1e-10): It divides every single number in a vector by its total length.Why + 1e-10? This is a tiny number added to prevent a "Division by Zero" crash in case you accidentally pass a vector that is completely empty/all zeros.

Why normalize? Normalizing forces every vector to have a geometric length of exactly 1.0. When two vectors have a length of 1.0, the Dot Product between them becomes mathematically identical to Cosine Similarity.

System Design Takeaway: By doing this expensive normalization work upfront inside the add function, we save massive amounts of CPU cycles later during the search function.

'''


'''
search function explained in detail:

The Goal: Brute-Force MatchingWhen a user asks a question, it gets converted into a single query vector. This function compares that single vector against your entire database matrix simultaneously.

Step-by-Step Execution:q = query / ...: Just like we did in the add function, we normalize the incoming query vector so its length is also exactly 1.0.

scores = self.vectors @ q: This single line is where the heavy lifting happens. The @ operator represents Matrix Multiplication.self.vectors has a shape of (N, D) (where N is the number of documents, and D is the dimensions). q has a shape of (D, 1).When you multiply them, it performs a dot product between your query and every single document vector instantly. The result scores is an array of shape (N,) containing a similarity score between -1.0 and 1.0 for every document in your database.

np.argsort(scores)[::-1][:top_k]:np.argsort(scores) sorts the scores from lowest to highest and returns their original index positions.[::-1] reverses that array, so it goes from highest to lowest (highest similarity first).[:top_k] slices the top items (e.g., if top_k=5, it grabs the 5 most similar document indices).

The List Comprehension: It loops through those top 5 indices, grabs the corresponding string ID from self.ids, grabs the float score, and returns them to you as a clean list of tuples.
'''


'''
___ Output ______________________________

Top 5 results:
doc_4239: 0.2056
doc_8138: 0.1812
doc_7230: 0.1780
doc_8919: 0.1672
doc_3834: 0.1634
'''