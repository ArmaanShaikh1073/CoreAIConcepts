# hnsw_intuition.py
# Conceptual HNSW — shows the structure without full implementation

import numpy as np
import heapq
from collections import defaultdict

class HNSWIndex:
    """
    Simplified HNSW to show the core ideas.
    Production HNSW: use hnswlib or faiss.
    
    Key parameters:
      M     : max connections per node per layer (typ. 16-64)
              Higher M = better recall, more memory, slower build
      ef_construction : beam width during build (typ. 100-500)
                        Higher = better quality graph, much slower build
      ef_search : beam width during query (typ. 50-200)
                  Higher = better recall, slower query
    """
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200):
        self.dim = dim
        self.M = M                # Maximum out-degree (links) per node on upper layers (e.g., 16)
        self.M0 = 2 * M           # Layer 0 can hold double the links (32) to handle the dense bottom layer
        self.ef_c = ef_construction
        self.ml = 1 / np.log(M)   # Normalization factor for exponential layer assignment decay
        self.vectors: list = []   # Raw storage of all added vectors: [vec0, vec1, ...]
        self.ids: list = []       # Raw storage of document string IDs: ["doc_0", "doc_1", ...]
        
        # Nested graph structure: self.graph[layer][node_index] = [neighbor_index1, neighbor_index2, ...]
        # NUMERIC EXAMPLE: self.graph[1][0] = [5, 12] means at Layer 1, Node 0 is linked to Nodes 5 and 12
        self.graph: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        self.entry_point: int = None  # Global entry point node index at the absolute top layer
        self.max_layer: int = 0       # Tracks highest layer currently active in the entire graph

    def _random_level(self) -> int:
        """Sample which layers this node exists in (exponential distribution)."""
        level = 0
        # Normalizes assignment so most nodes end up at Layer 0, and exponentially fewer reach higher layers
        while np.random.random() < self.ml and level < 16:
            level += 1
        return level # Returns an integer (e.g., 0 means it only exists on the base dense layer)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """cosine distance (assumes normalized)."""
        # Since vectors are L2-normalized to 1.0, dot product yields Cosine Similarity.
        # Distance = 1.0 - Similarity. (0.0 means identical, 2.0 means completely opposite)
        return 1.0 - np.dot(a, b)

    def _search_layer(self, query: np.ndarray, entry: int, ef: int, layer: int) -> list:
        """Beam search within one layer. Returns ef nearest candidates."""
        visited = {entry}  # Set to track node indices we have already calculated distances for
        
        # Min-Heap to track nodes to explore next. Format: (distance, node_index)
        # NUMERIC EXAMPLE: [(0.42, 5)] -> Node 5 is at distance 0.42 from the query
        candidates = [(self._distance(query, self.vectors[entry]), entry)]
        
        # Tracks the best results found on this layer so far
        results = [(self._distance(query, self.vectors[entry]), entry)]

        while candidates:
            # Pop the node closest to the query out of our exploration heap
            dist_c, c = heapq.heappop(candidates)
            
            # Optimization: If the closest candidate left to explore is further away 
            # than our worst tracked result, we cannot possibly find anything better. Stop.
            if results and dist_c > max(d for d, _ in results):
                break
                
            # Check all graph neighbors connected to current node 'c' on this specific layer
            for neighbor in self.graph[layer].get(c, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    d = self._distance(query, self.vectors[neighbor])
                    
                    # If we haven't filled our beam width 'ef', or this neighbor is closer than our worst result
                    if len(results) < ef or d < max(dist_ for dist_, _ in results):
                        heapq.heappush(candidates, (d, neighbor)) # Add to explore queue
                        results.append((d, neighbor))             # Add to candidate results
                        
                        # Prune results list to keep exactly the top 'ef' closest items
                        if len(results) > ef:
                            results.sort()
                            results = results[:ef]
                            
        return sorted(results) # Returns list of tuples sorted by distance ascending

    def add(self, vector: np.ndarray, doc_id: str):
        """Insert one vector — this is the expensive step that builds the graph."""
        # Force vector to a geometric length of 1.0 for accurate dot product calculation
        vector = vector / (np.linalg.norm(vector) + 1e-10)
        idx = len(self.vectors) # Assign an internal integer index to this node (e.g., node 0, node 1)
        self.vectors.append(vector)
        self.ids.append(doc_id)
        
        node_level = self._random_level() # Determine highest layer this node will live on (e.g., Layer 2)
        
        # Cold start: If this is the absolute first vector, set it as the global entry point and exit
        if self.entry_point is None:
            self.entry_point = 0
            self.max_layer = node_level
            return

        ep = self.entry_point # Start routing from the top global entry point
        
        # PHASE 1: Fast greedy descent from the maximum graph layer down to just above node's max layer
        # Uses ef=1 because we only care about finding a single entry point fast on upper levels
        for layer in range(self.max_layer, node_level, -1):
            results = self._search_layer(vector, ep, ef=1, layer=layer)
            ep = results[0][1] # Safely update entry point to the node closest to our vector on this layer

        # PHASE 2: Connect the new node to existing neighbors from its assigned top layer down to Layer 0
        for layer in range(min(node_level, self.max_layer), -1, -1):
            max_conn = self.M0 if layer == 0 else self.M # Cap links to 32 on Layer 0, or 16 on upper layers
            
            # Find closest existing graph nodes on this layer using build-time beam width (ef_construction)
            candidates = self._search_layer(vector, ep, self.ef_c, layer=layer)
            neighbors = [n for _, n in candidates[:max_conn]] # Pick the top 'M' closest nodes
            
            # Establish bidirectional links between the new node and its neighbors
            self.graph[layer][idx] = neighbors
            for neighbor in neighbors:
                self.graph[layer][neighbor].append(idx)
                
                # Heuristic Pruning: If a neighbor now has too many connections, force it to drop the furthest ones
                if len(self.graph[layer][neighbor]) > max_conn:
                    dists = [(self._distance(self.vectors[neighbor], self.vectors[n]), n) for n in self.graph[layer][neighbor]]
                    dists.sort()
                    self.graph[layer][neighbor] = [n for _, n in dists[:max_conn]] # Keep only the closest links
                    
            if candidates:
                ep = candidates[0][1] # Update our entry point to descend to the next layer down

        # If the new node randomly rolled a higher layer than the global maximum, update graph entry targets
        if node_level > self.max_layer:
            self.max_layer = node_level
            self.entry_point = idx

    def search(self, query: np.ndarray, top_k: int = 5, ef: int = 50):
        """
        Query: start at top layer, zoom in, beam search at layer 0.
        ef (ef_search): larger = better recall, more comparisons
        """
        # Normalize incoming query vector to length 1.0
        query = query / (np.linalg.norm(query) + 1e-10)
        ep = self.entry_point # Start routing at global top entry node
        
        # Fast routing pass: Greedily hop down through upper layers using a beam width of 1
        for layer in range(self.max_layer, 0, -1):
            results = self._search_layer(query, ep, ef=1, layer=layer)
            ep = results[0][1] # Set next layer's starting point to the closest node found
            
        # Dense baseline pass: Execute a high-precision Beam Search on Layer 0 using 'ef_search' (e.g., 50)
        results = self._search_layer(query, ep, ef=max(ef, top_k), layer=0)
        
        # Convert distances back to Cosine Similarity scores (Score = 1.0 - Distance)
        # Returns a mapped list of string doc IDs and their final floating-point similarity scores
        return [(self.ids[n], 1.0 - d) for d, n in results[:top_k]]

# Test it
dim = 128
index = HNSWIndex(dim=dim, M=16, ef_construction=200)
for i in range(1000):
    vec = np.random.randn(dim).astype('float32')
    index.add(vec, f"doc_{i}")      
query = np.random.randn(dim).astype('float32')
results = index.search(query, top_k=5, ef=50)
print("Top 5 results:")
for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")


'''
___ Output ______________________________

Top 5 results:
doc_434: 0.3043
doc_479: 0.2577
doc_26: 0.2431
doc_801: 0.2374
doc_368: 0.2312

'''