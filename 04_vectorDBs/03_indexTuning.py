# index_tuning.py
# How to choose M, ef_construction, ef_search, n_lists, n_probe

import hnswlib
import numpy as np
import time

def benchmark_hnsw(N=50_000, D=384, queries=100):
    """
    Run a sweep over HNSW parameters and measure recall vs latency.
    This is the exact experiment you run before choosing parameters for prod.
    """
    vectors = np.random.randn(N, D).astype('float32')
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # normalize
    q_vecs  = np.random.randn(queries, D).astype('float32')
    q_vecs  /= np.linalg.norm(q_vecs, axis=1, keepdims=True)

    # Ground truth: brute force
    gt_labels = []
    for q in q_vecs:
        scores = vectors @ q
        gt_labels.append(set(np.argsort(scores)[::-1][:10].tolist()))

    results = []
    for M in [8, 16, 32]:
        for ef_c in [100, 200]:
            # Build index
            index = hnswlib.Index(space='cosine', dim=D)
            index.init_index(max_elements=N, ef_construction=ef_c, M=M)
            t0 = time.time()
            index.add_items(vectors, np.arange(N))
            build_time = time.time() - t0

            for ef_s in [20, 50, 100, 200]:
                index.set_ef(ef_s)
                t0 = time.time()
                labels_batch, _ = index.knn_query(q_vecs, k=10)
                query_time = (time.time() - t0) / queries * 1000  # ms per query

                # Compute recall@10
                recall = np.mean([
                    len(set(labels_batch[i].tolist()) & gt_labels[i]) / 10
                    for i in range(queries)
                ])

                results.append({
                    "M": M, "ef_c": ef_c, "ef_s": ef_s,
                    "recall": recall,
                    "latency_ms": query_time,
                    "build_s": build_time,
                    "mem_MB": (N * M * 4 * 2) / 1e6,  # approx
                })

    return results

# ── Parameter guide (no need to run benchmark every time) ────────────────────
"""
HNSW Parameters:

M (connections per node) — the most impactful parameter
  M=8:  less memory, faster build, worse recall. Use for: 1M+ vectors, memory-constrained
  M=16: balanced — the standard default. Good for most cases.
  M=32: best recall, 2x memory, slower build. Use for: < 500k vectors, recall critical
  M=64: diminishing returns. Rarely worth it.
  
ef_construction (build-time beam width)
  ef=100: fast build, acceptable quality
  ef=200: standard. Recommended starting point.
  ef=500: very high quality graph, 3-5x slower build. Use when you build once, query forever.
  Rule: ef_construction >= 2*M always

ef_search (query-time beam width) — tune this WITHOUT rebuilding
  ef=20:  ~85% recall, fastest
  ef=50:  ~95% recall, good balance ← start here
  ef=100: ~98% recall
  ef=200: ~99.5% recall, 2-3x slower than ef=50
  
Memory formula: ~(M * 4 bytes * 2 * N) + (D * 4 * N)
  At M=16, D=384, N=1M: ~160MB + 1.5GB = ~1.66GB

IVF Parameters:

n_lists (number of Voronoi cells)
  Rule of thumb: sqrt(N) to 4*sqrt(N)
  N=100k  → n_lists=316 to 1264
  N=1M    → n_lists=1000 to 4000
  N=10M   → n_lists=3162 to 12649
  Too few → cells too large, n_probe must be high
  Too many → cells too small, build time explodes, poor centroids

n_probe (cells searched per query)
  n_probe/n_lists = search fraction
  0.01 → ~70% recall (1% of cells)
  0.05 → ~90% recall (5% of cells)
  0.10 → ~95% recall (10% of cells)
  0.50 → ~99% recall (50% of cells = approaching flat search)
  
Training size: 39 * n_lists examples minimum for stable centroids
"""

# ── Decision table ────────────────────────────────────────────────────────────
def recommend_params(n_vectors: int, recall_target: float,
                     latency_budget_ms: float, memory_gb: float):
    """Print recommended parameters for your use case."""

    if n_vectors <= 50_000:
        print(f"N={n_vectors:,}: Use FLAT search. ANN not needed.")
        return

    if memory_gb < 0.5 and n_vectors > 500_000:
        print("Tight memory: consider IVF + scalar quantization (INT8)")
        print(f"  n_lists = {int(n_vectors**0.5)}")
        print(f"  n_probe = {max(10, int(n_vectors**0.5 * 0.05))}")
        return

    M = 16 if recall_target < 0.97 else 32
    ef_c = 200 if recall_target < 0.99 else 400

    if latency_budget_ms >= 20:
        ef_s = 100
    elif latency_budget_ms >= 10:
        ef_s = 50
    else:
        ef_s = 20

    mem_est = (M * 4 * 2 * n_vectors + 384 * 4 * n_vectors) / 1e9

    print(f"\nRecommended HNSW config for N={n_vectors:,}, "
          f"recall≥{recall_target:.0%}, latency≤{latency_budget_ms}ms:")
    print(f"  M                = {M}")
    print(f"  ef_construction  = {ef_c}")
    print(f"  ef_search        = {ef_s}  (tune this at runtime)")
    print(f"  Estimated memory = {mem_est:.2f} GB")

recommend_params(500_00, recall_target=0.95, latency_budget_ms=15, memory_gb=4)
recommend_params(500_000, recall_target=0.95, latency_budget_ms=15, memory_gb=4)
recommend_params(2_000_000, recall_target=0.99, latency_budget_ms=20, memory_gb=16)



'''
____ Output _________________________________________________________

N=50,000: Use FLAT search. ANN not needed.

Recommended HNSW config for N=500,000, recall≥95%, latency≤15ms:
  M                = 16
  ef_construction  = 200
  ef_search        = 50  (tune this at runtime)
  Estimated memory = 0.83 GB

Recommended HNSW config for N=2,000,000, recall≥99%, latency≤20ms:
  M                = 32
  ef_construction  = 400
  ef_search        = 100  (tune this at runtime)
  Estimated memory = 3.58 GB
'''