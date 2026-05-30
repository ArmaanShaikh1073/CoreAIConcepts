# production_embeddings.py
# This is the code you'll actually write at work

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── 1. Sentence embeddings (most useful in practice) ─────────────────────────
# These encode whole sentences into single vectors, not just words
model = SentenceTransformer('all-mpnet-base-v2')   # 384-dim, fast, good quality
# Other options:
#   'all-mpnet-base-v2'       → 768-dim, higher quality, slower
#   'text-embedding-3-small'  → OpenAI API, 1536-dim
#   'nomic-embed-text-v1.5'   → 768-dim, open-source, very strong

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "What is the capital of France?",
    "Paris is the largest city in France",
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
]

# Each sentence → 384-dimensional vector
embeddings = model.encode(sentences, show_progress_bar=False)
print(f"Embedding shape: {embeddings.shape}")  # (6, 384)

# Compute all-pairs similarity
sim_matrix = cosine_similarity(embeddings)

print("\nSemantic similarity matrix (threshold > 0.5):")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = sim_matrix[i][j]
        if sim > 0.4:
            print(f"  {sim:.3f}  '{sentences[i][:35]}' ↔ '{sentences[j][:35]}'")

'''
___ Output 1 ______________________________
# Using 'all-MiniLM-L6-v2' --> 384-dim, fast, good quality

Embedding shape: (6, 384)

Semantic similarity matrix (threshold > 0.5):
  0.678  'How do I reset my password?' ↔ 'I forgot my login credentials'
  0.660  'What is the capital of France?' ↔ 'Paris is the largest city in France'
  0.425  'Machine learning is a subset of AI' ↔ 'Deep learning uses neural networks'
'''


'''
___ Output 2 ______________________________
# Using 'all-mpnet-base-v2' (768-dim) → 768-dim, higher quality, slower

Embedding shape: (6, 768)

Semantic similarity matrix (threshold > 0.5):
  0.627  'How do I reset my password?' ↔ 'I forgot my login credentials'
  0.663  'What is the capital of France?' ↔ 'Paris is the largest city in France'
  0.530  'Machine learning is a subset of AI' ↔ 'Deep learning uses neural networks'
'''