# mini_project_duplicate_detector.py
# Finds semantic duplicates in a dataset using embeddings + clustering
# Real use case: deduplicate FAQ questions, support tickets, dataset entries

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

@dataclass
class DuplicateGroup:
    canonical: str           # representative entry
    duplicates: list[str]    # entries too similar to canonical
    avg_similarity: float

class SemanticDuplicateDetector:
    """
    Finds semantically duplicate texts even when phrased differently.
    Works where exact-match or fuzzy-string methods fail.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.85):
        self.model     = SentenceTransformer(model_name)
        self.threshold = threshold  # similarity above this = duplicate
    
    def find_duplicates(self, texts: list[str]) -> list[DuplicateGroup]:
        if not texts:
            return []
        
        # Encode all texts
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        # Compute pairwise similarity
        sim_matrix = cosine_similarity(embeddings)
        
        # Greedy clustering: assign each text to its nearest unassigned group
        n = len(texts)
        assigned = [False] * n
        groups   = []
        
        for i in range(n):
            if assigned[i]:
                continue
            
            # Start a new group with texts[i] as canonical
            group_idxs = [i]
            assigned[i] = True
            
            for j in range(i+1, n):
                if not assigned[j] and sim_matrix[i][j] >= self.threshold:
                    group_idxs.append(j)
                    assigned[j] = True
            
            if len(group_idxs) > 1:
                # Compute average similarity within group
                sims = [sim_matrix[i][j] for j in group_idxs[1:]]
                groups.append(DuplicateGroup(
                    canonical       = texts[i],
                    duplicates      = [texts[j] for j in group_idxs[1:]],
                    avg_similarity  = float(np.mean(sims)),
                ))
        
        return sorted(groups, key=lambda g: -g.avg_similarity)
    
    def similarity_report(self, texts: list[str]) -> dict:
        """Full similarity matrix report — useful for debugging."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        sim_matrix = cosine_similarity(embeddings)
        
        pairs = []
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                pairs.append({
                    "text_a": texts[i], 
                    "text_b": texts[j],
                    "similarity": float(sim_matrix[i][j])
                })
        
        return sorted(pairs, key=lambda p: -p["similarity"])


# ── Test with FAQ-style questions ─────────────────────────────────────────────
faq_questions = [
    # Group 1 — password reset
    "How do I reset my password?",
    "I forgot my password, how can I recover it?",
    "Can't log in, need to reset login credentials",
    "Password recovery steps",
    "how to change my password?",  # tricky: "change" vs "reset"
    
    # Group 2 — billing
    "How do I cancel my subscription?",
    "I want to stop my monthly payment",
    "Cancel membership and stop being charged",
    
    # Group 3 — unique
    "What is your refund policy?",
    "How long does shipping take?",
    "Do you support international orders?",
    
    # Tricky: similar words but different meaning
    "How do I change my email address?",  # different from password
]

detector = SemanticDuplicateDetector(threshold=0.82)
groups   = detector.find_duplicates(faq_questions)

print(f"Found {len(groups)} duplicate groups:\n")
for i, g in enumerate(groups):
    print(f"Group {i+1} (avg similarity: {g.avg_similarity:.3f})")
    print(f"  Canonical: '{g.canonical}'")
    for d in g.duplicates:
        print(f"  Duplicate: '{d}'")
    print()

# Show top similarity pairs for inspection
print("Top 5 most similar pairs:")
report = detector.similarity_report(faq_questions)
for p in report[:5]:
    print(f"  {p['similarity']:.3f}  '{p['text_a'][:40]}' ↔ '{p['text_b'][:40]}'")