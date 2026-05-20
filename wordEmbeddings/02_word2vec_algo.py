# word2vec_from_scratch.py
# Skip-gram Word2Vec in pure NumPy — no ML framework
# This teaches you exactly what's happening inside

import numpy as np
from collections import defaultdict

class Word2VecSkipGram:
    """
    Minimal Word2Vec skip-gram implementation.
    Real Word2Vec uses negative sampling and subsampling for speed,
    but this vanilla version shows the core idea clearly.
    Learns dense vector representations by forcing a center word to predict 
    its surrounding neighbor (context) words.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, learning_rate: float = 0.01):
        self.vocab_size  = vocab_size   # V: e.g., 20 words
        self.embed_dim   = embed_dim    # D: e.g., 5 dimensions
        self.lr          = learning_rate
        
        # Two weight matrices — this is the key architectural detail
        # W_in:  embedding for words when they are CENTER words  (vocab_size × embed_dim)
        # W_out: embedding for words when they are CONTEXT words (embed_dim × vocab_size)
        # The final word embedding = average of W_in and W_out rows

        # --- MATRIX 1: W_in (Shape: V × D | Example: 5 × 3) --------------------------------------
        # Stores the dense vectors for words when they act as the CENTER word.
        # Conceptually, each row is a word's vector.
        # Example W_in:
        #           Dim1    Dim2    Dim3
        # 'the'    [ 0.01, -0.02,  0.03 ] -> Row 0
        # 'king'   [ 0.15,  0.89, -0.44 ] -> Row 1  <-- Target Center Word
        # 'rules'  [ -0.05, 0.12,  0.01 ] -> Row 2
        # 'queen'  [ 0.12,  0.82, -0.41 ] -> Row 3
        # 'palace' [ 0.02,  0.34,  0.77 ] -> Row 4

        # --- MATRIX 2: W_out (Shape: D × V | Example: 3 × 5) -----------------------------
        # Stores vectors for words when they act as ambient CONTEXT (neighbor) words.
        # Conceptually, each COLUMN is a context word's representation.
        # Example W_out:
        #          'the'   'king'  'rules' 'queen' 'palace'
        # Dim1   [ 0.02,   0.11,   -0.09,   0.10,   0.05  ]
        # Dim2   [ -0.01,  0.78,    0.23,   0.71,   0.45  ]
        # Dim3   [ 0.04,  -0.30,    0.05,  -0.29,   0.81  ]
        #           Col 0   Col 1    Col 2   Col 3   Col 4
        #                             ^
        #                             |-- Suppose 'rules' is our true Context Word

        self.W_in  = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_out = np.random.randn(embed_dim, vocab_size) * 0.01
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Converts raw scores into a clean probability distribution.
        Input 'x' Shape: (V,) e.g., (5,) raw similarity scores.
        Output Shape: (V,) e.g., (5,) probabilities summing to 1.0.
        """
        # Numerical Stability step: If x = [1000, 1001, 999], np.exp(1000) crashes as 'inf'.
        # Subtracting max turns x into [ -1, 0, -2 ]. np.exp([ -1, 0, -2 ]) is perfectly safe!

        e = np.exp(x - x.max())   # subtract max for numerical stability
        return e / e.sum()
    
    def forward(self, center_idx: int):
        """
        Forward pass for one (center, context) pair.
        Returns hidden layer and output probabilities.

        Calculates what context words the network predicts for a given center word.
        EXAMPLE WALKTHROUGH:        Let center_idx = 1 (the word 'king').
        """
        # Step 1: Extract the center word's dense vector from W_in.
        # Look up row 1 of W_in.
        # h Shape: (D,) -> (3,)
        # Example Vector 'h': [ 0.15,  0.89, -0.44 ]
        h = self.W_in[center_idx]          # shape: (embed_dim,)
        
        # Step 2: compute scores for all vocab words
        # Compute raw dot-product similarity scores against ALL context words.
        # Math: W_out.T (Shape: 5×3) Matrix-Multiply by h (Shape: 3,)
        # Vector matrix multiplication yields raw scores for every word in the vocabulary.
        # scores Shape: (V,) -> (5,)
        # Example Result:
        #   Index 0 ('the'):    Col 0 of W_out ⋅ h = (0.02*0.15) + (-0.01*0.89) + (0.04*-0.44)  = -0.023
        #   Index 1 ('king'):   Col 1 of W_out ⋅ h = (0.11*0.15) + (0.78*0.89)  + (-0.30*-0.44) =  0.842
        #   Index 2 ('rules'):  Col 2 of W_out ⋅ h = (-0.09*0.15) + (0.23*0.89) + (0.05*-0.44)  =  0.169
        #   Index 3 ('queen'):  Col 3 of W_out ⋅ h = ... =  0.774
        #   Index 4 ('palace'): Col 4 of W_out ⋅ h = ... =  0.072
        # scores array = [-0.023, 0.842, 0.169, 0.774, 0.072]
        scores = self.W_out.T @ h          # shape: (vocab_size,)
        
        # Step 3: convert to probabilities
        # Turn raw numbers into percentages/probabilities using Softmax.
        # probs Shape: (V,) -> (5,)
        # Example calculation output:
        # probs array = [ 0.05,  0.41,  0.09,  0.37,  0.08 ] (Notice they sum to 1.0)
        # Interpret: Model thinks there's a 9% chance the neighbor word is 'rules'.
        probs = self.softmax(scores)       # shape: (vocab_size,)
        
        return h, probs
    
    def backward(self, center_idx: int, context_idx: int, h: np.ndarray, probs: np.ndarray):
        """
        Backprop: nudge weights so the model predicts context_idx more strongly.

        Adjusts word vectors when predictions don't match reality.
        EXAMPLE WALKTHROUGH:
        - Center Word: 'king' (center_idx = 1)
        - True Context Word seen in text: 'rules' (context_idx = 2)
        - Model predicted probability distribution (probs): [0.05, 0.41, 0.09, 0.37, 0.08]
        """
        # Error: predicted probability vector minus one-hot truth
        # e.g. if context is "cat" at index 2: error = probs - [0,0,1,0,0,...]
        # Step 1: Calculate prediction error vector.
        # Target One-Hot Array for 'rules' (index 2) is: [0.0, 0.0, 1.0, 0.0, 0.0]
        # error = [0.05, 0.41, 0.09, 0.37, 0.08] - [0.0, 0.0, 1.0, 0.0, 0.0]
        # error Shape: (V,) -> (5,)
        # error Array = [ 0.05,  0.41, -0.91,  0.37,  0.08 ]
        # CRITICAL INSIGHT: The error at index 2 is negative (-0.91), meaning the model 
        # severely under-predicted it. Other words have positive errors (over-predicted).
        error = probs.copy()
        error[context_idx] -= 1.0         # shape: (vocab_size,)
        
        # Gradient for W_out: outer product of h and error
        # Step 2: Calculate updates for the Context Matrix (W_out).
        # We take the Outer Product of h (3,) and error (5,).
        # dW_out Shape: (D × V) -> (3 × 5) matches W_out exactly.
        # Element dW_out[dim, word_idx] = h[dim] * error[word_idx]
        # This creates an individual custom update trajectory for every single context vector cell.
        dW_out = np.outer(h, error)       # shape: (embed_dim, vocab_size)
        
        # Gradient for W_in: how much to update the center word's embedding
        # Step 3: Calculate updates for our Center Word Vector (W_in[center_idx]).
        # Accumulate the error across the vocabulary weightings.
        # Math: W_out (3×5) Matrix-Multiply by error vector (5,)
        # dW_in Shape: (D,) -> (3,)
        # Tells us exactly how to move our 3D 'king' vector row to optimize future predictions.
        dW_in  = self.W_out @ error       # shape: (embed_dim,)
        
        # Update weights
        self.W_out          -= self.lr * dW_out     # This shifts columns of W_out to pull 'rules' closer to 'king', while pushing others away.
        self.W_in[center_idx] -= self.lr * dW_in    # Nudges ONLY the specific row for 'king' inside W_in.
        
        # Return loss for logging
        # Step 5: Compute how poorly we performed (Negative Log-Likelihood Loss).
        # Our model predicted 0.09 (9%) for 'rules'.
        # Loss = -log(0.09) = 2.40. As probability moves toward 1.0, loss drops toward 0.
        return -np.log(probs[context_idx] + 1e-10)
    
    def train_pair(self, center_idx: int, context_idx: int):
        """Pipeline orchestration wrapper for executing a single step training cycle."""
        h, probs = self.forward(center_idx)
        loss     = self.backward(center_idx, context_idx, h, probs)
        return loss
    
    def get_embedding(self, word_idx: int) -> np.ndarray:
        """
        Extracts final combined word representations.
        Final embedding = average of input and output vectors.

        EXAMPLE SUMMARY:
        - 'king' center vector (W_in[1]):     [ 0.15,  0.89, -0.44 ]
        - 'king' context vector (W_out[:,1]): [ 0.11,  0.78, -0.30 ]
        - Final Averaged Embedding Vector:     [ 0.13,  0.835, -0.37 ]
        """
        return (self.W_in[word_idx] + self.W_out[:, word_idx]) / 2


# ── Build corpus and vocabulary ──────────────────────────────────────────────
corpus = [
    "the king rules the kingdom",
    "the queen rules the kingdom",
    "the man is strong",
    "the woman is strong",
    "the dog is an animal",
    "the cat is an animal",
    "the king is a man",
    "the queen is a woman",
    "the dog and cat are pets",
    "animals live in the kingdom",
]

# Tokenize
all_words = [w for sent in corpus for w in sent.lower().split()]
vocab      = list(set(all_words))
word2idx   = {w: i for i, w in enumerate(vocab)}
idx2word   = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

print(f"Vocabulary: {vocab_size} words")
print(f"Words: {vocab}\n")

# Generate training pairs (skip-gram with window=2)
def generate_pairs(sentences, window=2):
    pairs = []
    for sent in sentences:
        tokens = sent.lower().split()
        for i, center in enumerate(tokens):
            for j in range(max(0, i-window), min(len(tokens), i+window+1)):
                if j != i:
                    pairs.append((word2idx[center], word2idx[tokens[j]]))
    return pairs

pairs = generate_pairs(corpus, window=2)
print(f"Training pairs: {len(pairs)}")

# ── Train ────────────────────────────────────────────────────────────────────
model = Word2VecSkipGram(vocab_size=vocab_size, embed_dim=5, learning_rate=0.05)

for epoch in range(500):
    np.random.shuffle(pairs)
    total_loss = 0
    for center_idx, context_idx in pairs:
        total_loss += model.train_pair(center_idx, context_idx)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: loss = {total_loss/len(pairs):.4f}")

# ── Cosine similarity ─────────────────────────────────────────────────────────
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def most_similar(word, top_n=4):
    idx = word2idx[word]
    emb = model.get_embedding(idx)
    sims = {w: cosine_sim(emb, model.get_embedding(i)) 
            for w, i in word2idx.items() if w != word}
    return sorted(sims.items(), key=lambda x: -x[1])[:top_n]

# ── Results ───────────────────────────────────────────────────────────────────
print("\n=== Most similar words ===")
for query in ["king", "queen", "dog", "cat", "animals"]:
    print(f"\n'{query}' → {most_similar(query)}")

# ── Vector arithmetic ─────────────────────────────────────────────────────────
def analogy(a, b, c):
    """a is to b as c is to ???  →  b - a + c"""
    emb_a = model.get_embedding(word2idx[a])
    emb_b = model.get_embedding(word2idx[b])
    emb_c = model.get_embedding(word2idx[c])
    target = emb_b - emb_a + emb_c
    
    sims = {w: cosine_sim(target, model.get_embedding(i)) 
            for w, i in word2idx.items() if w not in {a,b,c}}
    return sorted(sims.items(), key=lambda x: -x[1])[:3]

print("\n=== Vector analogies ===")
print(f"king - man + woman = {analogy('king','man','woman')}")
print(f"king - man + woman = {analogy('king','woman','man')}")


'''

___ output ______________________________________________

Vocabulary: 20 words
Words: ['kingdom', 'are', 'a', 'cat', 'in', 'king', 'dog', 'an', 'the', 'woman', 'animal', 'rules', 'and', 'is', 'animals', 'man', 'queen', 'live', 'strong', 'pets']

Training pairs: 136
Epoch 100: loss = 1.9111
Epoch 200: loss = 1.8970
Epoch 300: loss = 1.8915
Epoch 400: loss = 1.8990
Epoch 500: loss = 1.9094

=== Most similar words ===

'king' → [('queen', np.float64(0.9985456320483825)), ('the', np.float64(0.8740868475406928)), ('a', np.float64(0.8396135875143028)), ('rules', np.float64(0.8173276722097262))]

'queen' → [('king', np.float64(0.9985456320483825)), ('the', np.float64(0.8582411379549564)), ('a', np.float64(0.8467786364835324)), ('rules', np.float64(0.8213556059956709))]

'dog' → [('cat', np.float64(0.9386092344092167)), ('and', np.float64(0.8496261941429715)), ('an', np.float64(0.6804214421872546)), ('is', np.float64(0.5672515838675628))]

'cat' → [('dog', np.float64(0.9386092344092167)), ('and', np.float64(0.9074765532939165)), ('are', np.float64(0.7410564829556069)), ('an', np.float64(0.6970644384351555))]

'animals' → [('live', np.float64(0.9029377841549022)), ('in', np.float64(0.8562227170268127)), ('kingdom', np.float64(0.4156253703540168)), ('pets', np.float64(0.23834910299396483))]

=== Vector analogies ===
king - man + woman = [('strong', np.float64(0.9531695926811845)), ('a', np.float64(0.7024949037576088)), ('is', np.float64(0.6855551249989044))]
king - man + woman = [('strong', np.float64(0.9531695926811845)), ('a', np.float64(0.7024949037576088)), ('is', np.float64(0.6855551249989044))]

'''