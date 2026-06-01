# fasttext_demo.py
from gensim.models import FastText
import numpy as np

# fastText's key innovation:
# "running" = embedding of "running" + embeddings of its character n-grams:
#   <ru, run, unn, nni, nin, ing, ng>, <running>
# (angle brackets mark word boundaries)

# This means even unseen words get good embeddings by composing their n-gram embeddings

corpus = [
    "the cat sat on the mat",
    "the cats sat on the mats",
    "running makes you faster than walking",
    "the runner ran a marathon",
    "tokenization is a preprocessing step",
    "tokenizer splits text into tokens",
    "the king ruled the kingdom",
    "the queen rules the queendom",
]

sentences = [sent.split() for sent in corpus]

model = FastText(
    sentences   = sentences,
    vector_size = 50,          # embedding dimensions
    window      = 3,           # context window size
    min_count   = 1,           # include all words (normally set higher)
    epochs      = 200,
    min_n       = 2,           # minimum character n-gram length
    max_n       = 5,           # maximum character n-gram length
    sg          = 1,           # 1=skip-gram, 0=CBOW
)

# ── The magic: unseen words still get embeddings ──────────────────────────────
print("=== Similarity (seen words) ===")
print(f"cat vs cats:   {model.wv.similarity('cat', 'cats'):.3f}")
print(f"running vs runner: {model.wv.similarity('running', 'runner'):.3f}")
print(f"king vs queen: {model.wv.similarity('king', 'queen'):.3f}")

print("\n=== OOV words (never seen, but fastText handles them!) ===")
# These words were never in the corpus, but fastText can embed them
# by composing the character n-grams it DID learn
oov_words = ["tokenizing", "queenly", "kingly", "preprocess"]
for word in oov_words:
    vec = model.wv[word]   # works even for OOV!
    # Find most similar seen words
    try:
        similar = model.wv.most_similar(word, topn=3)
        print(f"'{word}' (OOV) → similar to: {[(w,f'{s:.2f}') for w,s in similar]}")
    except:
        print(f"'{word}' (OOV) → vector shape: {vec.shape}")  

print("\n=== Morphology understanding ===")
pairs = [("run","running"), ("token","tokenizer"), ("king","kingdom")]
for a, b in pairs:
    if a in model.wv and b in model.wv:
        print(f"'{a}' ↔ '{b}': similarity = {model.wv.similarity(a,b):.3f}")

# fastText vs Word2Vec: when to use which?
# fastText wins for:
#   - Morphologically rich languages (German, Finnish, Turkish)
#   - Code (method names share prefixes/suffixes)
#   - Medical/legal text (neologisms, compound terms)
#   - Small datasets (character n-grams = implicit data augmentation)
# Word2Vec wins for:
#   - Simple English tasks with large corpora
#   - When you need the smallest possible model


'''

___ output ________________________________________________________________

=== Similarity (seen words) ===
cat vs cats:   0.691
running vs runner: 0.630
king vs queen: 0.412

=== OOV words (never seen, but fastText handles them!) ===
'tokenizing' (OOV) → similar to: [('tokenizer', '0.83'), ('tokens', '0.74'), ('tokenization', '0.73')]
'queenly' (OOV) → similar to: [('queen', '0.78'), ('queendom', '0.58'), ('king', '0.53')]
'kingly' (OOV) → similar to: [('king', '0.67'), ('kingdom', '0.49'), ('walking', '0.45')]
'preprocess' (OOV) → similar to: [('preprocessing', '0.77'), ('mats', '0.47'), ('makes', '0.45')]

=== Morphology understanding ===
'run' ↔ 'running': similarity = 0.419
'token' ↔ 'tokenizer': similarity = 0.868
'king' ↔ 'kingdom': similarity = 0.626

'''
