# Full BPE tokenizer — training + encoding

from collections import defaultdict, Counter

# ── Step 1: represent the corpus as character sequences ──────────────────────
def get_vocab(corpus: list[str]) -> dict:
    """Convert corpus to character-level with </w> end-of-word marker."""
    vocab = defaultdict(int)
    for word in corpus:
        # Each word → tuple of characters + end marker
        tokenized = tuple(list(word) + ['</w>'])
        vocab[tokenized] += 1
    return dict(vocab)

# ── Step 2: count all adjacent pairs ────────────────────────────────────────
def get_stats(vocab: dict) -> Counter:
    """Count frequency of every adjacent pair across all words."""
    pairs = Counter()
    for word_tokens, freq in vocab.items():
        for i in range(len(word_tokens) - 1):
            pairs[(word_tokens[i], word_tokens[i+1])] += freq
    return pairs

# ── Step 3: merge a pair everywhere in the vocabulary ───────────────────────
def merge_vocab(pair: tuple, vocab: dict) -> dict:
    """Replace all occurrences of pair with merged token."""
    new_vocab = {}
    bigram = ' '.join(pair)           # e.g. ('l','o') → 'l o'
    merged = ''.join(pair)            # e.g. ('l','o') → 'lo'
    
    for word_tokens, freq in vocab.items():
        # Rebuild the token tuple, merging where pair appears
        new_tokens = []
        i = 0
        word_list = list(word_tokens)
        while i < len(word_list):
            if (i < len(word_list) - 1 and 
                word_list[i] == pair[0] and 
                word_list[i+1] == pair[1]):
                new_tokens.append(merged)
                i += 2  # skip both
            else:
                new_tokens.append(word_list[i])
                i += 1
        new_vocab[tuple(new_tokens)] = freq
    return new_vocab

# ── Step 4: train BPE ────────────────────────────────────────────────────────
def train_bpe(corpus: list[str], num_merges: int) -> tuple[dict, list]:
    """
    Returns:
        vocab: final vocabulary (token tuples → frequency)
        merges: ordered list of merge rules applied
    """
    vocab = get_vocab(corpus)
    merges = []
    
    for i in range(num_merges):
        stats = get_stats(vocab)
        if not stats:
            break
        
        # Always merge the most frequent pair
        best_pair = max(stats, key=stats.get)
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)
        
        print(f"Merge {i+1}: {best_pair} → {''.join(best_pair)}  (freq={stats[best_pair]})")
    
    return vocab, merges

# ── Step 5: encode new text using learned merges ────────────────────────────
def encode(word: str, merges: list) -> list[str]:
    """Apply merge rules (in order!) to tokenize a new word."""
    tokens = list(word) + ['</w>']
    
    for pair in merges:
        merged = ''.join(pair)
        i = 0
        new_tokens = []
        while i < len(tokens):
            if (i < len(tokens) - 1 and 
                tokens[i] == pair[0] and 
                tokens[i+1] == pair[1]):
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    
    return tokens

# ── Run it ───────────────────────────────────────────────────────────────────
corpus = ["low", "low", "low", "wider", "newest", "newer", "new"]  # corpus 1 = gives output 1
#corpus = ["low", "low", "low", "wider", "newest", "newer", "new", "lower", "nearer", "latest", "dearer", "bearer"] # corpus 2 = gives output 2

print("Training BPE with 10 merges...\n")
final_vocab, merges = train_bpe(corpus, num_merges=10)

print("\nFinal vocabulary tokens:")
for tokens, freq in sorted(final_vocab.items(), key=lambda x: -x[1]):
    print(f"  {' | '.join(tokens):30s} freq={freq}")

print("\nEncoding unseen words:")
for word in ["lower", "newest", "lowering", "newish"]:
    print(f"  '{word}' → {encode(word, merges)}")


'''
# ── Output 1 ───────────────────────────────────────────────────────────────────

Training BPE with 10 merges...

Merge 1: ('w', '</w>') → w</w>  (freq=4)
Merge 2: ('l', 'o') → lo  (freq=3)
Merge 3: ('lo', 'w</w>') → low</w>  (freq=3)
Merge 4: ('n', 'e') → ne  (freq=3)
Merge 5: ('e', 'r') → er  (freq=2)
Merge 6: ('er', '</w>') → er</w>  (freq=2)
Merge 7: ('ne', 'w') → new  (freq=2)
Merge 8: ('w', 'i') → wi  (freq=1)
Merge 9: ('wi', 'd') → wid  (freq=1)
Merge 10: ('wid', 'er</w>') → wider</w>  (freq=1)

Final vocabulary tokens:
  low</w>                        freq=3
  wider</w>                      freq=1
  new | e | s | t | </w>         freq=1
  new | er</w>                   freq=1
  ne | w</w>                     freq=1

Encoding unseen words:
  'lower' → ['lo', 'w', 'er</w>']
  'newest' → ['new', 'e', 's', 't', '</w>']
  'lowering' → ['lo', 'w', 'er', 'i', 'n', 'g', '</w>']
  'newish' → ['new', 'i', 's', 'h', '</w>']

'''


'''
# ── Output 2 ───────────────────────────────────────────────────────────────────

Training BPE with 10 merges...

Merge 1: ('e', 'r') → er  (freq=6)
Merge 2: ('er', '</w>') → er</w>  (freq=6)
Merge 3: ('l', 'o') → lo  (freq=4)
Merge 4: ('lo', 'w') → low  (freq=4)
Merge 5: ('n', 'e') → ne  (freq=4)
Merge 6: ('low', '</w>') → low</w>  (freq=3)
Merge 7: ('ne', 'w') → new  (freq=3)
Merge 8: ('a', 'r') → ar  (freq=3)
Merge 9: ('ar', 'er</w>') → arer</w>  (freq=3)
Merge 10: ('e', 's') → es  (freq=2)

Final vocabulary tokens:
  low</w>                        freq=3
  w | i | d | er</w>             freq=1
  new | es | t | </w>            freq=1
  new | er</w>                   freq=1
  new | </w>                     freq=1
  low | er</w>                   freq=1
  ne | arer</w>                  freq=1
  l | a | t | es | t | </w>      freq=1
  d | e | arer</w>               freq=1
  b | e | arer</w>               freq=1

Encoding unseen words:
  'lower' → ['low', 'er</w>']
  'newest' → ['new', 'es', 't', '</w>']
  'lowering' → ['low', 'er', 'i', 'n', 'g', '</w>']
  'newish' → ['new', 'i', 's', 'h', '</w>']

'''