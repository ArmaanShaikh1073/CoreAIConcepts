# sentencepiece_demo.py
import sentencepiece as spm

# ── Train a SentencePiece model ──────────────────────────────────────────────
# First, create a training corpus file
corpus = """
The quick brown fox jumps over the lazy dog.
मैं हिंदी में लिख रहा हूँ।
நான் தமிழில் எழுதுகிறேன்.
Python is a great programming language.
def tokenize(text): return text.split()
""".strip()

with open('corpus.txt', 'w', encoding='utf-8') as f:
    f.write(corpus)

# Train with BPE algorithm (can also use unigram)
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_tokenizer',
    vocab_size=200,            # small for demo; real models use 32k-100k
    model_type='bpe',          # or 'unigram' (used by LLaMA)
    character_coverage=0.9995, # crucial for multilingual!
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)

# ── Load and use ─────────────────────────────────────────────────────────────
sp = spm.SentencePieceProcessor()
sp.load('my_tokenizer.model')

sentences = [
    "The quick brown fox",
    "Programming in Python",
    "मैं AI सीख रहा हूँ",      # Hindi + English mix
    "def hello_world():",
]

for s in sentences:
    tokens = sp.encode(s, out_type=str)
    ids    = sp.encode(s, out_type=int)
    print(f"Input:  {s}")
    print(f"Tokens: {tokens}")
    print(f"IDs:    {ids}\n")

# KEY: SentencePiece uses ▁ (U+2581) to mark word starts
# "hello world" → ["▁hello", "▁world"]  (▁ means "preceded by space")
# This handles ALL languages without language-specific rules

# ── Decode back ──────────────────────────────────────────────────────────────
ids = sp.encode("Hello, world!", out_type=int)
decoded = sp.decode(ids)
print(f"Round-trip: {decoded}")  # "Hello, world!"


'''

# ── Output 1 ───────────────────────────────────────────────────────────────────

Input:  The quick brown fox
Tokens: ['▁The', '▁quick', '▁brown', '▁fox']
IDs:    [94, 119, 115, 97]

Input:  Programming in Python
Tokens: ['▁', 'P', 'rogr', 'amming', '▁', 'i', 'n', '▁Python']
IDs:    [138, 173, 88, 112, 138, 146, 145, 123]

Input:  मैं AI सीख रहा हूँ
Tokens: ['▁मैं', '▁', 'AI', '▁', 'स', 'ी', 'ख', '▁रहा', '▁हूँ']
IDs:    [100, 138, 1, 138, 1, 187, 182, 101, 103]

Input:  def hello_world():
Tokens: ['▁def', '▁', 'he', 'l', 'l', 'o', '_', 'w', 'o', 'r', 'l', 'd', '()', ':']
IDs:    [95, 138, 7, 150, 150, 141, 1, 180, 141, 142, 150, 161, 17, 172]

Round-trip:  ⁇ ello ⁇  world ⁇ 

'''