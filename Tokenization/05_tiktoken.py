# ── tiktoken (GPT-4, GPT-3.5, Claude uses same cl100k_base) ─────────────────
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4's tokenizer

text = "Hello world! I'm learning tokenization in 2026."
tokens = enc.encode(text)
print(f"Token IDs: {tokens}")
print(f"Token count: {len(tokens)}")  # critical for cost/context tracking

# Decode back
decoded = enc.decode(tokens)
assert decoded == text  # always true — lossless

# See what each token looks like
for tok_id in tokens:
    print(f"  {tok_id:6d} → '{enc.decode([tok_id])}'")

# IMPORTANT: count tokens BEFORE sending to API
def count_tokens(text: str, model: str = "gpt-4") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

print(f"\nCost check: '{text}' = {count_tokens(text)} tokens")


'''
___ output ________________________________________________________________________

Token IDs: [9906, 1917, 0, 358, 2846, 6975, 4037, 2065, 304, 220, 2366, 21, 13]
Token count: 13
    9906 → 'Hello'
    1917 → ' world'
       0 → '!'
     358 → ' I'
    2846 → ''m'
    6975 → ' learning'
    4037 → ' token'
    2065 → 'ization'
     304 → ' in'
     220 → ' '
    2366 → '202'
      21 → '6'
      13 → '.'

Cost check: 'Hello world! I'm learning tokenization in 2026.' = 13 tokens
'''