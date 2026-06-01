# ── HuggingFace tokenizers (for open source models) ─────────────────────────
from transformers import AutoTokenizer

# LLaMA-3 / Mistral tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

text = "Tokenization is fundamental to LLMs."
result = tokenizer(text, return_tensors="pt")

print(f"\nInput IDs: {result['input_ids']}")
print(f"Token count: {result['input_ids'].shape[1]}")

# Decode
decoded = tokenizer.decode(result['input_ids'][0], skip_special_tokens=True)
print(f"Decoded: {decoded}")

# See individual tokens
tokens = tokenizer.convert_ids_to_tokens(result['input_ids'][0].tolist())
print(f"Tokens: {tokens}")
# ['<|begin_of_text|>', 'Token', 'ization', '▁is', '▁fundamental', '▁to', '▁LL', 'Ms', '.']
#  ^^^ special token                                                           ^^^^ LLM split!
