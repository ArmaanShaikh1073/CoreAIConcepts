# wordpiece_intuition.py
# WordPiece uses a different merge score:
#
#   score(A, B) = freq(AB) / (freq(A) * freq(B))
#
# This prefers merges where the combination is MORE informative
# than either part alone. "un" + "##usual" merges less eagerly
# than two characters that almost always appear together.

# Key difference from BPE:
# WordPiece uses ## prefix for continuation tokens (not </w> suffix)
# "unbelievable" → ["un", "##bel", "##iev", "##able"]
#                         ^^^ ## means "not a word start"

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("WordPiece tokenization with BERT's vocab:\n")

words = ["unbelievable", "tokenization", "ChatGPT", "running", "2024", "lower", "newest", "lowering", "newish"]
for word in words:
    tokens = tokenizer.tokenize(word)
    ids    = tokenizer.convert_tokens_to_ids(tokens)
    print(f"'{word}' → {tokens}")
    print(f"         → {ids}\n")