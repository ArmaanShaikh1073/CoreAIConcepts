# naive_tokenizer.py

text = "I'm running quickly! Don't stop."
print("Original text:", text)

# Approach 1: split on spaces
tokens = text.split()
print("Tokenization based on spaces:", tokens)
# ['I'm', 'running', 'quickly!', "Don't", 'stop.']
# Problem: punctuation sticks to words. "quickly!" ≠ "quickly"

# Approach 2: split on non-alphanumeric  
import re
tokens = re.findall(r'\w+', text)
print("Tokenization based on alphanumeric characters:", tokens)
# ['I', 'm', 'running', 'quickly', 'Don', 't', 'stop']
# Problem: "I'm" → ["I", "m"] which loses the contraction meaning

# The core issues with word-level tokenization:
# 1. Vocabulary explodes — "run","runs","running","ran" = 4 entries
# 2. OOV (out-of-vocabulary) problem — "ChatGPT" was never in training data
# 3. Different languages break differently
# 4. Numbers and code become a mess