# one_hot_problem.py
import numpy as np

vocab = ["king", "queen", "man", "woman", "dog", "cat"]
vocab_size = len(vocab)

def one_hot(word):
    idx = vocab.index(word)
    vec = np.zeros(vocab_size)
    vec[idx] = 1.0
    return vec

king  = one_hot("king")   # [1, 0, 0, 0, 0, 0]
queen = one_hot("queen")  # [0, 1, 0, 0, 0, 0]
dog   = one_hot("dog")    # [0, 0, 0, 0, 1, 0]

for word in vocab:
    print(f"{word}: {one_hot(word)}")

# Cosine similarity between any two words is ZERO
# Because all vectors are orthogonal — they share no dimensions
from numpy.linalg import norm
cos = lambda a, b: np.dot(a, b) / (norm(a) * norm(b))

print(cos(king, queen))  # 0.0  — model thinks king and queen are unrelated
print(cos(king, dog))    # 0.0  — same score! model can't tell the difference

# Also: vocab of 100k words = 100k-dimensional vectors
# 99.999% zeros — massive waste, no information in the structure

'''
___ Output ______________________________

king: [1. 0. 0. 0. 0. 0.]
queen: [0. 1. 0. 0. 0. 0.]
man: [0. 0. 1. 0. 0. 0.]
woman: [0. 0. 0. 1. 0. 0.]
dog: [0. 0. 0. 0. 1. 0.]
cat: [0. 0. 0. 0. 0. 1.]
0.0
0.0

'''