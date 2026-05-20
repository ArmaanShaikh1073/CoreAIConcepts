# glove_intuition.py
# GloVe in essence: factorize the log co-occurrence matrix

import numpy as np

# The core insight:
# log P(word_j | word_i) ≈ w_i · w_j + b_i + b_j
# where P is the co-occurrence probability
# This means: the DOT PRODUCT of two embeddings predicts how often they co-occur
# Words that co-occur a lot → similar embeddings (high dot product)

# Simplified co-occurrence matrix for intuition
words = ["ice", "steam", "solid", "gas", "water", "fashion"]
co_matrix = np.array([
    # ice  steam solid  gas  water fashion
    [  10,   1,    8,   1,   5,    1  ],  # ice
    [   1,  10,    1,   8,   5,    1  ],  # steam
    [   8,   1,   10,   1,   4,    1  ],  # solid
    [   1,   8,    1,  10,   4,    1  ],  # gas
    [   5,   5,    4,   4,  10,    1  ],  # water
    [   1,   1,    1,   1,   1,   10  ],  # fashion
], dtype=float)

# Ratio P(k|ice) / P(k|steam) for a probe word k reveals semantics:
# k = "solid":  ratio >> 1 → "solid" is associated with ice, not steam
# k = "gas":    ratio << 1 → "gas" is associated with steam, not ice
# k = "water":  ratio ≈ 1  → "water" relates to both
# k = "fashion":ratio ≈ 1  → neither

probe_words = ["solid", "gas", "water", "fashion"]
ice_row   = co_matrix[0]
steam_row = co_matrix[1]

print("Co-occurrence ratio analysis (ice vs steam):")
print(f"{'word':<12} {'P(w|ice)':<12} {'P(w|steam)':<12} {'ratio':<8} {'signal'}")
print("-" * 60)
for i, w in enumerate(probe_words):
    p_ice   = co_matrix[0][i] / co_matrix[0].sum()
    p_steam = co_matrix[1][i] / co_matrix[1].sum()
    ratio   = p_ice / (p_steam + 1e-10)
    signal  = "→ ice" if ratio > 2 else ("→ steam" if ratio < 0.5 else "neutral")
    print(f"{w:<12} {p_ice:<12.3f} {p_steam:<12.3f} {ratio:<8.2f} {signal}")

# GloVe's loss function:
# J = Σ f(X_ij) * (w_i·w_j + b_i + b_j - log X_ij)²
# where f(x) = weighting function that downweights very common co-occurrences
# Training minimizes this loss → embeddings encode co-occurrence structure


'''
_____ Challenge _____________________________________________________

text-corpora statistics: the fact that words like "the", "is", and "and" appear millions of times, while meaningful words like "steam" or "ice" appear rarely.
If you didn't have glove_weighting, your word vectors wouldn't learn real semantics. They would just learn how to group every single word close to the word "the"!

To fix this, Stanford researchers designed a saturation curve function
F(x) = (x / x_max)^alpha if x < x_max
     =  1 otherwise

Scenario A: Rare Word Co-occurrence (Count = 1)
Imagine the words "quantum" and "physics" appear together exactly 1 time in your text.
The code checks: Is 1 < 100? 
Yes.Math: (1 / 100)^{0.75} = (0.01)^{0.75} ~ 0.031 = 3.1%
Result: This pair gets a tiny weight (3.1%). 
This prevents random, one-off typos or accidental word alignments from changing your embedding geometry.

Scenario B: Sweet-Spot Semantic Co-occurrence (Count = 50)
Imagine "ice" and "solid" appear together 50 times.
The code checks: Is 50 < 100? 
Yes.Math: (50 / 100)^{0.75} = (0.5)^{0.75} ~ 0.594 = 59.4%
Result: This pair gets a strong, healthy weight of 59.4%. 
The model will actively change the geometric shapes of "ice" and "solid" to pull them closer together.

Scenario C: Mega-Frequent Stop Words (Count = 500,000)
Imagine "the" and "and" appear together 500,000 times.
The code checks: Is 500,000 < 100? No.It triggers the else 1.0 part of the code block.
Result: The weight hits a hard ceiling at 1.0 (100%).
'''

def glove_weighting(x, x_max=100, alpha=0.75):
    """Words that co-occur more than x_max times are weighted equally."""
    return (x / x_max) ** alpha if x < x_max else 1.0

print("\nGloVe weighting function (prevents 'the' dominating everything):")
for count in [1, 5, 20, 50, 100, 500]:
    w = glove_weighting(count)
    bar = '█' * int(w * 20)
    print(f"  count={count:4d}  weight={w:.3f}  {bar}")


'''

___ output ________________________________________________________________

Co-occurrence ratio analysis (ice vs steam):
word         P(w|ice)     P(w|steam)   ratio    signal
------------------------------------------------------------
solid        0.385        0.038        10.00    → ice
gas          0.038        0.385        0.10     → steam
water        0.308        0.038        8.00     → ice
fashion      0.038        0.308        0.12     → steam

GloVe weighting function (prevents 'the' dominating everything):
  count=   1  weight=0.032                              # If count = 1, it means a pair of words like ("quantum", "physics") appeared next to each other exactly once in the whole dataset.
  count=   5  weight=0.106  ██
  count=  20  weight=0.299  █████
  count=  50  weight=0.595  ███████████
  count= 100  weight=1.000  ████████████████████
  count= 500  weight=1.000  ████████████████████        # If count = 500, it means a pair of words like ("the", "and") appeared next to each other 500 times (or thousands of times) in the dataset.

'''