"""Quick vocab check - add as notebook cell"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Check actual vocab size
vectorizer_full = tf.keras.layers.TextVectorization(
    max_tokens=None,  # No limit
    output_sequence_length=128,
    standardize='lower_and_strip_punctuation'
)
vectorizer_full.adapt(X_train)
vocab = vectorizer_full.get_vocabulary()

print(f"Actual vocab size: {len(vocab):,}")
print(f"Current max_vocab: 30,000")
print(f"Coverage: {min(30000/len(vocab)*100, 100):.1f}%")

# Get token frequencies
X_train_tokens = vectorizer_full(X_train).numpy()
from collections import Counter
token_counts = Counter()
for seq in X_train_tokens:
    for tok in seq:
        if tok > 0: token_counts[tok] += 1

freqs = sorted(token_counts.values(), reverse=True)
total = sum(freqs)

# Coverage at different sizes
print("\nCoverage by vocab size:")
for size in [10000, 20000, 30000, 50000]:
    if size <= len(freqs):
        cov = sum(freqs[:size])/total*100
        print(f"  {size:>6,}: {cov:>5.2f}%")

# Find optimal for 99% coverage
cumsum = np.cumsum(freqs)
for i, val in enumerate(cumsum):
    if val/total >= 0.99:
        print(f"\nFor 99% coverage: max_vocab = {i+1:,}")
        break
