"""
Verification script to check the data pipeline and class weights alignment.
Run this in your notebook to diagnose issues.
"""

import numpy as np
import pickle
from datasets import load_from_disk
from collections import Counter

print("=" * 70)
print("PIPELINE VERIFICATION SCRIPT")
print("=" * 70)

# Load preprocessed data
processed_datasets = load_from_disk("huffpost_processed_milestone2")
train_ds = processed_datasets['train']
val_ds = processed_datasets['validation']
test_ds = processed_datasets['test']

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

class_weights = np.load('class_weights.npy')

print(f"\n1. DATASET SIZES:")
print(f"   Train: {len(train_ds)}")
print(f"   Val:   {len(val_ds)}")
print(f"   Test:  {len(test_ds)}")
print(f"   Total: {len(train_ds) + len(val_ds) + len(test_ds)}")

print(f"\n2. NUMBER OF CLASSES:")
print(f"   Label encoder classes: {len(le.classes_)}")
print(f"   Class weights length:  {len(class_weights)}")

print(f"\n3. LABEL ENCODER CLASSES (first 10):")
for i in range(min(10, len(le.classes_))):
    print(f"   {i}: {le.classes_[i]}")

print(f"\n4. CLASS WEIGHT VERIFICATION:")
print(f"   Min weight: {class_weights.min():.4f}")
print(f"   Max weight: {class_weights.max():.4f}")
print(f"   Mean weight: {class_weights.mean():.4f}")

# Get actual category counts from training data
print(f"\n5. CHECKING CLASS WEIGHTS ALIGNMENT:")
print(f"   {'Category':<20} {'Count':>8} {'Weight':>8} {'Expected Weight':>15}")
print("   " + "-" * 60)

# Count actual categories in training data
train_categories = [train_ds[i]['category'] for i in range(len(train_ds))]
category_counts = Counter(train_categories)

# For each class in label encoder, check the weight
for i in range(min(10, len(le.classes_))):
    category = le.classes_[i]
    count = category_counts[category]
    assigned_weight = class_weights[i]

    # Calculate what the weight SHOULD be (balanced weight formula)
    # balanced weight = total_samples / (n_classes * class_count)
    total_samples = len(train_ds)
    n_classes = len(le.classes_)
    expected_weight = total_samples / (n_classes * count)

    match = "✓" if abs(assigned_weight - expected_weight) < 0.01 else "✗ MISMATCH"
    print(f"   {category:<20} {count:>8} {assigned_weight:>8.4f} {expected_weight:>15.4f} {match}")

print(f"\n6. PREPROCESSED DATA FORMAT:")
print(f"   Columns in train_ds: {train_ds.column_names}")
sample = train_ds[0]
print(f"   Sample keys: {list(sample.keys())}")
print(f"   Has 'input_ids': {'input_ids' in sample}")
print(f"   Has 'attention_mask': {'attention_mask' in sample}")
print(f"   Has 'labels': {'labels' in sample}")

if 'input_ids' in sample:
    print(f"   input_ids length: {len(sample['input_ids'])}")
    print(f"   input_ids type: {type(sample['input_ids'])}")

print(f"\n7. LABEL ENCODING CHECK:")
# Check a few samples
for i in range(3):
    sample = train_ds[i]
    category = sample['category']
    encoded_label = sample['labels']
    # Verify encoding is correct
    expected_label = list(le.classes_).index(category)
    match = "✓" if encoded_label == expected_label else f"✗ Expected {expected_label}"
    print(f"   Sample {i}: '{category}' -> label {encoded_label} {match}")

print(f"\n8. CLASS DISTRIBUTION ACROSS SPLITS:")
print(f"   Checking if stratification worked...")
# Check a few categories
sample_categories = le.classes_[:5]
print(f"   {'Category':<20} {'Train %':>10} {'Val %':>10} {'Test %':>10}")
print("   " + "-" * 55)

for cat in sample_categories:
    train_cat_counts = sum(1 for c in train_categories if c == cat)
    val_categories = [val_ds[i]['category'] for i in range(len(val_ds))]
    test_categories = [test_ds[i]['category'] for i in range(len(test_ds))]

    val_cat_counts = sum(1 for c in val_categories if c == cat)
    test_cat_counts = sum(1 for c in test_categories if c == cat)

    train_pct = (train_cat_counts / len(train_ds)) * 100
    val_pct = (val_cat_counts / len(val_ds)) * 100
    test_pct = (test_cat_counts / len(test_ds)) * 100

    print(f"   {cat:<20} {train_pct:>9.2f}% {val_pct:>9.2f}% {test_pct:>9.2f}%")

print(f"\n9. RECOMMENDED CLASS WEIGHT DICT FOR MODEL:")
print("   Use this in model.fit():")
print("   class_weight_dict = {")
for i in range(min(5, len(le.classes_))):
    print(f"       {i}: {class_weights[i]:.4f},  # {le.classes_[i]}")
print("       ...")
print("   }")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

# Final check: Are weights correctly ordered?
print("\nFINAL DIAGNOSIS:")
print("Check section 5 above - if you see '✗ MISMATCH', your class weights")
print("are NOT aligned with your label encoder!")
print("\nIf all show '✓', then class weights are correctly aligned.")
