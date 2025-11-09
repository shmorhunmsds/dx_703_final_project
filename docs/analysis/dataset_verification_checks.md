# Dataset Verification Checks for Milestone 2

## Quick Verification Checklist

After running your preprocessing script, run these checks to verify your dataset is ready:

### 1. Verify Number of Classes

```python
# Check unique categories in original dataframe
print(f"Number of unique categories: {len(df_prob2['category'].unique())}")
print(f"Categories: {sorted(df_prob2['category'].unique())}")

# Check label encoder classes
print(f"\nNumber of encoded labels: {num_labels}")
print(f"Label encoder classes: {le.classes_}")

# Verify they match
assert len(df_prob2['category'].unique()) == num_labels, "Mismatch between categories and encoded labels!"
print("\n✓ Category count matches encoded labels")
```

### 2. Verify All Splits Have All Classes

```python
# Check each split has all 33 classes represented
train_cats = set(train_ds['category'])
val_cats = set(val_ds['category'])
test_cats = set(test_ds['category'])

print(f"Train set classes: {len(train_cats)}")
print(f"Val set classes: {len(val_cats)}")
print(f"Test set classes: {len(test_cats)}")

# Check for missing classes in any split
all_cats = set(df_prob2['category'].unique())
train_missing = all_cats - train_cats
val_missing = all_cats - val_cats
test_missing = all_cats - test_cats

if train_missing:
    print(f"⚠️ WARNING: Train set missing classes: {train_missing}")
if val_missing:
    print(f"⚠️ WARNING: Val set missing classes: {val_missing}")
if test_missing:
    print(f"⚠️ WARNING: Test set missing classes: {test_missing}")

if not (train_missing or val_missing or test_missing):
    print("✓ All splits contain all 33 classes")
```

### 3. Verify Split Sizes and Ratios

```python
# Check split sizes
total_samples = len(df_prob2)
train_size = len(train_ds)
val_size = len(val_ds)
test_size = len(test_ds)

print(f"Total samples after cleaning: {total_samples}")
print(f"Train: {train_size} ({train_size/total_samples*100:.1f}%)")
print(f"Val: {val_size} ({val_size/total_samples*100:.1f}%)")
print(f"Test: {test_size} ({test_size/total_samples*100:.1f}%)")

# Verify they sum correctly
assert train_size + val_size + test_size == total_samples, "Split sizes don't sum to total!"
print("\n✓ Split sizes sum correctly")

# Check approximate 70/15/15 split
train_ratio = train_size / total_samples
val_ratio = val_size / total_samples
test_ratio = test_size / total_samples

assert 0.68 <= train_ratio <= 0.72, f"Train ratio {train_ratio:.3f} not close to 0.70"
assert 0.13 <= val_ratio <= 0.17, f"Val ratio {val_ratio:.3f} not close to 0.15"
assert 0.13 <= test_ratio <= 0.17, f"Test ratio {test_ratio:.3f} not close to 0.15"
print("✓ Split ratios are approximately 70/15/15")
```

### 4. Verify Class Distribution (Stratification)

```python
# Check that class proportions are similar across splits
from collections import Counter

train_dist = Counter(train_ds['category'])
val_dist = Counter(val_ds['category'])
test_dist = Counter(test_ds['category'])

# Check a few representative classes
sample_classes = sorted(list(train_dist.keys()))[:5]

print("Class distribution verification (first 5 classes):")
print(f"{'Class':<20} {'Train %':>10} {'Val %':>10} {'Test %':>10}")
print("-" * 55)

for cat in sample_classes:
    train_pct = train_dist[cat] / train_size * 100
    val_pct = val_dist[cat] / val_size * 100
    test_pct = test_dist[cat] / test_size * 100
    print(f"{cat:<20} {train_pct:>9.2f}% {val_pct:>9.2f}% {test_pct:>9.2f}%")

print("\n✓ Check that percentages are similar across splits (stratification working)")
```

### 5. Verify Tokenization Applied Correctly

```python
# Check that tokenized fields exist
print("Checking tokenization...")
print(f"Train dataset columns: {train_ds.column_names}")

# Verify required fields exist
required_fields = ['input_ids', 'attention_mask', 'labels']
for field in required_fields:
    assert field in train_ds.column_names, f"Missing field: {field}"
    print(f"✓ {field} present")

# Check tokenization shapes
sample = train_ds[0]
print(f"\nSample tokenization:")
print(f"  input_ids shape: {len(sample['input_ids'])}")
print(f"  attention_mask shape: {len(sample['attention_mask'])}")
print(f"  Expected max_length: 128")

assert len(sample['input_ids']) == 128, "input_ids not correct length!"
assert len(sample['attention_mask']) == 128, "attention_mask not correct length!"
print("✓ Tokenization shapes correct")
```

### 6. Verify Labels are Properly Encoded

```python
# Check label encoding
print("Checking label encoding...")

# Get all unique labels from each split
train_labels = set(train_ds['labels'])
val_labels = set(val_ds['labels'])
test_labels = set(test_ds['labels'])

print(f"Train label range: {min(train_labels)} to {max(train_labels)}")
print(f"Val label range: {min(val_labels)} to {max(val_labels)}")
print(f"Test label range: {min(test_labels)} to {max(test_labels)}")

# Verify labels are in correct range
assert min(train_labels) >= 0, "Labels should start at 0"
assert max(train_labels) < num_labels, f"Max label {max(train_labels)} >= num_labels {num_labels}"
assert max(val_labels) < num_labels, f"Max val label {max(val_labels)} >= num_labels {num_labels}"
assert max(test_labels) < num_labels, f"Max test label {max(test_labels)} >= num_labels {num_labels}"

print(f"✓ All labels in range [0, {num_labels-1}]")

# Verify inverse transform works
sample_label = train_ds[0]['labels']
sample_category = le.inverse_transform([sample_label])[0]
print(f"\nLabel encoding test:")
print(f"  Label {sample_label} → Category '{sample_category}'")
print("✓ Label encoder working correctly")
```

### 7. Verify Class Weights Computed

```python
# Check class weights
print(f"Number of class weights: {len(class_weights)}")
print(f"Expected number: {num_labels}")
assert len(class_weights) == num_labels, "Class weights length mismatch!"

print(f"\nClass weight statistics:")
print(f"  Min weight: {class_weights.min():.4f}")
print(f"  Max weight: {class_weights.max():.4f}")
print(f"  Mean weight: {class_weights.mean():.4f}")

# Show weights for most/least common classes
category_counts = Counter(df_prob2['category'])
most_common = category_counts.most_common(3)
least_common = category_counts.most_common()[-3:]

print(f"\nWeights for most common classes:")
for cat, count in most_common:
    idx = list(le.classes_).index(cat)
    print(f"  {cat}: count={count}, weight={class_weights[idx]:.4f}")

print(f"\nWeights for least common classes:")
for cat, count in least_common:
    idx = list(le.classes_).index(cat)
    print(f"  {cat}: count={count}, weight={class_weights[idx]:.4f}")

print("\n✓ Class weights computed (higher weights for rare classes)")
```

### 8. Verify No Data Leakage

```python
# Check for duplicates across splits
train_texts = set(train_ds['text'])
val_texts = set(val_ds['text'])
test_texts = set(test_ds['text'])

train_val_overlap = train_texts & val_texts
train_test_overlap = train_texts & test_texts
val_test_overlap = val_texts & test_texts

print(f"Checking for data leakage...")
print(f"Train-Val overlap: {len(train_val_overlap)} samples")
print(f"Train-Test overlap: {len(train_test_overlap)} samples")
print(f"Val-Test overlap: {len(val_test_overlap)} samples")

if not (train_val_overlap or train_test_overlap or val_test_overlap):
    print("✓ No data leakage detected")
else:
    print("⚠️ WARNING: Data leakage detected!")
```

### 9. Display Sample Examples

```python
# Show a few examples to verify everything looks good
print("Sample preprocessed examples:\n")

for i in range(3):
    sample = train_ds[i]
    print(f"Example {i+1}:")
    print(f"  Text: {sample['text'][:100]}...")
    print(f"  Category: {sample['category']}")
    print(f"  Encoded label: {sample['labels']}")
    print(f"  Input IDs (first 10): {sample['input_ids'][:10]}")
    print(f"  Attention mask (first 10): {sample['attention_mask'][:10]}")
    print()
```

### 10. Final Summary Verification

```python
print("=" * 60)
print("DATASET READY FOR MODELING - FINAL SUMMARY")
print("=" * 60)
print(f"✓ Number of classes: {num_labels}")
print(f"✓ Train samples: {len(train_ds)}")
print(f"✓ Val samples: {len(val_ds)}")
print(f"✓ Test samples: {len(test_ds)}")
print(f"✓ Tokenizer: {model_name}")
print(f"✓ Max sequence length: 128")
print(f"✓ Class weights computed: Yes")
print(f"✓ Labels encoded: [0, {num_labels-1}]")
print(f"✓ Stratified splits: Yes")
print(f"✓ No data leakage: Check above")
print("=" * 60)
```

## Notes

- Run all these checks in order in a new notebook cell
- If any assertion fails, you'll know exactly what to fix
- The ✓ marks indicate successful checks
- ⚠️ warnings indicate potential issues to investigate
