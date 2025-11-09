# Cell Corrections for Milestone_02.ipynb

## Changes Made to Fix Data Loading Issues

### Cell 19 - Baseline Model

**BEFORE (Problematic):**
```python
X_train = train_df['text'].values  # train_df may not exist after Cell 7
X_val = val_df['text'].values
X_test = test_df['text'].values

y_train = train_df['category'].values
y_val = val_df['category'].values
y_test = test_df['category'].values
```

**AFTER (Corrected):**
```python
# Load from preprocessed datasets (from Cell 7)
X_train = np.array(train_ds['text'])
X_val = np.array(val_ds['text'])
X_test = np.array(test_ds['text'])

y_train = np.array(train_ds['category'])
y_val = np.array(val_ds['category'])
y_test = np.array(test_ds['category'])
```

**Why:** This properly uses the data loaded from disk in Cell 7 (`train_ds`, `val_ds`, `test_ds`) instead of relying on `train_df` which may not exist.

**Additional improvement:**
```python
# Verify alignment
print(f"Label binarizer classes match label encoder: {np.array_equal(lb.classes_, le.classes_)}")
```

### Cell 32 - Enhanced Baseline with Class Weights

**Key Additions:**

1. **Proper class weight verification:**
```python
# Verify class weights are aligned (they are, per verify_pipeline.py)
class_weight_dict = dict(enumerate(class_weights))

print("\nClass weights (first 5):")
for i in range(5):
    print(f"  {i} ({le.classes_[i]}): {class_weights[i]:.4f}")
```

2. **Better training metrics:**
```python
print(f"Best epoch: {np.argmin(history.history['val_loss']) + 1}")
print(f"Best val loss: {np.min(history.history['val_loss']):.4f}")
print(f"Best val accuracy: {np.max(history.history['val_accuracy']):.4f}")
```

3. **Improved evaluation function:**
```python
def evaluate_model(model, X_test, y_test_enc, y_test, le):
    # Added both macro and weighted F1
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
```

## Data Flow Summary

```
Cell 4: Raw data → Preprocessing → DataFrames → Tokenization → Save to disk
        ├─> huffpost_processed_milestone2/ (Datasets with DistilBERT tokens)
        ├─> label_encoder.pkl (LabelEncoder fitted on categories)
        └─> class_weights.npy (Balanced class weights)

Cell 7: Load from disk
        ├─> train_ds, val_ds, test_ds (Datasets)
        ├─> le (LabelEncoder)
        └─> class_weights (numpy array)

Cell 19: Baseline Model (Simple TextVectorization)
        ├─> Extract raw text from train_ds['text']
        ├─> Use TextVectorization (NOT DistilBERT)
        └─> Train simple baseline

Cell 32: Enhanced Baseline
        ├─> Same data as Cell 19
        ├─> Same architecture as Cell 19
        ├─> Add: callbacks, class_weights
        └─> Expected: Similar performance, less overfitting

Cell XX: Problem 4 - Pretrained Model (TO BE IMPLEMENTED)
        ├─> Use train_ds['input_ids'] and train_ds['attention_mask']
        ├─> Load TFDistilBertForSequenceClassification
        └─> Fine-tune pretrained model
```

## What Each Cell Does

| Cell | Purpose | Uses DistilBERT Tokens? | Expected Accuracy |
|------|---------|------------------------|-------------------|
| Cell 19 | Simple baseline | ❌ No (TextVectorization) | ~60-61% |
| Cell 32 | Enhanced baseline | ❌ No (TextVectorization) | ~60-61% (less overfit) |
| Cell XX (Problem 4) | Pretrained model | ✅ Yes (input_ids) | ~70-80%+ |

## Important Notes

1. **Cell 19 and Cell 32 ignore DistilBERT tokenization**
   - This is FINE for baseline models
   - They use TensorFlow's TextVectorization instead
   - Simpler and appropriate for Problem 2

2. **DistilBERT tokens are for Problem 4**
   - The `input_ids` and `attention_mask` in the datasets
   - Use these when you load the actual DistilBERT model
   - This is where you'll see major performance gains

3. **Class weights are correctly aligned**
   - Verified by verify_pipeline.py
   - All showing ✓ in alignment check
   - Safe to use with `dict(enumerate(class_weights))`

4. **Expected performance:**
   - Baseline without class weights: ~60-61%
   - Baseline with class weights: ~60-61% (but better per-class performance)
   - Custom model (Problem 3): ~63-68% (with LSTM/GRU layers)
   - Pretrained DistilBERT: ~70-80%+ (proper transfer learning)

## Files Created for Reference

1. `verify_pipeline.py` - Confirms data is correct
2. `diagnose_model_issue.md` - Full explanation of issues
3. `cell19_corrected.py` - Corrected Cell 19 code
4. `cell32_corrected.py` - Corrected Cell 32 code
5. `CELL_CORRECTIONS.md` - This file

## Next Steps for Problem 3 (Custom Model)

Your "enhanced baseline" in Cell 32 is still the SAME architecture. For Problem 3, you need to:

**Add actual architectural changes:**
```python
# Example: Bidirectional LSTM architecture
inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
x = vectorizer(inputs)
x = tf.keras.layers.Embedding(input_dim=max_vocab, output_dim=128)(x)

# NEW: Add LSTM layers
x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(128, return_sequences=True)
)(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(64)
)(x)
x = tf.keras.layers.Dropout(0.3)(x)

# Dense layers
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

custom_model = tf.keras.Model(inputs, outputs)
```

This would be a genuinely different architecture and appropriate for Problem 3.
