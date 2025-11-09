# Model Performance Issue Diagnosis

## ✓ GOOD NEWS: Class Weights Are Correctly Aligned
Your class weights are properly matched to your label encoder. This is NOT the issue.

## ✗ MAIN ISSUE: Data Pipeline Confusion

### The Problem in Cell 19 (Baseline Model):

```python
# Cell 19 - You're doing this:
X_train = train_df['text'].values  # ← Using train_df (raw text)
X_val = val_df['text'].values
X_test = test_df['text'].values
```

**BUT `train_df` is NOT defined in your notebook after Cell 7!**

Looking at your cells:
- Cell 4: Creates `train_df`, `val_df`, `test_df` from pandas
- Cell 6: Saves to disk and COMMENTS OUT the DataFrames
- Cell 7: Loads from disk into `train_ds`, `val_ds`, `test_ds` (Datasets, not DataFrames)

**In Cell 19, `train_df` should cause a NameError unless you ran Cell 4 without running Cell 7.**

### The Disconnect:

You have TWO separate workflows:

**Workflow A: Cell 4 (Your teammate's work)**
- Loads data → Creates DataFrames → Tokenizes with DistilBERT → Saves

**Workflow B: Cell 19 & 32 (Your baseline model)**
- Uses `train_df` (which shouldn't exist if you ran Cell 7)
- Re-tokenizes with TextVectorization (ignoring DistilBERT)
- This is a completely separate pipeline

## The Real Issues:

### 1. You're NOT using the preprocessed data from Cell 7

Cell 7 loads:
```python
train_ds = processed_datasets['train']  # Has input_ids, attention_mask, labels
```

Cell 19 uses:
```python
X_train = train_df['text'].values  # Raw text - ignores all the preprocessing!
```

### 2. Your baseline model is starting from scratch

- TextVectorization re-learns vocabulary from scratch
- Completely ignores the DistilBERT tokenization you already did
- This is fine for a simple baseline, BUT it means:
  - You wasted computation tokenizing with DistilBERT
  - Your preprocessing (Cell 4-6) is disconnected from your modeling (Cell 19+)

### 3. Your "enhanced" model (Cell 32) is the same architecture

Cell 32 says "rebuild baseline model with callbacks and class weights" but:
- It's the EXACT same architecture as Cell 19
- You just added callbacks and class weights
- This is good! But it's not a "custom model" - it's an enhanced baseline

## Why Performance Isn't Improving:

Looking at your Cell 32 results:
- Val accuracy peaks at ~60.7% at epoch 7
- Then validation loss INCREASES while training loss DECREASES
- Classic overfitting

**The model architecture is too simple:**
- Just: Embedding → GlobalAveragePooling → Dense(128) → Dense(37)
- This is appropriate for a baseline
- But it won't get much better than 60-61% accuracy

**What you need for Problem 3 (Custom Model):**
- Deeper architecture: Add LSTM, GRU, or more Dense layers
- Bidirectional layers
- Attention mechanisms
- More sophisticated pooling

**What you need for Problem 4 (Pretrained):**
- USE the DistilBERT tokenized data you already created
- Load actual DistilBERT model
- Fine-tune it

## Recommendations:

### Fix for Cell 19 (Baseline):

**Option A: Keep it simple and self-contained**
```python
# Get raw text from the loaded datasets
X_train = [train_ds[i]['text'] for i in range(len(train_ds))]
X_val = [val_ds[i]['text'] for i in range(len(val_ds))]
X_test = [test_ds[i]['text'] for i in range(len(test_ds))]

# Get labels
y_train = [train_ds[i]['category'] for i in range(len(train_ds))]
y_val = [val_ds[i]['category'] for i in range(len(val_ds))]
y_test = [test_ds[i]['category'] for i in range(len(test_ds))]

# Then continue with LabelBinarizer as you have
```

**Option B: Convert to numpy arrays properly**
```python
import numpy as np

X_train = np.array(train_ds['text'])
X_val = np.array(val_ds['text'])
X_test = np.array(test_ds['text'])

y_train = np.array(train_ds['category'])
y_val = np.array(val_ds['category'])
y_test = np.array(test_ds['category'])
```

### For Problem 3 (Custom Model):

You need to actually CHANGE the architecture. Try:

```python
# More sophisticated architecture
inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
x = vectorizer(inputs)
x = tf.keras.layers.Embedding(input_dim=max_vocab, output_dim=embedding_dim)(x)

# Add bidirectional LSTM
x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(128, return_sequences=True)
)(x)
x = tf.keras.layers.Dropout(0.3)(x)

# Add another LSTM layer
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

### For Problem 4 (Pretrained - DistilBERT):

NOW you use the DistilBERT tokenized data:

```python
from transformers import TFDistilBertForSequenceClassification

# Load the actual pretrained model
model = TFDistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_labels
)

# Prepare data in the format DistilBERT expects
train_inputs = {
    'input_ids': np.array(train_ds['input_ids']),
    'attention_mask': np.array(train_ds['attention_mask'])
}
train_labels = np.array(train_ds['labels'])

# Similar for val and test...

# Compile and train
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    epochs=3,  # Fewer epochs for pretrained
    batch_size=16,  # Smaller batch for BERT
    class_weight=dict(enumerate(class_weights))
)
```

## Summary:

1. ✓ Your preprocessing is correct
2. ✓ Your class weights are aligned
3. ✗ Your baseline model doesn't use the preprocessed data (but that's OK for a baseline)
4. ✗ Your "custom" model is just baseline + callbacks (not a different architecture)
5. ✗ You haven't used the DistilBERT tokenized data anywhere yet
6. ✗ Your model is overfitting and hitting a performance ceiling (~61%)

**To improve:**
- Problem 2 (Baseline): Current approach is fine, ~61% is reasonable for this simple model
- Problem 3 (Custom): Need to add LSTM/GRU layers or deeper architecture
- Problem 4 (Pretrained): Use the DistilBERT model with the tokenized data you prepared
