# Using Transformer Tokenizers with TF.keras - Milestone 2 Notes

## Current Situation

Your Milestone 1 implementation uses:
- **Tokenizer**: `AutoTokenizer.from_pretrained('distilbert-base-uncased')`
- **Framework**: PyTorch + Hugging Face Transformers
- **Model**: DistilBERT with frozen backbone

But Milestone 2 requires **TF.keras** models for the baseline and custom model sections.

## The Answer: Yes, It Works - With Some Adjustments

Using a Hugging Face transformer tokenizer with TF.keras is **completely compatible**. Here's how:

### 1. Tokenizer Output → TF.keras Input

The tokenizer produces dictionary outputs that can feed into TF.keras models:

```python
# Your current tokenization (works fine!)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

tokenized = tokenizer(
    texts,
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='tf'  # ← Key change: return TensorFlow tensors
)
```

### 2. For Custom TF.keras Models (Milestone 2)

You have two approaches:

#### Option A: Use just the input_ids with Keras Embedding layer
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=tokenizer.vocab_size,  # DistilBERT vocab size
        output_dim=128,
        input_length=128
    ),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Train with just input_ids
model.fit(tokenized['input_ids'], labels, ...)
```

#### Option B: Use functional API for multiple inputs
```python
# For using both input_ids and attention_mask
input_ids = tf.keras.Input(shape=(128,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.Input(shape=(128,), dtype=tf.int32, name='attention_mask')

embedding = tf.keras.layers.Embedding(tokenizer.vocab_size, 128)(input_ids)
# Use attention_mask for masking padded tokens
pooled = tf.keras.layers.GlobalAveragePooling1D()(embedding)
dense = tf.keras.layers.Dense(256, activation='relu')(pooled)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(dense)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
```

### 3. Key Considerations

**Pros of keeping the transformer tokenizer:**
- Already implemented and working
- Sophisticated tokenization (WordPiece)
- Handles special tokens properly
- Consistent with your Milestone 1 work

**What you need to change for Milestone 2:**
1. Switch from PyTorch models to TF.keras models
2. Use `return_tensors='tf'` in tokenizer
3. Build custom Keras architectures (not using pretrained DistilBERT for the "custom model" section)
4. For the pretrained section (Problem 4), you can use `TFAutoModelForSequenceClassification`

## Recommended Milestone 2 Structure

1. **Baseline Model (Problem 2)**: Simple Keras model with Embedding layer, using transformer tokenizer's input_ids
2. **Custom Model (Problem 3)**: Enhanced Keras model (add LSTM/GRU, more layers, etc.), still using tokenized inputs
3. **Pretrained Model (Problem 4)**: Use `TFDistilBertForSequenceClassification` from Hugging Face - this brings you back to similar territory as Milestone 1

## Bottom Line

The transformer tokenizer works seamlessly with TF.keras - you just need to build TF.keras models that consume the tokenized outputs instead of PyTorch models.
