# Overfitting Analysis and Fixes

## Current Results

| Metric | M2 DistilBERT | Final (w/ EDA + Class Merge) | Change |
|--------|---------------|------------------------------|--------|
| **Test Accuracy** | 72.24% | 73.99% | **+1.75%** |
| **Macro F1** | 0.6319 | 0.6447 | **+0.0128** |
| **Weighted F1** | 0.7159 | 0.7346 | +0.0187 |
| **Val Accuracy** | 72.70% | 73.90% | +1.2% |

## The Overfitting Problem

| Epoch | Train Acc | Val Acc | Val Loss | Gap |
|-------|-----------|---------|----------|-----|
| 1 | 62.29% | 72.52% | 0.955 | -10.2% (underfitting) |
| 2 | 82.43% | 73.81% | 0.993 | **8.6%** |
| 3 | 91.36% | 73.78% | 1.235 | **17.6%** |
| 4 | 95.58% | 73.54% | 1.581 | **22.0%** |
| 5 | 97.56% | 73.90% | 1.783 | **23.7%** |

**Key observation:** Val loss was best at epoch 1-2 but kept increasing while val accuracy stayed flat. The model is memorizing training data.

---

## Recommended Fixes

### 1. Early Stopping on Validation Loss (Priority: HIGH)

Currently saving best model based on `val_acc`. Switch to `val_loss` with patience:

```python
# In TRAIN_CONFIG, add:
'patience': 2,  # Stop if val_loss doesn't improve for 2 epochs

# In training loop, change:
if val_loss < best_val_loss:  # Instead of val_acc > best_val_acc
    best_val_loss = val_loss
    best_epoch = epoch
    patience_counter = 0
    # save model
else:
    patience_counter += 1
    if patience_counter >= TRAIN_CONFIG['patience']:
        print("Early stopping triggered!")
        break
```

### 2. Reduce EDA Augmentation (Priority: MEDIUM)

The augmented samples may be too similar to originals, making training "too easy":

```python
# In CONFIG, change:
'eda_num_augmentations': 1,    # Was 2
'eda_aug_probability': 0.15,   # Was 0.1 - more aggressive changes
```

Or try disabling EDA entirely to see if class consolidation alone helps.

### 3. Increase Regularization (Priority: MEDIUM)

```python
# In TRAIN_CONFIG, change:
'weight_decay': 0.05,  # Was 0.01

# Or add dropout to the classifier - requires modifying model config:
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    classifier_dropout=0.3,  # Add this
)
```

### 4. Reduce Training Duration (Priority: LOW)

Simple fix - just train fewer epochs:

```python
'epochs': 3,  # Was 5
```

### 5. Lower Learning Rate (Priority: LOW)

```python
'learning_rate': 1e-5,  # Was 2e-5
```

---

## Suggested Experiment Order

1. **First try:** Add early stopping on val_loss with patience=2
2. **If still overfitting:** Increase weight_decay to 0.05
3. **If still overfitting:** Reduce eda_num_augmentations to 1
4. **If still overfitting:** Add classifier_dropout=0.3
5. **Compare:** Run without EDA (just class consolidation) as baseline

---

## Quick Code Changes for Early Stopping

Replace the training loop section with:

```python
# Add to TRAIN_CONFIG
TRAIN_CONFIG = {
    'batch_size': 16,
    'learning_rate': 2e-5,
    'epochs': 5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'patience': 2,  # NEW: early stopping patience
}

# Replace best model tracking in training loop
start_time = time.time()
best_val_acc = 0
best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0  # NEW

for epoch in range(TRAIN_CONFIG['epochs']):
    # ... training code ...

    # Save best model based on val_loss (not val_acc)
    if val_loss < best_val_loss:
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0  # Reset
        model.save_pretrained('best_distilbert_final')
        tokenizer.save_pretrained('best_distilbert_final')
        print(f"  >> New best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= TRAIN_CONFIG['patience']:
            print(f"  >> Early stopping at epoch {epoch+1}")
            break
```

This should stop training around epoch 2-3 where val_loss is optimal.
