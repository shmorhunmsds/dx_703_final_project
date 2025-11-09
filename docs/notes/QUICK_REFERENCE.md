# Quick Reference - Milestone 2 Custom Model

## Files Created for You

### Main Implementation Files
1. **problem3_FINAL.py** - Copy this into your Problem 3 notebook cell ⭐
2. **cell19_corrected.py** - Fixed baseline model data loading
3. **custom_architectures.py** - Alternative architectures (if you want to try others)

### Verification & Documentation
4. **verify_pipeline.py** - Run to verify your data (already verified ✓)
5. **CELL_CORRECTIONS.md** - Explanation of all fixes
6. **diagnose_model_issue.md** - Full diagnosis of issues

## What to Do Now

### Step 1: Fix Cell 19 (Baseline)
Replace the data loading lines:
```python
# OLD (may cause issues):
X_train = train_df['text'].values

# NEW (correct):
X_train = np.array(train_ds['text'])
```
See [cell19_corrected.py](cell19_corrected.py) for complete code.

### Step 2: Add Problem 3 Custom Model
Copy the entire contents of [problem3_FINAL.py](problem3_FINAL.py) into a new cell for Problem 3.

This implements:
- Bidirectional LSTM (vs baseline's simple pooling)
- Dual pooling strategy (max + average)
- Deeper dense layers with BatchNormalization
- Better regularization (L2, higher dropout)

### Step 3: Run and Compare
The code automatically compares with your baseline and shows improvement.

## Expected Results

| Model | Test Accuracy | Macro F1 | Parameters |
|-------|--------------|----------|------------|
| Baseline | ~61.0% | ~0.45 | 3.86M |
| Custom (BiLSTM) | ~63-67% | ~0.48-0.52 | ~7-8M |
| Pretrained (DistilBERT) | ~70-80% | ~0.65-0.75 | ~66M |

## Key Architectural Differences

### Baseline (Problem 2)
```
Input → Vectorization → Embedding → GlobalAveragePooling → Dense(128) → Dense(37)
```

### Custom Model (Problem 3)
```
Input → Vectorization → Embedding → BiLSTM → [MaxPool + AvgPool] → Dense(256) → BN → Dense(128) → Dense(37)
```

**Key improvements:**
1. ✅ BiLSTM captures bidirectional context
2. ✅ Dual pooling captures more features
3. ✅ Deeper network with BatchNormalization
4. ✅ Better regularization (L2, dropout 0.5)

## Graded Questions for Problem 3

When answering the graded questions, mention:

**3.1 Model Design:**
- Added Bidirectional LSTM to capture context from both directions
- Implemented dual pooling (max + average) to capture different feature types
- Increased depth with 256→128 dense layers
- Added BatchNormalization for training stability
- Stronger regularization (L2 + higher dropout) to combat overfitting

**3.2 Training Results:**
- Report the test accuracy and macro F1 from the output
- Compare with baseline (should see 2-6% improvement)
- Note: validation curves should show less overfitting than baseline

**3.3 Interpretation:**
- BiLSTM helps capture sequential dependencies in text
- Dual pooling preserves more information than single average pooling
- BatchNormalization helps with training stability
- May train slower but generalizes better

**3.4 Reflection:**
- Increasing model complexity helps BUT requires careful regularization
- BiLSTM is computationally expensive compared to simple pooling
- Trade-off: better accuracy vs longer training time
- Still hitting a ceiling (~65-67%) - need pretrained models for higher accuracy

## Troubleshooting

### If you get NameError for train_df:
Use the corrected code from cell19_corrected.py that loads from `train_ds` instead.

### If training is too slow:
- Reduce LSTM units from 128 to 64
- Remove one Dense layer
- Or try the TextCNN model from custom_architectures.py (faster)

### If you want to try other architectures:
See [custom_architectures.py](custom_architectures.py) for:
- Stacked BiLSTM (v2)
- BiGRU (v3) - faster alternative
- TextCNN (v4) - fastest, good accuracy
- CNN-LSTM hybrid (v5)

## Next Steps (Problem 4 - For your teammate)

The DistilBERT tokenized data is ready in:
- `train_ds['input_ids']`
- `train_ds['attention_mask']`
- `train_ds['labels']`

Your teammate should:
1. Load `TFDistilBertForSequenceClassification`
2. Use the preprocessed input_ids and attention_mask
3. Fine-tune with small learning rate (2e-5)
4. Use smaller batch size (16 or 32)
5. Train for only 3-5 epochs

This should get ~70-80% accuracy.

## Summary

✅ Data loading fixed
✅ Class weights verified
✅ Custom architecture ready
✅ Comparison code included
✅ Ready for Problem 3

Just copy [problem3_FINAL.py](problem3_FINAL.py) into your notebook and run!
