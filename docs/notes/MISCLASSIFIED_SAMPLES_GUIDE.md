# Misclassified Samples Extraction Guide

## Overview

Two functions have been added to [Milestone_02.ipynb](Milestone_02.ipynb) to help extract and analyze misclassified samples from all three model architectures:

1. **`get_misclassified_samples()`** - Extracts misclassified samples from any model
2. **`display_misclassified_samples()`** - Pretty prints the misclassified samples

## Function Details

### `get_misclassified_samples()`

Extracts misclassified samples from Baseline, CNN, or DistilBERT models.

**Parameters:**
- `model`: The trained model (Keras or PyTorch)
- `X_test`: Test data (raw text for Keras models, HuggingFace Dataset for PyTorch)
- `y_test`: True labels (category names as strings)
- `model_type`: Either `'keras'` or `'pytorch'`
- `n_samples`: Number of misclassified samples to return per category (default: 5)
- `le`: LabelEncoder for converting between indices and category names
- `tokenizer`: AutoTokenizer (required for PyTorch models only)
- `device`: torch.device (required for PyTorch models only)

**Returns:**
A pandas DataFrame with columns:
- `text`: Original text
- `true_label`: Actual category
- `predicted_label`: Model's prediction
- `confidence`: Prediction confidence (0-1)
- `true_label_idx`: True label index
- `pred_label_idx`: Predicted label index

### `display_misclassified_samples()`

Pretty prints misclassified samples in a readable format.

**Parameters:**
- `df`: DataFrame from `get_misclassified_samples()`
- `max_text_length`: Maximum text length to display (default: 200)

## Usage Examples

### For Baseline Model (Keras)

```python
misclassified_baseline = get_misclassified_samples(
    model=baseline_model,
    X_test=X_test,  # Raw text strings
    y_test=y_test,  # Category names
    model_type='keras',
    n_samples=3,
    le=le
)

# Display first 10 samples
display_misclassified_samples(misclassified_baseline.head(10))

# Save for later analysis
misclassified_baseline.to_csv('misclassified_baseline.csv', index=False)
```

### For CNN Model (Keras)

```python
misclassified_cnn = get_misclassified_samples(
    model=cnn_model,
    X_test=X_test_vec,  # Use vectorized input for CNN
    y_test=y_test,
    model_type='keras',
    n_samples=3,
    le=le
)

display_misclassified_samples(misclassified_cnn.head(10))
misclassified_cnn.to_csv('misclassified_cnn.csv', index=False)
```

**Note:** For the CNN model, use `X_test_vec` (the vectorized version) instead of `X_test`!

### For DistilBERT Model (PyTorch)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained('best_distilbert_model').to(device)
tokenizer = AutoTokenizer.from_pretrained('best_distilbert_model')

# Extract misclassified samples
misclassified_distilbert = get_misclassified_samples(
    model=model,
    X_test=test_ds,  # HuggingFace Dataset
    y_test=y_test,
    model_type='pytorch',
    n_samples=3,
    le=le,
    tokenizer=tokenizer,
    device=device
)

display_misclassified_samples(misclassified_distilbert.head(10))
misclassified_distilbert.to_csv('misclassified_distilbert.csv', index=False)
```

## Key Differences Between Models

| Model | X_test Format | model_type | Extra Requirements |
|-------|---------------|------------|--------------------|
| Baseline | Raw text (`X_test`) | `'keras'` | `le` only |
| CNN | Vectorized (`X_test_vec`) | `'keras'` | `le` only |
| DistilBERT | HF Dataset (`test_ds`) | `'pytorch'` | `le`, `tokenizer`, `device` |

## Analysis Tips

1. **Compare across models**: Run all three and compare which categories are commonly misclassified
2. **Look for patterns**: Check if certain text patterns cause consistent errors
3. **Confidence analysis**: Low confidence predictions might indicate ambiguous samples
4. **Category confusion**: Check which categories get confused with each other

## Output Example

```
================================================================================
Sample 1
================================================================================
Text: Breaking: Scientists discover new species of deep-sea creature with bioluminescent properties...
True Label:      SCIENCE
Predicted Label: ENVIRONMENT
Confidence:      0.7234
================================================================================
```

## Location in Notebook

The functions are located just before **Problem 5** in [Milestone_02.ipynb](Milestone_02.ipynb), with example usage cells for each model type.
