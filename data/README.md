# Data Artifacts

Preprocessed data, encoders, weights, and analysis outputs.

## Files

- **`label_encoder.pkl`** - Scikit-learn LabelEncoder for category labels (37 classes)
- **`class_weights.npy`** - Computed class weights for handling imbalance
- **`category_distribution.csv`** - Category frequency distribution
- **`category_overlaps.csv`** - Analysis of category overlaps
- **`missing_data_stats.csv`** - Missing data statistics
- **`text_length_stats.csv`** - Text length distribution analysis

## Usage

These artifacts are generated during data preprocessing (Problem 1) and can be loaded for model training:

```python
import pickle
import numpy as np

# Load label encoder
with open('data/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load class weights
class_weights = np.load('data/class_weights.npy')
```

## Note

The main dataset (`huffpost_processed_milestone2/`) is stored separately in the root directory.
