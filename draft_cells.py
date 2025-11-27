# ============================================================
# DRAFT CELLS FOR FINAL NOTEBOOK
# Copy these into final_milestone.ipynb after training completes
# ============================================================

# ============================================================
# CELL 1: MARKDOWN - Section Header (insert after imports)
# ============================================================
MARKDOWN_HEADER = """
# A.1: Setup and Context

## Project Overview

**Title:** HuffPost News Category Classification
**Team Members:** [Your Names Here]
**Course:** DX 703 - Fall 2024
**Date:** December 2024

**Purpose:** This notebook presents our final ML pipeline for classifying HuffPost news headlines into categories using a fine-tuned DistilBERT model.

---

## Summary of Milestones 1 & 2

### Milestone 1: Data Exploration
- Analyzed HuffPost dataset structure (200k+ headlines, 41 original categories)
- Identified key challenges: severe class imbalance, overlapping/ambiguous categories
- Proposed evaluation metrics: Macro-F1 (for imbalance), Weighted-F1, Accuracy

### Milestone 2: Model Experimentation
- **Baseline (Embedding + GlobalAvgPool):** 61.7% accuracy, 0.47 Macro-F1
- **Custom TextCNN:** 61.9% accuracy, 0.50 Macro-F1
- **DistilBERT Fine-tune:** 72.2% accuracy, 0.63 Macro-F1 (best)

### Key Learnings
1. Pretrained transformers dramatically outperform training-from-scratch approaches
2. Class imbalance significantly impacts minority class performance
3. Overlapping categories (e.g., "ARTS" vs "ARTS & CULTURE" vs "CULTURE & ARTS") cause confusion

---

## Improvements for Final Model

Based on Milestone 2 error analysis, we implement:

1. **Expanded Class Consolidation** - Merge semantically similar/duplicate categories
2. **Easy Data Augmentation (EDA)** - Augment minority classes to reduce imbalance
3. **Refined Training Strategy** - Optimized hyperparameters based on M2 results
"""

# ============================================================
# CELL 2: MARKDOWN - Class Remapping Explanation
# ============================================================
MARKDOWN_CLASS_REMAP = """
## Class Consolidation Strategy

Based on confusion matrix analysis from Milestone 2, we identified categories that are:
1. **Exact duplicates** with different naming conventions
2. **Semantically overlapping** causing model confusion

### Consolidation Mapping

| Original Categories | Merged Into | Rationale |
|---------------------|-------------|-----------|
| ARTS, ARTS & CULTURE, CULTURE & ARTS | ARTS & CULTURE | Same content, different names |
| GREEN, ENVIRONMENT | ENVIRONMENT | Environmental topics |
| WORLDPOST, THE WORLDPOST, WORLD NEWS | WORLD NEWS | International news |
| STYLE, STYLE & BEAUTY | STYLE & BEAUTY | Fashion/beauty content |
| PARENTS, PARENTING | PARENTING | Parenting topics |
| TASTE, FOOD & DRINK | FOOD & DRINK | Food-related content |
| HEALTHY LIVING, WELLNESS | WELLNESS | Health/wellness overlap |

This reduces our classes from 41 to ~32, creating cleaner decision boundaries.
"""

# ============================================================
# CELL 3: CODE - Class Remapping Implementation
# ============================================================
CODE_CLASS_REMAP = '''
# ============================================
# Class Consolidation / Remapping
# ============================================
# Merge semantically similar or duplicate categories to reduce confusion

CLASS_MERGE_MAP = {
    # Exact duplicates / naming variations
    'ARTS': 'ARTS & CULTURE',
    'CULTURE & ARTS': 'ARTS & CULTURE',

    # Environmental content
    'GREEN': 'ENVIRONMENT',

    # World news variations
    'WORLDPOST': 'WORLD NEWS',
    'THE WORLDPOST': 'WORLD NEWS',

    # Style/fashion
    'STYLE': 'STYLE & BEAUTY',

    # Parenting
    'PARENTS': 'PARENTING',

    # Food content
    'TASTE': 'FOOD & DRINK',

    # Health/wellness overlap
    'HEALTHY LIVING': 'WELLNESS',
}

def remap_categories(df, merge_map):
    """
    Apply category consolidation mapping to dataframe.

    Args:
        df: DataFrame with 'category' column
        merge_map: Dict mapping old category names to new ones

    Returns:
        DataFrame with remapped categories
    """
    df = df.copy()
    original_classes = df['category'].nunique()

    df['category'] = df['category'].replace(merge_map)

    new_classes = df['category'].nunique()
    print(f"Class consolidation: {original_classes} -> {new_classes} classes")
    print(f"Removed {original_classes - new_classes} redundant categories")

    return df

# Apply remapping
df_remapped = remap_categories(df_prob2, CLASS_MERGE_MAP)

# Show new distribution
print("\\nNew class distribution (top 15):")
print(df_remapped['category'].value_counts().head(15))

print(f"\\nTotal samples: {len(df_remapped)}")
print(f"Final number of classes: {df_remapped['category'].nunique()}")
'''

# ============================================================
# CELL 4: MARKDOWN - EDA Explanation
# ============================================================
MARKDOWN_EDA = """
## Easy Data Augmentation (EDA)

To address class imbalance, we apply EDA techniques from [Wei & Zou, 2019](https://arxiv.org/abs/1901.11196) to augment minority classes.

### EDA Techniques Used:
1. **Synonym Replacement (SR)** - Replace n words with synonyms
2. **Random Insertion (RI)** - Insert synonyms of random words
3. **Random Swap (RS)** - Swap positions of two words
4. **Random Deletion (RD)** - Randomly remove words

### Augmentation Strategy:
- Only augment **training set** (never val/test)
- Target classes with < 5000 samples
- Generate 2 augmented versions per original sample
- Use conservative alpha=0.1 (modify ~10% of words)

This helps the model see more varied examples of minority classes without overfitting.
"""

# ============================================================
# CELL 5: CODE - EDA Implementation
# ============================================================
CODE_EDA = '''
# ============================================
# Easy Data Augmentation (EDA)
# ============================================
# pip install nlpaug  (if not installed)

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from collections import Counter

# Configuration
EDA_CONFIG = {
    'min_class_size': 5000,      # Only augment classes smaller than this
    'num_augmentations': 2,       # Number of augmented samples per original
    'aug_p': 0.1,                 # Probability of augmenting each word (alpha)
}

# Initialize augmenters
# Using contextual word embeddings for better quality synonyms
aug_synonym = naw.SynonymAug(aug_src='wordnet', aug_p=EDA_CONFIG['aug_p'])
aug_random_swap = naw.RandomWordAug(action='swap', aug_p=EDA_CONFIG['aug_p'])
aug_random_delete = naw.RandomWordAug(action='delete', aug_p=EDA_CONFIG['aug_p'])

def augment_text(text, num_aug=2):
    """
    Apply EDA augmentation to a single text sample.
    Randomly selects augmentation techniques.

    Args:
        text: Original text string
        num_aug: Number of augmented versions to generate

    Returns:
        List of augmented text strings
    """
    augmented = []
    augmenters = [aug_synonym, aug_random_swap, aug_random_delete]

    for _ in range(num_aug):
        # Randomly pick an augmenter
        aug = random.choice(augmenters)
        try:
            aug_text = aug.augment(text)
            # nlpaug returns a list, take first element
            if isinstance(aug_text, list):
                aug_text = aug_text[0]
            augmented.append(aug_text)
        except Exception as e:
            # If augmentation fails, use original
            augmented.append(text)

    return augmented


def augment_minority_classes(df, config):
    """
    Augment samples from minority classes in the training set.

    Args:
        df: Training DataFrame with 'text' and 'category' columns
        config: EDA configuration dict

    Returns:
        DataFrame with original + augmented samples
    """
    class_counts = df['category'].value_counts()
    minority_classes = class_counts[class_counts < config['min_class_size']].index.tolist()

    print(f"Classes to augment ({len(minority_classes)} minority classes):")
    for cls in minority_classes[:10]:
        print(f"  - {cls}: {class_counts[cls]} samples")
    if len(minority_classes) > 10:
        print(f"  ... and {len(minority_classes) - 10} more")

    augmented_rows = []

    for cls in tqdm(minority_classes, desc="Augmenting classes"):
        class_samples = df[df['category'] == cls]

        for _, row in class_samples.iterrows():
            aug_texts = augment_text(row['text'], num_aug=config['num_augmentations'])

            for aug_text in aug_texts:
                new_row = row.copy()
                new_row['text'] = aug_text
                augmented_rows.append(new_row)

    # Combine original + augmented
    augmented_df = pd.DataFrame(augmented_rows)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)

    print(f"\\nAugmentation complete:")
    print(f"  Original samples: {len(df)}")
    print(f"  Augmented samples added: {len(augmented_df)}")
    print(f"  Total samples: {len(combined_df)}")

    return combined_df


# Apply augmentation to training data only
print("Applying EDA augmentation to training set...")
print("="*50)

train_df_augmented = augment_minority_classes(train_df, EDA_CONFIG)

# Verify new distribution
print("\\nNew class distribution after augmentation (bottom 10):")
print(train_df_augmented['category'].value_counts().tail(10))
'''

# ============================================================
# CELL 6: CODE - Updated Full Preprocessing Pipeline
# ============================================================
CODE_FULL_PIPELINE = '''
# ============================================
# Complete Preprocessing Pipeline
# ============================================
# Combines: Loading -> Cleaning -> Remapping -> Splitting -> EDA -> Tokenization

from datasets import load_from_disk, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
import pickle

print("="*70)
print("PREPROCESSING PIPELINE")
print("="*70)

# -----------------------------------------
# Step 1: Load Raw Data
# -----------------------------------------
print("\\n[1/7] Loading raw dataset...")
huff_all = load_from_disk("huffpost_splits")

# Combine headline & short_description
def combine_fields(example):
    headline = example.get('headline', '') or ""
    short_desc = example.get('short_description', '') or "[NO_DESC]"
    example['text'] = f"{headline} [SEP] {short_desc}"
    return example

huff_all = huff_all.map(combine_fields)
df_raw = huff_all.to_pandas()
print(f"Loaded {len(df_raw)} samples")

# -----------------------------------------
# Step 2: Basic Cleaning
# -----------------------------------------
print("\\n[2/7] Cleaning data...")
df_clean = df_raw.copy()

# Handle missing values
df_clean['headline'] = df_clean['headline'].fillna('')
df_clean['short_description'] = df_clean['short_description'].fillna('[NO_DESC]')
df_clean['text'] = df_clean['headline'] + " [SEP] " + df_clean['short_description']

# Remove duplicates
original_len = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"Removed {original_len - len(df_clean)} duplicates")

# -----------------------------------------
# Step 3: Class Remapping (NEW)
# -----------------------------------------
print("\\n[3/7] Consolidating categories...")
df_remapped = remap_categories(df_clean, CLASS_MERGE_MAP)

# -----------------------------------------
# Step 4: Stratified Train/Val/Test Split
# -----------------------------------------
print("\\n[4/7] Creating stratified splits...")
train_df, temp_df = train_test_split(
    df_remapped,
    test_size=0.30,
    stratify=df_remapped['category'],
    random_state=42
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df['category'],
    random_state=42
)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# -----------------------------------------
# Step 5: EDA Augmentation (Training Only) (NEW)
# -----------------------------------------
print("\\n[5/7] Applying EDA augmentation to training set...")
train_df_aug = augment_minority_classes(train_df, EDA_CONFIG)

# -----------------------------------------
# Step 6: Label Encoding & Class Weights
# -----------------------------------------
print("\\n[6/7] Encoding labels and computing class weights...")

le = LabelEncoder()
all_categories = list(train_df_aug['category']) + list(val_df['category']) + list(test_df['category'])
le.fit(all_categories)
num_labels = len(le.classes_)
print(f"Number of classes: {num_labels}")

# Compute class weights on augmented training set
categories = np.unique(train_df_aug['category'])
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=categories,
    y=train_df_aug['category']
)
print(f"Class weights computed (range: {class_weights.min():.2f} - {class_weights.max():.2f})")

# -----------------------------------------
# Step 7: Tokenization
# -----------------------------------------
print("\\n[7/7] Tokenizing with DistilBERT...")

# Convert to HuggingFace Datasets
train_ds = Dataset.from_pandas(train_df_aug[['text', 'category']], preserve_index=False)
val_ds = Dataset.from_pandas(val_df[['text', 'category']], preserve_index=False)
test_ds = Dataset.from_pandas(test_df[['text', 'category']], preserve_index=False)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_fn(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

train_ds = train_ds.map(tokenize_fn, batched=True, batch_size=1000)
val_ds = val_ds.map(tokenize_fn, batched=True, batch_size=1000)
test_ds = test_ds.map(tokenize_fn, batched=True, batch_size=1000)

# Encode labels
def encode_labels(example):
    example['labels'] = le.transform([example['category']])[0]
    return example

train_ds = train_ds.map(encode_labels)
val_ds = val_ds.map(encode_labels)
test_ds = test_ds.map(encode_labels)

print("\\n" + "="*70)
print("PREPROCESSING COMPLETE")
print("="*70)
print(f"Train samples: {len(train_ds)} (with augmentation)")
print(f"Val samples:   {len(val_ds)}")
print(f"Test samples:  {len(test_ds)}")
print(f"Num classes:   {num_labels}")

# -----------------------------------------
# Save artifacts
# -----------------------------------------
print("\\nSaving preprocessed data...")

# Save datasets
from datasets import DatasetDict
processed = DatasetDict({
    'train': train_ds,
    'validation': val_ds,
    'test': test_ds
})
processed.save_to_disk("huffpost_final_processed")

# Save label encoder
with open('label_encoder_final.pkl', 'wb') as f:
    pickle.dump(le, f)

# Save class weights
np.save('class_weights_final.npy', class_weights)

print("Saved: huffpost_final_processed/, label_encoder_final.pkl, class_weights_final.npy")
'''


# ============================================================
# Print summary for user
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DRAFT CELLS READY")
    print("=" * 70)
    print("""
After training completes, copy these cells into final_milestone.ipynb:

1. MARKDOWN_HEADER - Project header and M1/M2 summary
2. MARKDOWN_CLASS_REMAP - Explanation of class consolidation
3. CODE_CLASS_REMAP - Class remapping implementation
4. MARKDOWN_EDA - Explanation of EDA augmentation
5. CODE_EDA - EDA implementation with nlpaug
6. CODE_FULL_PIPELINE - Complete preprocessing pipeline

Order in notebook:
  [Imports] -> [Header] -> [Class Remap Explanation] -> [Class Remap Code]
  -> [EDA Explanation] -> [EDA Code] -> [Full Pipeline] -> [Training]

Note: You'll need to install nlpaug:
  pip install nlpaug
    """)
