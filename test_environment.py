#!/usr/bin/env python3
"""
Environment Test Script
Tests: Data loading, GPU availability, and model compatibility
"""

import sys
import os
import numpy as np
import pickle

print("=" * 70)
print("ENVIRONMENT VERIFICATION SCRIPT")
print("=" * 70)

# =============================================================================
# TEST 1: Data Loading
# =============================================================================
print("\n[TEST 1] Loading preprocessed data...")
try:
    from datasets import load_from_disk

    processed_datasets = load_from_disk("huffpost_processed_milestone2")
    train_ds = processed_datasets['train']
    val_ds = processed_datasets['validation']
    test_ds = processed_datasets['test']

    print(f"✓ Train: {len(train_ds)} samples")
    print(f"✓ Val:   {len(val_ds)} samples")
    print(f"✓ Test:  {len(test_ds)} samples")

    # Load label encoder
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    print(f"✓ Label encoder loaded: {len(le.classes_)} classes")

    # Load class weights
    class_weights = np.load('class_weights.npy')
    print(f"✓ Class weights loaded: {len(class_weights)} weights")

    # Verify alignment
    assert len(le.classes_) == len(class_weights), "Class count mismatch!"
    print("✓ Data and weights are aligned")

    DATA_OK = True
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    DATA_OK = False
    sys.exit(1)

# =============================================================================
# TEST 2: TensorFlow and GPU
# =============================================================================
print("\n[TEST 2] TensorFlow and GPU availability...")
try:
    import tensorflow as tf

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU devices found: {len(gpus)}")

    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")

        # Try to use GPU
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                c = tf.matmul(a, b)
                _ = c.numpy()
            print("✓ GPU computation successful")
            GPU_OK = True
        except Exception as e:
            print(f"✗ GPU computation failed: {e}")
            GPU_OK = False
    else:
        print("⚠️  No GPU detected - will use CPU")
        GPU_OK = False

    TF_OK = True
except Exception as e:
    print(f"✗ TensorFlow test failed: {e}")
    TF_OK = False
    GPU_OK = False

# =============================================================================
# TEST 3: Build and Test Simple Model
# =============================================================================
print("\n[TEST 3] Building test model...")
if TF_OK and DATA_OK:
    try:
        from sklearn.preprocessing import LabelBinarizer

        # Prepare small sample of data
        X_train_sample = np.array(train_ds['text'][:1000], dtype=object)
        y_train_sample = np.array(train_ds['category'][:1000])

        X_val_sample = np.array(val_ds['text'][:200], dtype=object)
        y_val_sample = np.array(val_ds['category'][:200])

        # Binarize labels
        lb = LabelBinarizer()
        y_train_enc = lb.fit_transform(y_train_sample)
        y_val_enc = lb.transform(y_val_sample)

        num_classes = len(lb.classes_)

        # Build simple model
        max_vocab = 10000
        max_len = 128

        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_vocab,
            output_sequence_length=max_len,
            standardize='lower_and_strip_punctuation'
        )
        vectorizer.adapt(X_train_sample)

        # Build model
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
        x = vectorizer(inputs)
        x = tf.keras.layers.Embedding(input_dim=max_vocab, output_dim=64)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        test_model = tf.keras.Model(inputs, outputs)
        test_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("✓ Model built successfully")

        # Try training 1 epoch
        print("\n[TEST 4] Testing training (1 epoch)...")
        import time
        start = time.time()

        history = test_model.fit(
            X_train_sample, y_train_enc,
            validation_data=(X_val_sample, y_val_enc),
            epochs=1,
            batch_size=32,
            verbose=0
        )

        elapsed = time.time() - start

        train_acc = history.history['accuracy'][0]
        val_acc = history.history['val_accuracy'][0]

        print(f"✓ Training successful!")
        print(f"  Time: {elapsed:.2f} seconds")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Val accuracy: {val_acc:.4f}")

        # Estimate full training time
        samples_per_sec = 1000 / elapsed
        full_samples = len(train_ds)
        estimated_time_per_epoch = full_samples / samples_per_sec

        print(f"\n  Performance estimate:")
        print(f"    Samples/sec: {samples_per_sec:.0f}")
        print(f"    Time per epoch (full dataset): {estimated_time_per_epoch:.1f} sec")
        print(f"    Time for 20 epochs: {estimated_time_per_epoch * 20 / 60:.1f} min")

        MODEL_OK = True

    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        MODEL_OK = False
else:
    print("⊘ Skipping model test (TF or data not OK)")
    MODEL_OK = False

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Data loading:        {'✓ PASS' if DATA_OK else '✗ FAIL'}")
print(f"TensorFlow:          {'✓ PASS' if TF_OK else '✗ FAIL'}")
print(f"GPU available:       {'✓ YES' if GPU_OK else '⚠️  NO (CPU mode)'}")
print(f"Model works:         {'✓ PASS' if MODEL_OK else '✗ FAIL'}")

if DATA_OK and TF_OK and MODEL_OK:
    print("\n✓ READY TO TRAIN MODELS")
    if not GPU_OK:
        print("⚠️  Training will use CPU (slower but functional)")
    sys.exit(0)
else:
    print("\n✗ ENVIRONMENT NOT READY")
    sys.exit(1)
