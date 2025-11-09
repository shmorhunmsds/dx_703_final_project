"""
Experiment: Does increasing vocab size help?
Quick test to see if vocab size is bottlenecking performance
"""

import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import f1_score

def quick_vocab_test(X_train, y_train_enc, X_val, y_val_enc,
                     X_test, y_test_enc, y_test, le, class_weights,
                     vocab_sizes=[10000, 30000, 50000, 75000]):
    """
    Quickly test different vocab sizes with a simple baseline model
    to see if vocab size matters for your dataset
    """

    results = []

    for max_vocab in vocab_sizes:
        print(f"\n{'='*80}")
        print(f"Testing max_vocab = {max_vocab:,}")
        print(f"{'='*80}")

        tf.keras.backend.clear_session()

        # Create vectorizer
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_vocab,
            output_sequence_length=128,
            standardize='lower_and_strip_punctuation'
        )
        vectorizer.adapt(X_train)

        # Vectorize data
        X_train_vec = vectorizer(X_train).numpy()
        X_val_vec = vectorizer(X_val).numpy()
        X_test_vec = vectorizer(X_test).numpy()

        # Simple CNN model
        inputs = tf.keras.Input(shape=(128,), dtype=tf.int32)
        x = tf.keras.layers.Embedding(max_vocab, 128)(inputs)
        x = tf.keras.layers.SpatialDropout1D(0.2)(x)

        # Three parallel convs
        conv3 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        conv3 = tf.keras.layers.GlobalMaxPooling1D()(conv3)

        conv4 = tf.keras.layers.Conv1D(128, 4, activation='relu', padding='same')(x)
        conv4 = tf.keras.layers.GlobalMaxPooling1D()(conv4)

        conv5 = tf.keras.layers.Conv1D(128, 5, activation='relu', padding='same')(x)
        conv5 = tf.keras.layers.GlobalMaxPooling1D()(conv5)

        x = tf.keras.layers.Concatenate()([conv3, conv4, conv5])
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(37, activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs)

        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train quickly (just 10 epochs for testing)
        start = time.time()
        history = model.fit(
            X_train_vec, y_train_enc,
            validation_data=(X_val_vec, y_val_enc),
            epochs=10,
            batch_size=256,
            class_weight=dict(enumerate(class_weights)),
            verbose=0
        )
        train_time = time.time() - start

        # Evaluate
        test_loss, test_acc = model.evaluate(X_test_vec, y_test_enc,
                                             batch_size=512, verbose=0)
        y_pred = le.classes_[model.predict(X_test_vec, batch_size=512,
                                           verbose=0).argmax(axis=1)]
        macro_f1 = f1_score(y_test, y_pred, average='macro')

        # Get best val acc from history
        best_val_acc = max(history.history['val_accuracy'])

        print(f"  Test Acc: {test_acc:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Best Val Acc: {best_val_acc:.4f}")
        print(f"  Train time: {train_time:.1f}s")
        print(f"  Params: {model.count_params():,}")

        results.append({
            'vocab_size': max_vocab,
            'test_acc': test_acc,
            'macro_f1': macro_f1,
            'best_val_acc': best_val_acc,
            'train_time': train_time,
            'params': model.count_params()
        })

    # Summary
    print(f"\n{'='*80}")
    print("VOCAB SIZE COMPARISON (10 epochs each)")
    print(f"{'='*80}")
    print(f"\n{'Vocab Size':<12} {'Test Acc':<12} {'Macro F1':<12} {'Params':<15} {'Time (s)'}")
    print("-" * 70)

    for r in results:
        print(f"{r['vocab_size']:<12,} {r['test_acc']:<12.4f} {r['macro_f1']:<12.4f} "
              f"{r['params']:<15,} {r['train_time']:<.1f}")

    # Find best
    best_acc = max(results, key=lambda x: x['test_acc'])
    best_f1 = max(results, key=lambda x: x['macro_f1'])

    print(f"\n{'='*80}")
    print(f"Best test accuracy: {best_acc['vocab_size']:,} vocab ({best_acc['test_acc']:.4f})")
    print(f"Best macro F1: {best_f1['vocab_size']:,} vocab ({best_f1['macro_f1']:.4f})")

    # Recommendation
    acc_range = max(r['test_acc'] for r in results) - min(r['test_acc'] for r in results)

    if acc_range < 0.01:  # Less than 1% difference
        print(f"\n💡 RECOMMENDATION: Vocab size has MINIMAL impact (<{acc_range*100:.1f}% difference)")
        print(f"   → Stick with 30,000 for efficiency")
        print(f"   → Focus on architecture/regularization instead")
    elif best_acc['vocab_size'] > 30000:
        print(f"\n💡 RECOMMENDATION: Larger vocab helps!")
        print(f"   → Try max_vocab = {best_acc['vocab_size']:,}")
    else:
        print(f"\n💡 RECOMMENDATION: Smaller vocab is sufficient")
        print(f"   → Can use max_vocab = {best_acc['vocab_size']:,}")

    print(f"{'='*80}\n")

    return results

# Usage in notebook:
"""
from vocab_experiment import quick_vocab_test

results = quick_vocab_test(
    X_train, y_train_enc,
    X_val, y_val_enc,
    X_test, y_test_enc,
    y_test, le, class_weights,
    vocab_sizes=[10000, 30000, 50000, 75000]
)
"""
