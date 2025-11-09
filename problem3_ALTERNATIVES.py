"""
Problem 3 - Alternative Architectures

BiLSTM not working well? Try these proven alternatives for text classification.
All use pre-tokenized data for speed.

Pick ONE to try - I recommend starting with TextCNN (fastest and often best for this task)
"""

# =============================================================================
# OPTION 1: TEXT CNN (RECOMMENDED - Fast and Effective)
# =============================================================================

def build_textcnn_model():
    """
    Text CNN - Often BETTER than LSTM for classification tasks!

    Advantages:
    - Much faster to train (no sequential processing)
    - Better GPU utilization
    - Less prone to overfitting
    - Works great for news article classification
    """
    print("=" * 70)
    print("Building Text CNN Model")
    print("=" * 70)

    inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='token_input')

    # Embedding
    embedded = tf.keras.layers.Embedding(
        input_dim=max_vocab,
        output_dim=128,
        name='embedding'
    )(inputs)

    # Multiple parallel CNN layers with different kernel sizes
    # Captures 3-grams, 4-grams, and 5-grams
    conv_blocks = []
    for kernel_size in [3, 4, 5]:
        conv = tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            name=f'conv_{kernel_size}'
        )(embedded)
        conv = tf.keras.layers.GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)

    # Concatenate all conv outputs
    x = tf.keras.layers.Concatenate()(conv_blocks)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Dense layers
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TextCNN')
    return model


# =============================================================================
# OPTION 2: GRU (Simpler than LSTM, Sometimes Better)
# =============================================================================

def build_gru_model():
    """
    GRU - Simpler alternative to LSTM

    Advantages:
    - Fewer parameters than LSTM
    - Faster to train
    - Less prone to overfitting
    """
    print("=" * 70)
    print("Building GRU Model")
    print("=" * 70)

    inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='token_input')

    # Embedding
    x = tf.keras.layers.Embedding(
        input_dim=max_vocab,
        output_dim=128,
        name='embedding'
    )(inputs)

    # Bidirectional GRU
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(64, dropout=0.3, recurrent_dropout=0.3)
    )(x)

    # Dense
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='BiGRU')
    return model


# =============================================================================
# OPTION 3: SIMPLE BUT EFFECTIVE (Deeper Dense Network)
# =============================================================================

def build_deep_dense_model():
    """
    Deep Dense Network - Sometimes simplest is best!

    Advantages:
    - Very fast to train
    - Easy to tune
    - Good baseline
    """
    print("=" * 70)
    print("Building Deep Dense Model")
    print("=" * 70)

    inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='token_input')

    # Embedding
    x = tf.keras.layers.Embedding(
        input_dim=max_vocab,
        output_dim=128,
        name='embedding'
    )(inputs)

    # Multiple pooling strategies
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([max_pool, avg_pool])

    # Deeper dense network
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='DeepDense')
    return model


# =============================================================================
# READY-TO-RUN: TEXT CNN VERSION (RECOMMENDED)
# =============================================================================

print("=" * 70)
print("PROBLEM 3 - TEXT CNN MODEL (Alternative Architecture)")
print("=" * 70)
print("\nWhy CNN for text?")
print("- Captures local patterns (n-grams) effectively")
print("- Parallel processing (faster than sequential LSTM)")
print("- Less prone to overfitting on this size dataset")
print("- Often outperforms RNNs for news classification!")
print("=" * 70)

# Pre-tokenize if not already done
if 'X_train_vec' not in locals():
    print("\nPre-tokenizing data...")
    X_train_vec = vectorizer(X_train).numpy()
    X_val_vec = vectorizer(X_val).numpy()
    X_test_vec = vectorizer(X_test).numpy()
    print("Done!")

# Build TextCNN model
inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='token_input')

# Embedding
embedded = tf.keras.layers.Embedding(
    input_dim=max_vocab,
    output_dim=128,
    name='embedding'
)(inputs)

# Parallel CNN layers with different kernel sizes (3, 4, 5)
conv_3 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(embedded)
conv_3 = tf.keras.layers.GlobalMaxPooling1D()(conv_3)

conv_4 = tf.keras.layers.Conv1D(128, 4, activation='relu', padding='same')(embedded)
conv_4 = tf.keras.layers.GlobalMaxPooling1D()(conv_4)

conv_5 = tf.keras.layers.Conv1D(128, 5, activation='relu', padding='same')(embedded)
conv_5 = tf.keras.layers.GlobalMaxPooling1D()(conv_5)

# Concatenate all
x = tf.keras.layers.Concatenate()([conv_3, conv_4, conv_5])
x = tf.keras.layers.Dropout(0.5)(x)

# Dense layers
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)

# Output
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Create model
cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TextCNN')

print("\nTextCNN Architecture:")
cnn_model.summary()

# Compile
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
cnn_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_textcnn_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train
print("\n" + "=" * 70)
print("TRAINING TEXT CNN")
print("=" * 70)
print("Expected: MUCH faster than BiLSTM!")
print("=" * 70)

start_time = time.time()
cnn_history = cnn_model.fit(
    X_train_vec, y_train_enc,
    validation_data=(X_val_vec, y_val_enc),
    epochs=100,
    batch_size=256,
    class_weight=dict(enumerate(class_weights)),
    callbacks=cnn_callbacks,
    verbose=1
)
training_time = time.time() - start_time

print(f"\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
print(f"Best epoch: {np.argmin(cnn_history.history['val_loss']) + 1}")
print(f"Best val loss: {np.min(cnn_history.history['val_loss']):.4f}")
print(f"Best val accuracy: {np.max(cnn_history.history['val_accuracy']):.4f}")

# Plot
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'], label='Train Acc', linewidth=2)
plt.plot(cnn_history.history['val_accuracy'], label='Val Acc', linewidth=2)
plt.axvline(x=np.argmax(cnn_history.history['val_accuracy']),
            color='r', linestyle='--', alpha=0.5, label='Best Epoch')
plt.title('TextCNN: Training vs Validation Accuracy', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(cnn_history.history['val_loss'], label='Val Loss', linewidth=2)
plt.axvline(x=np.argmin(cnn_history.history['val_loss']),
            color='r', linestyle='--', alpha=0.5, label='Best Epoch')
plt.title('TextCNN: Training vs Validation Loss', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Evaluate
print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

test_loss, test_acc = cnn_model.evaluate(X_test_vec, y_test_enc, batch_size=512, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

y_pred_probs = cnn_model.predict(X_test_vec, batch_size=512, verbose=0)
y_pred = le.classes_[y_pred_probs.argmax(axis=1)]

macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nOverall Metrics:")
print(f"  Macro F1:    {macro_f1:.4f}")
print(f"  Weighted F1: {weighted_f1:.4f}")

print("\nPer-Class Performance:")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# Comparison
print("\n" + "=" * 70)
print("COMPARISON WITH BASELINE")
print("=" * 70)

baseline_test_acc = 0.6100
baseline_macro_f1 = 0.4549

print(f"\n{'Model':<25} {'Test Acc':<12} {'Macro F1':<12} {'Training Time':<15}")
print("-" * 75)
print(f"{'Baseline':<25} {baseline_test_acc:<12.4f} {baseline_macro_f1:<12.4f} {'~23 sec':<15}")
print(f"{'TextCNN':<25} {test_acc:<12.4f} {macro_f1:<12.4f} {f'{training_time:.0f} sec':<15}")

improvement_pct = ((test_acc / baseline_test_acc) - 1) * 100
print(f"\nAccuracy improvement: {improvement_pct:+.2f}%")

print("\n" + "=" * 70)
print("TextCNN Advantages:")
print("  ✓ Faster training (parallel processing)")
print("  ✓ Better GPU utilization")
print("  ✓ Less overfitting (fewer parameters)")
print("  ✓ Captures n-gram patterns effectively")
print("=" * 70)
