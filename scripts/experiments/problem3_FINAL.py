"""
Problem 3 - Custom Model (Final Version)
Copy this into your notebook cell for Problem 3.

This is a Bidirectional LSTM with dual pooling - a significant architectural
improvement over the baseline's simple GlobalAveragePooling approach.
"""

# =============================================================================
# PROBLEM 3 - CUSTOM MODEL IMPLEMENTATION
# =============================================================================

print("=" * 70)
print("PROBLEM 3 - CUSTOM MODEL (Bidirectional LSTM)")
print("=" * 70)

# Build Custom Model Architecture
inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name='text_input')
x = vectorizer(inputs)

# Embedding layer with L2 regularization (helps prevent overfitting)
x = tf.keras.layers.Embedding(
    input_dim=max_vocab,
    output_dim=embedding_dim,
    embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
    name='embedding'
)(x)

# KEY CHANGE 1: Bidirectional LSTM instead of simple pooling
# This captures context from both directions in the text
x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(
        128,
        return_sequences=True,  # Return all timesteps for pooling
        dropout=0.2,
        recurrent_dropout=0.2
    ),
    name='bilstm'
)(x)

# KEY CHANGE 2: Dual pooling (max + average) instead of just average
# Captures both the most important features and general patterns
max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Concatenate()([max_pool, avg_pool])

# KEY CHANGE 3: Deeper dense layers with batch normalization
x = tf.keras.layers.Dense(
    256,
    activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(
    128,
    activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
)(x)
x = tf.keras.layers.Dropout(0.4)(x)

# Output layer
outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

# Create model
custom_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='BiLSTM_Custom')

# Display architecture
print("\nCustom Model Architecture:")
custom_model.summary()

# Compare with baseline
total_params = custom_model.count_params()
print(f"\nParameter Comparison:")
print(f"  Baseline:     3,861,285 parameters")
print(f"  Custom Model: {total_params:,} parameters")
print(f"  Increase:     {total_params - 3861285:,} (+{((total_params/3861285 - 1) * 100):.1f}%)")

# =============================================================================
# COMPILE MODEL
# =============================================================================

custom_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =============================================================================
# CALLBACKS
# =============================================================================

custom_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,  # More patience for complex model
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_custom_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# =============================================================================
# TRAIN MODEL
# =============================================================================

print("\n" + "=" * 70)
print("TRAINING CUSTOM MODEL")
print("=" * 70)

start_time = time.time()
custom_history = custom_model.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=100,
    batch_size=128,
    class_weight=dict(enumerate(class_weights)),
    callbacks=custom_callbacks,
    verbose=1
)
training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
print(f"Best epoch: {np.argmin(custom_history.history['val_loss']) + 1}")
print(f"Best val loss: {np.min(custom_history.history['val_loss']):.4f}")
print(f"Best val accuracy: {np.max(custom_history.history['val_accuracy']):.4f}")

# =============================================================================
# PLOT TRAINING HISTORY
# =============================================================================

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(custom_history.history['accuracy'], label='Train Acc', linewidth=2)
plt.plot(custom_history.history['val_accuracy'], label='Val Acc', linewidth=2)
plt.axvline(x=np.argmax(custom_history.history['val_accuracy']),
            color='r', linestyle='--', alpha=0.5, label='Best Epoch')
plt.title('Custom Model: Training vs Validation Accuracy', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(custom_history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(custom_history.history['val_loss'], label='Val Loss', linewidth=2)
plt.axvline(x=np.argmin(custom_history.history['val_loss']),
            color='r', linestyle='--', alpha=0.5, label='Best Epoch')
plt.title('Custom Model: Training vs Validation Loss', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# EVALUATE ON TEST SET
# =============================================================================

print("\n" + "=" * 70)
print("CUSTOM MODEL - TEST SET EVALUATION")
print("=" * 70)

test_loss, test_acc = custom_model.evaluate(X_test, y_test_enc, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions
y_pred_probs = custom_model.predict(X_test, verbose=0)
y_pred = le.classes_[y_pred_probs.argmax(axis=1)]

# Calculate metrics
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
macro_precision = precision_score(y_test, y_pred, average='macro')
macro_recall = recall_score(y_test, y_pred, average='macro')

print(f"\nOverall Metrics:")
print(f"  Macro F1:        {macro_f1:.4f}")
print(f"  Weighted F1:     {weighted_f1:.4f}")
print(f"  Macro Precision: {macro_precision:.4f}")
print(f"  Macro Recall:    {macro_recall:.4f}")

# Full classification report
print("\nPer-Class Performance:")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# =============================================================================
# COMPARISON WITH BASELINE
# =============================================================================

print("\n" + "=" * 70)
print("COMPARISON: BASELINE vs CUSTOM MODEL")
print("=" * 70)

# Baseline metrics (from your Cell 21 output)
baseline_test_acc = 0.6100
baseline_macro_f1 = 0.4549

custom_test_acc = test_acc
custom_macro_f1 = macro_f1

print(f"\n{'Metric':<30} {'Baseline':<15} {'Custom':<15} {'Improvement':<15}")
print("-" * 75)
print(f"{'Test Accuracy':<30} {baseline_test_acc:<15.4f} {custom_test_acc:<15.4f} {custom_test_acc - baseline_test_acc:+.4f}")
print(f"{'Macro F1':<30} {baseline_macro_f1:<15.4f} {custom_macro_f1:<15.4f} {custom_macro_f1 - baseline_macro_f1:+.4f}")
print(f"{'Weighted F1':<30} {'0.5853':<15} {weighted_f1:<15.4f} {weighted_f1 - 0.5853:+.4f}")

improvement_pct = ((custom_test_acc / baseline_test_acc) - 1) * 100
print(f"\nAccuracy improvement: {improvement_pct:+.2f}%")

print("\n" + "=" * 70)
