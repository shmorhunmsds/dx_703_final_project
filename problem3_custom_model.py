"""
Problem 3 - Custom Model Implementation

Ready-to-run code for your Milestone 2 notebook.
This builds a Bidirectional LSTM model with dual pooling strategy.
"""

# =============================================================================
# CUSTOM MODEL - BIDIRECTIONAL LSTM WITH DUAL POOLING
# =============================================================================

print("=" * 70)
print("PROBLEM 3 - CUSTOM MODEL (Bidirectional LSTM)")
print("=" * 70)

# Build custom model architecture
inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name='text_input')
x = vectorizer(inputs)

# Embedding layer with L2 regularization
x = tf.keras.layers.Embedding(
    input_dim=max_vocab,
    output_dim=embedding_dim,
    embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
    name='embedding'
)(x)

# Bidirectional LSTM layer (returns sequences for dual pooling)
x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(
        128,
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2
    ),
    name='bilstm_1'
)(x)

# Dual pooling: capture both max and average features
max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Concatenate()([max_pool, avg_pool])

# Dense layers with batch normalization and dropout
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.4)(x)

# Output layer
outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

# Create model
custom_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='BiLSTM_Custom')

# Model summary
print("\nCustom Model Architecture:")
custom_model.summary()

# Count parameters
total_params = custom_model.count_params()
print(f"\nTotal parameters: {total_params:,}")
print(f"Baseline parameters: 3,861,285")
print(f"Increase: {total_params - 3861285:,} parameters (+{((total_params/3861285 - 1) * 100):.1f}%)")

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
        patience=8,
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
    epochs=100,  # Will stop early if no improvement
    batch_size=128,
    class_weight=dict(enumerate(class_weights)),  # Use class weights
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

# Evaluate
test_loss, test_acc = custom_model.evaluate(X_test, y_test_enc, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
y_pred_probs = custom_model.predict(X_test, verbose=0)
y_pred = le.classes_[y_pred_probs.argmax(axis=1)]

# Metrics
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

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

# Get baseline metrics (from history variable from Cell 19)
baseline_best_val_acc = np.max(history.history['val_accuracy'])
baseline_test_acc = 0.6100  # From your Cell 21 output

custom_best_val_acc = np.max(custom_history.history['val_accuracy'])
custom_test_acc = test_acc

print(f"\n{'Metric':<30} {'Baseline':<15} {'Custom':<15} {'Improvement':<15}")
print("-" * 75)
print(f"{'Best Val Accuracy':<30} {baseline_best_val_acc:<15.4f} {custom_best_val_acc:<15.4f} {custom_best_val_acc - baseline_best_val_acc:+.4f}")
print(f"{'Test Accuracy':<30} {baseline_test_acc:<15.4f} {custom_test_acc:<15.4f} {custom_test_acc - baseline_test_acc:+.4f}")
print(f"{'Training Time (s)':<30} {'~23':<15} {training_time:<15.1f} {training_time - 23:+.1f}")
print(f"{'Parameters':<30} {'3,861,285':<15} {f'{total_params:,}':<15} {f'+{total_params - 3861285:,}'}")

# Save model
print(f"\nBest model saved to: best_custom_model.keras")

print("\n" + "=" * 70)
