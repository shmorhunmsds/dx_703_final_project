# Enhanced Baseline Model with Callbacks and Class Weights
# This is still the same architecture as the baseline, but with better training configuration

import tensorflow as tf
from tensorflow.keras import layers, models, Input, Sequential
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, f1_score

# Helper functions
def plot_training_history(history):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def evaluate_model(model, X_test, y_test_enc, y_test, le):
    test_loss, test_acc = model.evaluate(X_test, y_test_enc)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = le.classes_[y_pred_probs.argmax(axis=1)]

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    report = classification_report(y_test, y_pred, target_names=le.classes_, digits=4)
    print(report)

# Enhanced callbacks
callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint('best_baseline_model.keras', monitor='val_loss', save_best_only=True)
]

# Rebuild baseline model with same architecture
# Note: Using Sequential API for clarity
enhanced_baseline = Sequential([
    layers.Input(shape=(1,), dtype=tf.string),
    vectorizer,
    layers.Embedding(input_dim=max_vocab, output_dim=embedding_dim),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
], name='enhanced_baseline')

# Model summary
enhanced_baseline.summary()

# Compile model with slightly lower learning rate
enhanced_baseline.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Create class weight dictionary
# class_weights array is already aligned with le.classes_ (verified in verify_pipeline.py)
class_weight_dict = dict(enumerate(class_weights))

print("\nClass weights (first 5):")
for i in range(5):
    print(f"  {i} ({le.classes_[i]}): {class_weights[i]:.4f}")

# Train with class weights and enhanced callbacks
start_time = time.time()
history = enhanced_baseline.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=100,  # More epochs, but early stopping will prevent overfitting
    batch_size=128,
    class_weight=class_weight_dict,
    callbacks=callbacks_list,
    verbose=1
)
training_time = time.time() - start_time

print(f"\nTraining completed in {round(training_time, 2)} seconds")
print(f"Best epoch: {np.argmin(history.history['val_loss']) + 1}")
print(f"Best val loss: {np.min(history.history['val_loss']):.4f}")
print(f"Best val accuracy: {np.max(history.history['val_accuracy']):.4f}")

# Plot training history
plot_training_history(history)

# Evaluate on test set
print("\n" + "="*70)
print("ENHANCED BASELINE MODEL - TEST SET EVALUATION")
print("="*70)
evaluate_model(enhanced_baseline, X_test, y_test_enc, y_test, le)
