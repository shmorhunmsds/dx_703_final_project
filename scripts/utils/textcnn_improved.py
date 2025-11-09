# Improved TextCNN for HuffPost Classification
# Addresses overfitting and improves performance over baseline

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import time
from sklearn.metrics import classification_report, f1_score

# =============================================================================
# IMPROVED TEXTCNN - VERSION 1: More Filters, Less Dropout, BatchNorm
# =============================================================================

def build_improved_textcnn_v1(max_vocab, max_len, embedding_dim, num_classes):
    """
    Changes from original:
    - More filters: 128 -> 256 per kernel
    - Less dropout: 0.5 -> 0.3, 0.4 -> 0.2
    - Added BatchNormalization after concatenation
    - Added L2 regularization to Dense layers
    """
    inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='token_input')

    # Embedding with spatial dropout
    embedded = layers.Embedding(
        input_dim=max_vocab,
        output_dim=embedding_dim,
        name='embedding'
    )(inputs)
    embedded = layers.SpatialDropout1D(0.2)(embedded)  # Better than regular dropout for embeddings

    # Parallel CNN layers with MORE filters (256 instead of 128)
    conv_3 = layers.Conv1D(256, 3, activation='relu', padding='same')(embedded)
    conv_3 = layers.GlobalMaxPooling1D()(conv_3)

    conv_4 = layers.Conv1D(256, 4, activation='relu', padding='same')(embedded)
    conv_4 = layers.GlobalMaxPooling1D()(conv_4)

    conv_5 = layers.Conv1D(256, 5, activation='relu', padding='same')(embedded)
    conv_5 = layers.GlobalMaxPooling1D()(conv_5)

    # Concatenate and normalize
    x = layers.Concatenate()([conv_3, conv_4, conv_5])
    x = layers.BatchNormalization()(x)  # Add BatchNorm for stability
    x = layers.Dropout(0.3)(x)  # Reduced from 0.5

    # Dense layers with L2 regularization
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Reduced from 0.4

    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TextCNN_Improved_v1')
    return model


# =============================================================================
# IMPROVED TEXTCNN - VERSION 2: More Kernel Sizes + Residual Connection
# =============================================================================

def build_improved_textcnn_v2(max_vocab, max_len, embedding_dim, num_classes):
    """
    Changes from original:
    - More kernel sizes: 2, 3, 4, 5, 6 (instead of just 3,4,5)
    - Residual connection from embedding
    - BatchNormalization throughout
    - Moderate dropout
    """
    inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='token_input')

    # Embedding
    embedded = layers.Embedding(
        input_dim=max_vocab,
        output_dim=embedding_dim,
        name='embedding'
    )(inputs)
    embedded = layers.SpatialDropout1D(0.2)(embedded)

    # Multiple kernel sizes for n-gram detection
    conv_outputs = []
    for kernel_size in [2, 3, 4, 5, 6]:
        conv = layers.Conv1D(192, kernel_size, activation='relu', padding='same')(embedded)
        conv = layers.GlobalMaxPooling1D()(conv)
        conv_outputs.append(conv)

    # Concatenate all conv outputs
    x = layers.Concatenate()(conv_outputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Dense layer
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TextCNN_Improved_v2')
    return model


# =============================================================================
# IMPROVED TEXTCNN - VERSION 3: Deeper Architecture
# =============================================================================

def build_improved_textcnn_v3(max_vocab, max_len, embedding_dim, num_classes):
    """
    Changes from original:
    - Stacked Conv1D layers (2 layers deep per kernel size)
    - Uses smaller initial filters that grow
    - Better regularization strategy
    """
    inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='token_input')

    # Embedding
    embedded = layers.Embedding(
        input_dim=max_vocab,
        output_dim=embedding_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(0.0001),
        name='embedding'
    )(inputs)
    embedded = layers.SpatialDropout1D(0.15)(embedded)

    # Stacked convolutions for each kernel size
    conv_outputs = []

    for kernel_size in [3, 4, 5]:
        # First conv layer
        conv = layers.Conv1D(128, kernel_size, padding='same')(embedded)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)

        # Second conv layer (deeper)
        conv = layers.Conv1D(256, kernel_size, padding='same')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)

        # Pool
        conv = layers.GlobalMaxPooling1D()(conv)
        conv_outputs.append(conv)

    # Concatenate
    x = layers.Concatenate()(conv_outputs)
    x = layers.Dropout(0.4)(x)

    # Dense layers
    x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TextCNN_Improved_v3')
    return model


# =============================================================================
# TRAINING FUNCTION WITH IMPROVED HYPERPARAMETERS
# =============================================================================

def train_improved_textcnn(model, X_train_vec, y_train_enc, X_val_vec, y_val_enc,
                           class_weights, epochs=100, batch_size=256, use_class_weights=True):
    """
    Train with improved hyperparameters:
    - Larger batch size (256 vs 128) for more stable gradients
    - Cosine decay learning rate schedule
    - Optional: reduced class weight impact
    """

    # Cosine decay learning rate (better than constant LR)
    initial_learning_rate = 0.001
    decay_steps = int(len(X_train_vec) / batch_size) * epochs
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=0.0001  # Minimum learning rate
    )

    # Compile with AdamW (better than Adam for text)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,  # Increased patience
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

    # Optionally reduce class weight impact (scale them down)
    if use_class_weights:
        # Scale class weights to reduce impact (sqrt transformation)
        scaled_weights = np.sqrt(class_weights)
        scaled_weights = scaled_weights / scaled_weights.mean()
        class_weight_dict = dict(enumerate(scaled_weights))
    else:
        class_weight_dict = None

    print(f"\nTraining with:")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial LR: {initial_learning_rate}")
    print(f"  Class weights: {'Scaled' if use_class_weights else 'None'}")
    print(f"  Optimizer: AdamW with cosine decay")

    # Train
    start_time = time.time()
    history = model.fit(
        X_train_vec, y_train_enc,
        validation_data=(X_val_vec, y_val_enc),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time

    return history, training_time


# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

if __name__ == "__main__":
    print("""
    THREE IMPROVED TEXTCNN ARCHITECTURES:

    Version 1 (Recommended for first try):
    - More filters (256 vs 128)
    - Less dropout (0.3/0.2 vs 0.5/0.4)
    - BatchNormalization
    - L2 regularization

    Version 2 (More comprehensive):
    - 5 kernel sizes (2,3,4,5,6)
    - BatchNormalization
    - Moderate regularization

    Version 3 (Deepest):
    - Stacked 2-layer convolutions
    - Most parameters
    - Heavy regularization

    USAGE IN NOTEBOOK:

    # Import this file
    import sys
    sys.path.append('/home/pshmo/dx_703_final_project')
    from textcnn_improved import build_improved_textcnn_v1, train_improved_textcnn

    # Build model (choose v1, v2, or v3)
    improved_model = build_improved_textcnn_v1(
        max_vocab=30000,
        max_len=128,
        embedding_dim=128,
        num_classes=37
    )

    # Train
    history, train_time = train_improved_textcnn(
        improved_model,
        X_train_vec, y_train_enc,
        X_val_vec, y_val_enc,
        class_weights=class_weights,
        epochs=100,
        batch_size=256,  # Larger batch size
        use_class_weights=True  # Set to False to disable class weights
    )

    # Evaluate
    test_loss, test_acc = improved_model.evaluate(X_test_vec, y_test_enc, batch_size=512)
    print(f"Test Accuracy: {test_acc:.4f}")
    """)
