"""
Custom Model Architectures for Problem 3 - Milestone 2

These architectures go beyond the simple baseline and incorporate:
- Bidirectional LSTMs
- GRU layers
- CNN layers for text
- Deeper dense networks
- Better regularization

Expected performance: 63-68% accuracy (improvement over 61% baseline)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.regularizers import l2

# =============================================================================
# Architecture 1: Bidirectional LSTM with Attention-like Pooling
# =============================================================================

def build_bilstm_model(max_vocab, embedding_dim, num_classes, max_len=128):
    """
    Bidirectional LSTM model with dual pooling strategy.

    Architecture:
    - Embedding layer
    - Bidirectional LSTM (returns sequences)
    - Dual pooling: GlobalMaxPooling + GlobalAveragePooling
    - Dense layers with dropout
    - Output layer

    Expected improvement: Better at capturing long-range dependencies in text
    """
    inputs = Input(shape=(1,), dtype=tf.string, name='text_input')

    # Vectorization (assume vectorizer is defined globally)
    # x = vectorizer(inputs)  # This will be added when using

    # For the function, assume x comes in as vectorized sequences
    # We'll show the full model build below

    return None  # See full implementation below


def build_custom_model_v1(vectorizer, max_vocab=30000, embedding_dim=128, num_classes=37):
    """
    Version 1: Bidirectional LSTM with Max + Average Pooling

    Key features:
    - Bidirectional LSTM to capture context from both directions
    - Dual pooling (max + average) to capture different features
    - L2 regularization to reduce overfitting
    - Higher dropout rates
    """
    inputs = Input(shape=(1,), dtype=tf.string, name='text_input')
    x = vectorizer(inputs)

    # Embedding with some L2 regularization
    x = layers.Embedding(
        input_dim=max_vocab,
        output_dim=embedding_dim,
        embeddings_regularizer=l2(1e-5),
        name='embedding'
    )(x)

    # Bidirectional LSTM layer (return sequences for pooling)
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name='bilstm_1'
    )(x)

    # Dual pooling strategy
    max_pool = layers.GlobalMaxPooling1D()(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    x = layers.Concatenate()([max_pool, avg_pool])

    # Dense layers with strong regularization
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='BiLSTM_DualPool')
    return model


# =============================================================================
# Architecture 2: Stacked Bidirectional LSTM (Deeper)
# =============================================================================

def build_custom_model_v2(vectorizer, max_vocab=30000, embedding_dim=128, num_classes=37):
    """
    Version 2: Stacked Bidirectional LSTMs

    Key features:
    - Two stacked BiLSTM layers for hierarchical feature learning
    - First layer returns sequences, second returns final state
    - Progressive dropout (less in early layers, more in later)
    """
    inputs = Input(shape=(1,), dtype=tf.string, name='text_input')
    x = vectorizer(inputs)

    # Embedding layer
    x = layers.Embedding(
        input_dim=max_vocab,
        output_dim=embedding_dim,
        name='embedding'
    )(x)

    # First BiLSTM layer (returns sequences)
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name='bilstm_1'
    )(x)
    x = layers.BatchNormalization()(x)

    # Second BiLSTM layer (returns final state)
    x = layers.Bidirectional(
        layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3),
        name='bilstm_2'
    )(x)
    x = layers.BatchNormalization()(x)

    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='Stacked_BiLSTM')
    return model


# =============================================================================
# Architecture 3: GRU-based Model (Faster alternative to LSTM)
# =============================================================================

def build_custom_model_v3(vectorizer, max_vocab=30000, embedding_dim=128, num_classes=37):
    """
    Version 3: Bidirectional GRU

    Key features:
    - GRU instead of LSTM (faster, fewer parameters)
    - Still bidirectional for context
    - Simpler architecture, may train faster
    """
    inputs = Input(shape=(1,), dtype=tf.string, name='text_input')
    x = vectorizer(inputs)

    # Embedding layer
    x = layers.Embedding(
        input_dim=max_vocab,
        output_dim=embedding_dim,
        name='embedding'
    )(x)

    # Bidirectional GRU (return sequences)
    x = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name='bigru_1'
    )(x)

    # Another GRU layer
    x = layers.Bidirectional(
        layers.GRU(64, dropout=0.3, recurrent_dropout=0.3),
        name='bigru_2'
    )(x)

    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='BiGRU')
    return model


# =============================================================================
# Architecture 4: CNN for Text Classification
# =============================================================================

def build_custom_model_v4(vectorizer, max_vocab=30000, embedding_dim=128, num_classes=37):
    """
    Version 4: CNN for Text Classification

    Key features:
    - Multiple parallel Conv1D layers with different kernel sizes
    - Captures different n-gram patterns (3, 4, 5 words)
    - Max pooling to get most important features
    - Faster than RNNs, good for text classification
    """
    inputs = Input(shape=(1,), dtype=tf.string, name='text_input')
    x = vectorizer(inputs)

    # Embedding layer
    embedded = layers.Embedding(
        input_dim=max_vocab,
        output_dim=embedding_dim,
        name='embedding'
    )(x)

    # Multiple parallel convolutional layers with different kernel sizes
    conv_blocks = []
    for kernel_size in [3, 4, 5]:
        conv = layers.Conv1D(
            filters=128,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            name=f'conv_{kernel_size}'
        )(embedded)
        conv = layers.GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)

    # Concatenate all conv outputs
    x = layers.Concatenate()(conv_blocks)
    x = layers.Dropout(0.5)(x)

    # Dense layers
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='TextCNN')
    return model


# =============================================================================
# Architecture 5: Hybrid CNN-LSTM
# =============================================================================

def build_custom_model_v5(vectorizer, max_vocab=30000, embedding_dim=128, num_classes=37):
    """
    Version 5: CNN + LSTM Hybrid

    Key features:
    - CNN to extract local features (n-grams)
    - LSTM to capture sequential dependencies
    - Combines strengths of both architectures
    """
    inputs = Input(shape=(1,), dtype=tf.string, name='text_input')
    x = vectorizer(inputs)

    # Embedding layer
    x = layers.Embedding(
        input_dim=max_vocab,
        output_dim=embedding_dim,
        name='embedding'
    )(x)

    # CNN layers to extract local features
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)

    # BiLSTM to capture sequential patterns
    x = layers.Bidirectional(
        layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3)
    )(x)

    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Hybrid')
    return model


# =============================================================================
# Training Configuration
# =============================================================================

def get_training_config():
    """
    Recommended training configuration for custom models.
    """
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    # Optimizer with lower learning rate
    optimizer = Adam(learning_rate=1e-3)

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_custom_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    return optimizer, callbacks


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    print("Custom Model Architectures for Problem 3")
    print("=" * 70)
    print("\nAvailable models:")
    print("1. BiLSTM with Dual Pooling (build_custom_model_v1)")
    print("2. Stacked BiLSTM (build_custom_model_v2)")
    print("3. BiGRU (build_custom_model_v3)")
    print("4. Text CNN (build_custom_model_v4)")
    print("5. CNN-LSTM Hybrid (build_custom_model_v5)")
    print("\nRecommendation: Start with v1 or v4 (fastest to train)")
    print("For best accuracy: Try v2 or v5")
