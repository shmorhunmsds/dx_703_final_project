# Your code here; add as many cells as you need but make it clear what the structure is.
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Load data from the preprocessed datasets (loaded in Cell 7)
# Extract raw text for baseline model (which uses TextVectorization, not DistilBERT)
X_train = np.array(train_ds['text'])
X_val = np.array(val_ds['text'])
X_test = np.array(test_ds['text'])

# Extract categories for labels
y_train = np.array(train_ds['category'])
y_val = np.array(val_ds['category'])
y_test = np.array(test_ds['category'])

# Binarize labels for categorical_crossentropy
lb = LabelBinarizer()
y_train_enc = lb.fit_transform(y_train)
y_val_enc = lb.transform(y_val)
y_test_enc = lb.transform(y_test)

num_classes = len(lb.classes_)
print("Number of classes:", num_classes)
print(f"Label binarizer classes match label encoder: {np.array_equal(lb.classes_, le.classes_)}")

# Vectorization (Note: This is separate from DistilBERT tokenization)
# For baseline, we're using TensorFlow's TextVectorization
max_vocab = 30000
max_len = 128
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=max_vocab,
    output_sequence_length=max_len,
    standardize='lower_and_strip_punctuation'
)
vectorizer.adapt(X_train)

# Baseline Model Architecture
embedding_dim = 128

inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
x = vectorizer(inputs)
x = tf.keras.layers.Embedding(input_dim=max_vocab, output_dim=embedding_dim)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

baseline_model = tf.keras.Model(inputs, outputs)
baseline_model.summary()

# Compile model
baseline_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train model
start_time = time.time()
history = baseline_model.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=20,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

print(f"Training completed in {round(time.time() - start_time, 2)} seconds.")
