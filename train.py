import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from sklearn.model_selection import train_test_split
from data_loader import load_metadata
from preprocess import preprocess_dataset, data_generator
from config import MODEL_PATH, EPOCHS, BATCH_SIZE, IMG_SIZE
import os

# Enable GPU memory growth to prevent out-of-memory errors
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory configuration applied.")
        except RuntimeError as e:
            print(f"⚠ GPU Configuration Error: {e}")

configure_gpu()

# Load dataset and split into training, validation, and test sets
meta = load_metadata()

# Check if metadata loaded correctly
if meta is None or meta.empty:
    raise ValueError("Error: Metadata could not be loaded. Please check dataset and metadata.csv!")
else:
    print(f"Metadata Loaded: {meta.shape}")

# Split dataset
Train_set, Test_set = train_test_split(meta, test_size=0.2, stratify=meta['label'], random_state=42)
Train_set, Val_set = train_test_split(Train_set, test_size=0.3, stratify=Train_set['label'], random_state=42)

print(f"📊 Dataset Split: Train={len(Train_set)}, Val={len(Val_set)}, Test={len(Test_set)}")

# Use data generator for efficient memory usage
train_generator = data_generator(Train_set, batch_size=BATCH_SIZE) 
val_generator = data_generator(Val_set, batch_size=BATCH_SIZE)

# Define CNN Model (Optimized Architecture)
model = Sequential([
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  # Use IMG_SIZE from config

    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(256, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Binary classification (Fake/Real)
])

# Compile Model (Lower LR for stability)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# Implement Callbacks: Early Stopping & Reduce LR on Plateau
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, len(Train_set) // BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=max(1, len(Val_set) // BATCH_SIZE),
    callbacks=callbacks  # Apply callbacks
)

# Save the trained model in the recommended `.keras` format
NEW_MODEL_PATH = MODEL_PATH.replace(".h5", ".keras")

# Ensure the models directory exists
os.makedirs(os.path.dirname(NEW_MODEL_PATH), exist_ok=True)

model.save(NEW_MODEL_PATH)
print(f"✅ Model saved at {NEW_MODEL_PATH}")
