import os

# Dataset Paths
DATASET_PATH = "dataset"
METADATA_FILE = os.path.join(DATASET_PATH, "metadata.csv")
FACES_FOLDER = os.path.join(DATASET_PATH, "faces_224")

# Model Storage
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fakevision_detection.keras")  # Updated to .keras format

# Training Parameters
IMG_SIZE = (128, 128, 3)  # Increased size for better model accuracy
BATCH_SIZE = 16        # Optimized for balanced memory usage & performance
EPOCHS = 10             # Avoids overfitting with limited data

# Ensure required directories exist
for directory in [MODEL_DIR, FACES_FOLDER]:
    os.makedirs(directory, exist_ok=True)

print("Configuration loaded successfully.")
