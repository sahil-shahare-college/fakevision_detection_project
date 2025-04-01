import os
import cv2
import numpy as np
from config import FACES_FOLDER, IMG_SIZE, BATCH_SIZE, DATASET_PATH

def data_generator(df, batch_size=BATCH_SIZE):
    """Generates batches of images and labels for training in real-time."""
    if df is None or df.empty:
        raise ValueError("❌ Error: DataFrame is empty! Check dataset and metadata.csv.")

    while True:
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_df = df.iloc[start:end]

            images, labels = [], []
            for img_name, label in zip(batch_df['videoname'], batch_df['label']):  
                img = load_and_preprocess_image(img_name)
                if img is not None:
                    images.append(img)
                    labels.append(1 if label == "FAKE" else 0)

            if images:  # Ensure at least one image is loaded before yielding
                yield np.array(images, dtype="float32"), np.array(labels, dtype="float32")
            else:
                print(f"⚠ Warning: No valid images in batch starting at index {start}. Skipping...")


def load_and_preprocess_image(img_name):
    """Loads, resizes, and normalizes an image."""
    img_path = os.path.join(FACES_FOLDER, f"{img_name[:-4]}.jpg")

    if not os.path.exists(img_path):
        print(f"⚠ Warning: Missing image {img_path}")
        return None

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Error: Could not load image {img_path}")
        return None

    # ✅ Fix: Ensure `cv2.resize()` receives correct format (width, height)
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0])).astype(np.float32) / 255.0  # Normalize

    return img


def preprocess_dataset(df):
    """Prepares images and labels from dataset DataFrame."""
    if df is None or df.empty:
        raise ValueError("❌ Error: DataFrame is empty! Check dataset and metadata.csv.")

    images, labels = [], []
    missing_count = 0

    # ✅ Automatically detect correct column name
    file_col = 'filename' if 'filename' in df.columns else 'videoname' if 'videoname' in df.columns else None
    if not file_col:
        raise KeyError("❌ Error: Expected column ('filename' or 'videoname') not found in metadata.csv!")

    for img_name, label in zip(df[file_col], df['label']):
        img = load_and_preprocess_image(img_name)
        if img is not None:
            images.append(img)
            labels.append(1 if label == 'FAKE' else 0)
        else:
            missing_count += 1

    if not images:  # Ensure dataset is not empty
        raise ValueError("❌ Error: No images were loaded. Check dataset paths!")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    print(f"✅ Dataset Loaded: {len(images)} images (Missing: {missing_count})")
    print(f"📊 Final Shape of Dataset: {images.shape}")  # Should be (num_samples, IMG_SIZE[0], IMG_SIZE[1], 3)

    return images, labels
