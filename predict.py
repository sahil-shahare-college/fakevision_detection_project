import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from config import MODEL_PATH, IMG_SIZE

def preprocess_image(img_path):
    """Loads and preprocesses an image for prediction."""
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found.")
        return None, None  

    try:
        img = image.load_img(img_path, target_size=IMG_SIZE[:2])  # Resize
        x = image.img_to_array(img) / 255.0  # Normalize
        x = np.expand_dims(x, axis=0)
        return img, x
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def predict_image(img_path):
    """Predict if an image is FAKE or REAL using a trained model."""
    
    # Ensure model exists before loading
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    try:
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Preprocess Image
    img, x = preprocess_image(img_path)
    if img is None or x is None:
        print("Skipping prediction due to image error.")
        return

    # Make prediction
    try:
        prediction = model.predict(x, verbose=0)[0][0]  # Suppress logs
        label = "FAKE" if prediction >= 0.5 else "REAL"
        confidence = max(prediction, 1 - prediction)  # Ensure correct confidence value
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # Display results
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{label} (Confidence: {confidence:.2f})")
    plt.show()

    print(f"Prediction: {label} (Confidence: {confidence:.2f})")
