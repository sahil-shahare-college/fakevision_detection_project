from tensorflow.keras.models import load_model
from evaluate import evaluate_model
from predict import predict_image
from config import MODEL_PATH

# Load trained model
try:
    model = load_model(MODEL_PATH, compile=False)  # Load without compiling
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # ✅ Compile the model
    print(f"Model successfully loaded and compiled from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Evaluate Model Accuracy
if model:
    accuracy = evaluate_model(model)  # Call function from `evaluate.py`
    print(f"Model trained with {accuracy:.2f}% accuracy.")

# Test with an image  
test_image_path = "dataset/aaunkxgnup.jpg"
predict_image(test_image_path) # Call function from `predict.py`

