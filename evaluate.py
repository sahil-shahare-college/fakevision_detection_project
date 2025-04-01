from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_loader import load_metadata
from preprocess import data_generator  # Use generator instead of full dataset loading
from config import BATCH_SIZE

def evaluate_model(model):
    """Evaluates model performance on the test dataset using data generator."""
    meta = load_metadata()
    _, Test_set = train_test_split(meta, test_size=0.2, stratify=meta['label'], random_state=42)

    test_generator = data_generator(Test_set, batch_size=BATCH_SIZE)  # Use generator
    steps = len(Test_set) // BATCH_SIZE

    loss, accuracy = model.evaluate(test_generator, steps=steps)
    return accuracy * 100  # Return accuracy in percentage
