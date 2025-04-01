import os
import pandas as pd
import zipfile
from config import METADATA_FILE, DATASET_PATH

def extract_dataset(zip_path="archive.zip"):
    """Extracts dataset if not already extracted."""
    
    try:
        # Ensure dataset directory exists
        os.makedirs(DATASET_PATH, exist_ok=True)

        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATASET_PATH)
            print("Dataset extraction complete.")
            extracted_files = os.listdir(DATASET_PATH)
            print(f"Extracted Files: {extracted_files}" if extracted_files else "⚠ No files found in dataset!")
        else:
            print("Warning: Dataset ZIP file not found!")

    except zipfile.BadZipFile:
        print("Error: The provided ZIP file is corrupted or invalid.")
    except Exception as e:
        print(f"Error extracting dataset: {e}")

def load_metadata():
    """Loads metadata CSV and returns a DataFrame."""
    
    try:
        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError("Metadata file not found! Extract dataset first.")
        
        print("Loading metadata...")
        df = pd.read_csv(METADATA_FILE)
        
        if df.empty:
            raise ValueError("Metadata file is empty!")
        
        return df
    
    except pd.errors.EmptyDataError:
        print("Error: Metadata file is empty or corrupt.")
        return None
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

if __name__ == "__main__":
    extract_dataset()
    
    metadata = load_metadata()
    if metadata is not None:
        print(metadata.head())  # Display the first few rows of metadata
