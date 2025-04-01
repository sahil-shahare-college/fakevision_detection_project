import gc
import tensorflow as tf

gc.collect()  # Explicitly runs Python's garbage collector to free memory
tf.keras.backend.clear_session()  # Clears the TensorFlow backend session to release GPU/CPU memory
