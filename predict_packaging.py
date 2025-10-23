# filename: predict_packaging.py

import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Configuration ---
PACKING_PADDING_INCHES = 0.5 
MODEL_FILE = 'product_dimension_model.keras'
INPUT_SIZE = (224, 224) # Must match the training script


def load_and_preprocess_image(image_path, target_size=INPUT_SIZE):
    """Loads and prepares a single image for prediction."""
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    
    # ** BUG FIX: DO NOT use img_array / 255.0 here **
    
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Apply the same preprocessing as we did in training
    img_array = preprocess_input(img_array)
    
    return img_array

def run_prediction(image_path, model):
    """Predicts dimensions and calculates the optimal custom box size."""
    print(f"Analyzing image: {image_path}\n")
    
    # 1. Load and preprocess the image
    processed_img = load_and_preprocess_image(image_path)
    
    # 2. Predict dimensions
    predicted_dims = model.predict(processed_img)[0]
    # The model outputs 3 values, but they are not in a guaranteed order.
    # To be robust, we sort them from smallest to largest.
    dims_sorted = sorted(predicted_dims)
    dim_1, dim_2, dim_3 = dims_sorted[0], dims_sorted[1], dims_sorted[2]
    
    print(f"--- AI Model Prediction (Sorted) ---")
    print(f"Predicted Dimension 1: {dim_1:.2f} inches")
    print(f"Predicted Dimension 2: {dim_2:.2f} inches")
    print(f"Predicted Dimension 3: {dim_3:.2f} inches\n")
    
    # 3. Calculate Optimal Box Size
    optimal_dim_1 = dim_1 + PACKING_PADDING_INCHES
    optimal_dim_2 = dim_2 + PACKING_PADDING_INCHES
    optimal_dim_3 = dim_3 + PACKING_PADDING_INCHES
    
    print(f"--- Optimal Custom Box Recommendation ---")
    print(f"Based on a {PACKING_PADDING_INCHES}-inch padding on each side:")
    print(f"Optimal Box (Sorted Dims): ({optimal_dim_1:.2f}, {optimal_dim_2:.2f}, {optimal_dim_3:.2f})")
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_packaging.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    try:
        # 1. Load the trained model
        print(f"Loading trained model '{MODEL_FILE}'...")
        model = tf.keras.models.load_model(MODEL_FILE)
        print("Model loaded successfully.\n")
        
        # 2. Run the prediction
        run_prediction(image_path, model)
        
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{image_path}' or '{MODEL_FILE}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")