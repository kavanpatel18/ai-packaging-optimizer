# filename: predict_packaging.py

import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.preprocessing import image

# --- 1. Define Our Standard Box Sizes ---
# This is the updated list including the flat mailer
STANDARD_BOXES = [
    (10, 8, 1),  # Flat mailer
    (6, 4, 4),
    (8, 6, 4),
    (10, 8, 6),
    (12, 12, 8),
    (18, 18, 16),
    (24, 18, 18)
]

def find_optimal_box(product_dims, available_boxes):
    """
    Finds the smallest box a product can fit in, allowing for rotation.
    """
    # Sort product dimensions from largest to smallest
    p_dims = sorted(product_dims, reverse=True)
    
    best_box = None
    min_volume = float('inf')

    for box in available_boxes:
        # Sort box dimensions from largest to smallest
        b_dims = sorted(box, reverse=True)
        
        # Check if the product fits (largest dim fits largest box dim, etc.)
        if (p_dims[0] <= b_dims[0] and
            p_dims[1] <= b_dims[1] and
            p_dims[2] <= b_dims[2]):
            
            box_volume = b_dims[0] * b_dims[1] * b_dims[2]
            
            # If it fits and is smaller than the current best box
            if box_volume < min_volume:
                min_volume = box_volume
                best_box = box
                
    return best_box

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Loads and prepares a single image for prediction."""
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

def run_prediction(image_path, model):
    """Predicts dimensions and finds the optimal box for a single image."""
    print(f"Analyzing image: {image_path}\n")
    
    # 1. Load and preprocess the image
    processed_img = load_and_preprocess_image(image_path)
    
    # 2. Predict dimensions
    predicted_dims = model.predict(processed_img)[0]
    # The model outputs dimensions as [length, width, height]
    length, width, height = predicted_dims[0], predicted_dims[1], predicted_dims[2]
    
    print(f"--- AI Model Prediction ---")
    print(f"Predicted Length: {length:.2f} inches")
    print(f"Predicted Width:  {width:.2f} inches")
    print(f"Predicted Height: {height:.2f} inches\n")
    
    # 3. Find the optimal box
    product_volume = length * width * height
    optimal_box = find_optimal_box(predicted_dims, STANDARD_BOXES)
    
    print(f"--- Packaging Recommendation ---")
    if optimal_box:
        box_volume = optimal_box[0] * optimal_box[1] * optimal_box[2]
        wasted_space = (1 - product_volume / box_volume) * 100
        
        print(f"Recommended Box (L, W, H): {optimal_box}")
        print(f"Estimated Wasted Space: {wasted_space:.2f}%")
    else:
        print("No suitable box found in STANDARD_BOXES for the predicted dimensions.")
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_packaging.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    try:
        # 1. Load the trained model
        print("Loading trained model 'product_dimension_model.h5'...")
        model = tf.keras.models.load_model('product_dimension_model.h5')
        print("Model loaded successfully.\n")
        
        # 2. Run the prediction
        run_prediction(image_path, model)
        
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{image_path}' or 'product_dimension_model.h5' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")