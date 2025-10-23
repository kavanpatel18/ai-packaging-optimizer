# filename: extract_features.py

import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import hog
from tqdm import tqdm

INPUT_FILE = 'final_dataset_with_paths.csv'
OUTPUT_FILE = 'features.csv'

def extract_image_features(image_path):
    """
    Loads an image and extracts a feature vector using
    Color Histograms and Histogram of Oriented Gradients (HOG).
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # 1. Resize to a standard size
        img_resized = cv2.resize(img, (256, 256))
        
        # 2. Color Features (Mean Color in HSV space)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        mean_color = np.mean(hsv.reshape(-1, 3), axis=0)
        
        # 3. Texture/Shape Features (HOG)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray, orientations=8, pixels_per_cell=(32, 32),
                           cells_per_block=(1, 1), visualize=False, channel_axis=None)
        
        # Combine all features into one vector
        # (3 mean color features + 512 HOG features)
        features = np.concatenate([mean_color, hog_features])
        return features
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# --- Main Script ---
print(f"Loading clean dataset from '{INPUT_FILE}'...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"FATAL ERROR: '{INPUT_FILE}' not found. Please run prepare_data.py and download_images.py first.")
    exit()

print(f"Starting feature extraction for {len(df)} images...")

all_features = []
valid_indices = []

# Loop over all images with a progress bar
for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
    image_path = row['image_path']
    
    if not os.path.exists(image_path):
        continue
        
    features = extract_image_features(image_path)
    
    if features is not None:
        all_features.append(features)
        valid_indices.append(index) # Keep track of which rows were successful

print(f"\nSuccessfully extracted features for {len(all_features)} images.")

# Create a new DataFrame for the features
feature_df = pd.DataFrame(all_features)
# Add "feature_" prefix to all columns
feature_df.columns = [f'feature_{i}' for i in range(feature_df.shape[1])]

# Get the original dimension data for the successful images
df_clean = df.loc[valid_indices].reset_index(drop=True)

# Combine the features and the dimensions
final_df = pd.concat([df_clean[['image_path', 'product_length', 'product_width', 'product_height']], feature_df], axis=1)

# Save the final feature set
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"Feature dataset saved successfully to '{OUTPUT_FILE}'.")