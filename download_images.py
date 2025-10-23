# filename: download_images.py

import pandas as pd
import requests
import os
import time

INPUT_FILE = 'model_training_data.csv'
OUTPUT_FILE = 'final_dataset_with_paths.csv'
IMAGE_DIR = 'images'

# Create the 'images' directory if it doesn't exist
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
    print(f"Created directory: '{IMAGE_DIR}'")

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"FATAL ERROR: File not found: '{INPUT_FILE}'")
    print("Please run prepare_data.py first.")
    exit()

print(f"Found {len(df)} images to download...")

# We will add a new column for the local image path
local_image_paths = []
download_count = 0

# Set up a session for downloading
session = requests.Session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

for index, row in df.iterrows():
    image_url = row['image_url']
    
    # Create a unique, simple filename for each image
    # Example: 'image_0001.jpg', 'image_0002.jpg', etc.
    file_extension = os.path.splitext(image_url)[-1]
    if '?' in file_extension: # Clean up extensions like .jpg?_encoding=...
        file_extension = file_extension.split('?')[0]
    if not file_extension:
        file_extension = '.jpg' # Default to .jpg
        
    filename = f"image_{index:04d}{file_extension}"
    local_path = os.path.join(IMAGE_DIR, filename)
    
    # --- Download the image ---
    try:
        response = session.get(image_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            local_image_paths.append(local_path)
            download_count += 1
            print(f"Success ({download_count}/{len(df)}): Downloaded {filename}")
        else:
            print(f"Failed to download: {image_url} (Status code: {response.status_code})")
            local_image_paths.append(None)
            
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")
        local_image_paths.append(None)
    
    # Be polite to the server
    time.sleep(0.1) 

print(f"\nSuccessfully downloaded {download_count} out of {len(df)} images.")

# Add the new 'image_path' column to our dataframe
df['image_path'] = local_image_paths

# Drop any rows where the download failed
df.dropna(subset=['image_path'], inplace=True)

# Save the final dataset
df.to_csv(OUTPUT_FILE, index=False)
print(f"Final dataset with local paths saved to '{OUTPUT_FILE}'.")