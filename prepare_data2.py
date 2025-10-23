# filename: prepare_data.py (Definitive Final Version)

import pandas as pd
import numpy as np

INPUT_FILE = 'final_dim_extract_data.csv'
OUTPUT_FILE = 'model_training_data.csv'
CM_TO_INCH = 1 / 2.54

# --- !! FINAL TWEAKS !! ---
MAX_DIMENSION_INCHES = 100  # 8.3 feet
MIN_DIMENSION_INCHES = 0.5   # Half an inch

print(f"Loading data from '{INPUT_FILE}'...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"FATAL ERROR: File not found: '{INPUT_FILE}'")
    exit()

print(f"Original dataset has {len(df)} rows.")

# --- 1. Standardize all units to INCHES ---
df['product_height'] = df['height inch'].fillna(df['height cm'] * CM_TO_INCH)
df['product_width'] = df['width inch'].fillna(df['width cm'] * CM_TO_INCH)

# --- 2. Create 'product_length' by coalescing l, d, and t ---
len_from_l = df['length inch'].fillna(df['length cm'] * CM_TO_INCH)
len_from_d = df['depth inch'].fillna(df['depth cm'] * CM_TO_INCH)
len_from_t = df['thickness inch'].fillna(df['thickness cm'] * CM_TO_INCH)
df['product_length'] = len_from_l.fillna(len_from_d).fillna(len_from_t)

# --- 3. Filter, rename, and save the final clean dataset ---
final_columns = ['product_length', 'product_width', 'product_height', 'Image link']
df_clean = df[final_columns].copy()
df_clean.rename(columns={'Image link': 'image_url'}, inplace=True)

# Drop any rows that *still* don't have all 3 dimensions
df_clean.dropna(subset=['product_length', 'product_width', 'product_height', 'image_url'], inplace=True)
print(f"Found {len(df_clean)} rows with 3 dimensions and an image link.")

# --- 4. !! FINAL TWEAK: Remove Extreme Outliers (Max AND Min) !! ---
print(f"Filtering items with any dimension > {MAX_DIMENSION_INCHES} or < {MIN_DIMENSION_INCHES} inches...")

initial_rows = len(df_clean)
df_clean = df_clean[
    (df_clean['product_length'] <= MAX_DIMENSION_INCHES) &
    (df_clean['product_width']  <= MAX_DIMENSION_INCHES) &
    (df_clean['product_height'] <= MAX_DIMENSION_INCHES) &
    (df_clean['product_length'] >= MIN_DIMENSION_INCHES) &
    (df_clean['product_width']  >= MIN_DIMENSION_INCHES) &
    (df_clean['product_height'] >= MIN_DIMENSION_INCHES)
]

print(f"Removed {initial_rows - len(df_clean)} outlier items.")
print(f"Final, clean dataset has {len(df_clean)} items.")

# Save the clean data
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"Clean, ready-to-use data saved to '{OUTPUT_FILE}'.")