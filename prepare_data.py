# filename: prepare_data.py

import pandas as pd
import numpy as np

def prepare_dataset(input_filepath, output_filepath):
    """
    Loads, cleans, and prepares the master dataset for model training.

    Args:
        input_filepath (str): Path to the raw master CSV file.
        output_filepath (str): Path to save the cleaned CSV file.
    """
    # --- 1. Load the Master Dataset ---
    try:
        print(f"Step 1: Loading data from '{input_filepath}'...")
        df = pd.read_csv(input_filepath)
        print("File loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: The file was not found at '{input_filepath}'")
        print("Please make sure this script is in the same directory as your CSV file.")
        return

    # --- 2. Clean the Data ---
    print("\nStep 2: Cleaning the dataset...")

    # Define the columns that are absolutely essential for the model
    required_columns = ['product_height', 'product_length', 'product_width', 'image_path']

    # Check if all required columns exist
    if not all(col in df.columns for col in required_columns):
        print(f"FATAL ERROR: The CSV is missing one of the required columns: {required_columns}")
        return

    initial_rows = len(df)

    # Drop rows where any of these essential columns have missing values
    df.dropna(subset=required_columns, inplace=True)

    # Convert dimension columns to numeric types, forcing errors to become NaN
    for col in ['product_height', 'product_length', 'product_width']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows that might have become NaN during the conversion
    df.dropna(subset=required_columns, inplace=True)

    final_rows = len(df)
    print(f"Cleaning complete. Removed {initial_rows - final_rows} rows with missing data.")
    print(f"Dataset now has {final_rows} complete records.")

    # --- 3. Save the Prepared File ---
    try:
        print(f"\nStep 3: Saving the cleaned dataset to '{output_filepath}'...")
        df.to_csv(output_filepath, index=False)
        print("Successfully saved!")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

if __name__ == '__main__':
    # Define the input and output filenames
    INPUT_FILE = 'MASTER_product_dataset.csv'
    OUTPUT_FILE = 'MASTER_product_dataset_prepared.csv'

    # Run the preparation process
    prepare_dataset(INPUT_FILE, OUTPUT_FILE)