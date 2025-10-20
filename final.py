import pandas as pd
import json
import gzip
import os


def process_abo_listings(directory):
    """
    Reads all listings files from a directory and combines them into a single DataFrame,
    handling missing columns gracefully.

    Args:
        directory (str): The path to the directory containing the listings_*.json.gz files.

    Returns:
        pd.DataFrame: A DataFrame containing the combined and cleaned product metadata.
    """
    all_data = []

    # Get a list of all files in the directory that match the pattern
    listings_files = [f for f in os.listdir(directory) if f.startswith('listings_') and f.endswith('.json.gz')]

    if not listings_files:
        print("No listings_*.json.gz files found in the specified directory.")
        return pd.DataFrame()

    for filename in listings_files:
        file_path = os.path.join(directory, filename)
        print(f"Processing {filename}...")

        try:
            with gzip.open(file_path, 'rb') as f:
                for line in f:
                    try:
                        all_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        print(f"Skipping bad line in {filename}: {e}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

    print("\nAll listings files processed.")

    if not all_data:
        print("No valid data was extracted. Returning empty DataFrame.")
        return pd.DataFrame()

    # Create a DataFrame from the combined data
    df = pd.DataFrame(all_data)

    # --- Check for and handle missing columns before filtering ---
    required_columns = ['item_id', 'main_image_id', 'dimensions', 'weight']
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in the dataset. Filling with None.")
            df[col] = None

    # Filter for the necessary columns. This will not raise an error now.
    df_filtered = df[required_columns].copy()

    # Drop rows where crucial data (dimensions or main_image_id) is missing
    df_filtered.dropna(subset=['dimensions', 'main_image_id'], inplace=True)

    # Check if 'dimensions' column is empty before normalizing
    if not df_filtered['dimensions'].empty:
        # Normalize the dimensions column
        dimensions_df = pd.json_normalize(df_filtered['dimensions'])

        # Join the normalized dimensions back to the original DataFrame
        df_final = df_filtered.join(dimensions_df)

        # Drop the original dimensions column
        df_final.drop('dimensions', axis=1, inplace=True)
    else:
        # If 'dimensions' is empty, just keep the filtered DataFrame
        print("No products with dimension data were found.")
        df_final = df_filtered

    return df_final


def create_image_paths(df, images_directory):
    """
    Adds a column to the DataFrame with the local file path for each image.

    Args:
        df (pd.DataFrame): The DataFrame with product data.
        images_directory (str): The path to the directory containing the images.

    Returns:
        pd.DataFrame: The DataFrame with a new 'image_path' column.
    """

    def get_image_path(image_id):
        if isinstance(image_id, str) and len(image_id) >= 2:
            # The images are organized in folders named by the first two characters of the ID
            folder = image_id[:2]
            # Construct the full file path
            full_path = os.path.join(images_directory, folder, f"{image_id}.jpg")

            # Check if the file exists on disk
            if os.path.exists(full_path):
                return full_path
            return None

    df['image_path'] = df['main_image_id'].apply(get_image_path)

    # Drop rows where the image file was not found
    df.dropna(subset=['image_path'], inplace=True)

    return df


if __name__ == "__main__":
    # Define the directories where your files are located
    listings_dir = './'
    images_dir = './images/small'

    # Process the metadata
    product_df = process_abo_listings(listings_dir)

    if not product_df.empty:
        # Link the metadata to the image file paths
        final_dataset = create_image_paths(product_df, images_dir)

        if not final_dataset.empty:
            # Display a sample of the final dataset
            print("\nFinal Dataset Sample:")
            print(final_dataset.head())

            # Save the final dataset to a CSV file for future use
            csv_file_path = 'final_product_dataset.csv'
            final_dataset.to_csv(csv_file_path, index=False)
            print(f"\nFinal dataset saved to {csv_file_path}")
        else:
            print("\nNo products with both image and dimension data were found.")
