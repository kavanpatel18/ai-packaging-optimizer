import pandas as pd
import gzip
import os


def process_abo_images(images_csv_path, images_directory):
    """
    Reads the images.csv.gz file and creates a DataFrame with image paths.

    Args:
        images_csv_path (str): The path to the images.csv.gz file.
        images_directory (str): The path to the directory containing the images.

    Returns:
        pd.DataFrame: A DataFrame containing image metadata and file paths.
    """
    try:
        df = pd.read_csv(images_csv_path, compression='gzip')
        print(f"Successfully loaded {images_csv_path}")
    except FileNotFoundError:
        print(f"Error: {images_csv_path} not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return pd.DataFrame()

    def create_image_path(row):
        # Normalize the path to use the correct separator for the OS
        relative_path = row['path'].replace('/', os.sep)
        full_path = os.path.join(images_directory, relative_path)

        # Check if the file exists on disk
        if not os.path.exists(full_path):
            print(f"Warning: Image file not found at {full_path}")
            return None
        return full_path

    df['image_path'] = df.apply(create_image_path, axis=1)

    # Drop rows where the image file was not found
    df.dropna(subset=['image_path'], inplace=True)

    return df


def main():
    # Define the directories and file paths
    images_csv_path = './images.csv.gz'
    images_dir = './images/small'

    # Process the images data
    image_df = process_abo_images(images_csv_path, images_dir)

    if not image_df.empty:
        print("\nFinal Dataset Sample:")
        print(image_df.head())

        # Save the final dataset to a CSV file for future use
        csv_file_path = 'final_image_dataset.csv'
        image_df.to_csv(csv_file_path, index=False)
        print(f"\nFinal image dataset saved to {csv_file_path}")
    else:
        print("\nNo valid image data was processed.")


if __name__ == "__main__":
    main()
