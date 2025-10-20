# filename: train_model.py

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def train_dimension_model(input_csv_path, image_directory):
    """
    Loads data, defines a CNN model, trains it, and saves the result.

    Args:
        input_csv_path (str): Path to the prepared CSV data.
        image_directory (str): The root directory where images are stored.
    """
    # --- 1. Load the Prepared Data ---
    print("Step 1: Loading prepared data...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: The file was not found at '{input_csv_path}'")
        print("Please run the prepare_data.py script from Phase 1 first.")
        return

    print(f"Loaded {len(df)} records for training.")

    # --- 2. Split Data into Training and Validation Sets ---
    print("\nStep 2: Splitting data into training and validation sets...")
    train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set size: {len(train_df)} images")
    print(f"Validation set size: {len(validation_df)} images")

    # --- 3. Set Up Image Data Generators ---
    print("\nStep 3: Setting up image data generators...")
    # Normalize pixel values from [0, 255] to [0, 1]
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Define the columns for dimensions (our 'y' values or labels)
    dimension_cols = ['product_length', 'product_width', 'product_height']
    
    # Create generators that will read images from paths and associate them with dimensions
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_directory,
        x_col='image_path',
        y_col=dimension_cols,
        target_size=(128, 128),  # Resize all images to 128x128
        class_mode='raw',        # For regression, not classification
        batch_size=32            # Process images in batches of 32
    )

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validation_df,
        directory=image_directory,
        x_col='image_path',
        y_col=dimension_cols,
        target_size=(128, 128),
        class_mode='raw',
        batch_size=32
    )

    # --- 4. Define the CNN Model Architecture ---
    print("\nStep 4: Building the CNN model...")
    model = Sequential([
        # Input layer expects 128x128 pixel images with 3 color channels (RGB)
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Helps prevent overfitting

        # Output layer with 3 neurons (for length, width, height)
        Dense(3) 
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error', # Good for regression tasks
        metrics=['mean_absolute_error']
    )
    model.summary()

    # --- 5. Train the Model ---
    print("\nStep 5: Starting model training...")
    # NOTE: 10 epochs is a low number. For good results, you may need 50+.
    history = model.fit(
        train_generator,
        epochs=10, 
        validation_data=validation_generator
    )

    # --- 6. Save the Trained Model ---
    print("\nStep 6: Saving the trained model...")
    model.save('product_dimension_model.h5')
    print("Model saved successfully as 'product_dimension_model.h5'")


if __name__ == '__main__':
    # The prepared CSV from Phase 1
    PREPARED_CSV = 'MASTER_product_dataset_prepared.csv'
    # The root directory where your images are stored.
    # Set this to '.' if the image paths in the CSV are already correct from the root.
    IMAGE_ROOT_DIR = '.' 

    train_dimension_model(PREPARED_CSV, IMAGE_ROOT_DIR)