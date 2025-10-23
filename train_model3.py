# filename: train_model.py (Fast Test Version)

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 1. Define Model Parameters ---
INPUT_SIZE = (224, 224) 
BATCH_SIZE = 32
PREPARED_CSV = 'final_dataset_with_paths.csv'
IMAGE_ROOT_DIR = '.'
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 35


def train_dimension_model(input_csv_path, image_directory):
    """
    Loads data, defines a TRANSFER LEARNING model, and trains it
    using a 2-stage fine-tuning process.
    """
    
    # --- 2. Load and Split the Data ---
    print("Step 1: Loading prepared data...")
    df = pd.read_csv(input_csv_path)
    
    # --- !! THIS IS THE NEW LINE FOR A FAST TEST !! ---
    print("\n!!! --- FAST TEST RUN: USING ONLY 25% OF DATA --- !!!\n")
    df = df.sample(frac=0.25, random_state=42)
    # --- !! END OF NEW LINE !! ---
    
    train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set size: {len(train_df)} images")
    print(f"Validation set size: {len(validation_df)} images")

    # --- 3. Set Up Image Data Generators ---
    print("\nStep 2: Setting up image data generators...")
    # (Rest of the script is identical)
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )
    validation_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )
    dimension_cols = ['product_length', 'product_width', 'product_height']
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df, directory=image_directory, x_col='image_path',
        y_col=dimension_cols, target_size=INPUT_SIZE, class_mode='raw',      
        batch_size=BATCH_SIZE          
    )
    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validation_df, directory=image_directory, x_col='image_path',
        y_col=dimension_cols, target_size=INPUT_SIZE, class_mode='raw',
        batch_size=BATCH_SIZE
    )

    # --- 4. Build the Transfer Learning Model ---
    print("\nStep 3: Building the Transfer Learning model...")
    base_model = MobileNetV2(input_shape=(*INPUT_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3)(x) 
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error']
    )

    # --- 5. Stage 1: Train the "Head" (Warm-up) ---
    print("\n--- STAGE 1: Training the 'Head' (Base Model Frozen) ---")
    history = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS, 
        validation_data=validation_generator
    )

    # --- 6. Stage 2: Fine-Tuning (Unfreeze the "Brain") ---
    print("\n--- STAGE 2: Fine-Tuning (Unfreezing Top Layers) ---")
    base_model.trainable = True
    fine_tune_at = 100 
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # 0.00001
        loss='mean_absolute_error',
        metrics=['mean_absolute_error']
    )
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_mean_absolute_error', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.2, patience=2, min_lr=1e-6)

    history_fine_tune = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=history.epoch[-1] + 1,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr] 
    )

    # --- 7. Save the Trained Model ---
    print("\nStep 6: Saving the new, fine-tuned model...")
    model.save('product_dimension_model_TEST.keras')
    print("New TEST model saved successfully as 'product_dimension_model_TEST.keras'")


if __name__ == '__main__':
    train_dimension_model(PREPARED_CSV, IMAGE_ROOT_DIR)