# AI Packaging Optimization 📦

This project uses a Convolutional Neural Network (CNN) to predict a product's dimensions (length, width, height) from a single image. It then recommends the most efficient shipping box from a predefined list to minimize wasted space.

## Project Goal

The main goal is to reduce shipping costs and waste by automating the selection of packaging. By predicting an item's dimensions from its image, an e-commerce system can instantly identify the smallest possible box, reducing the use of void-fill (like air pillows or paper) and saving on dimensional weight shipping fees.

## How It Works

This project is built as a 4-part pipeline:

1.  **`prepare_data.py`**: This script loads the raw `MASTER_product_dataset.csv`, cleans it, removes rows with missing data, and saves a new `MASTER_product_dataset_prepared.csv` file ready for training.
2.  **`train_model.py`**: This is the core AI component. It loads the prepared data, builds a CNN (Convolutional Neural Network) using TensorFlow/Keras, and trains it on the image dataset. The final, trained model is saved as `product_dimension_model.h5`.
3.  **`predict_packaging.py`**: This is the practical application. It's a command-line tool that loads the trained `product_dimension_model.h5`, takes a new product image as input, and outputs the AI's predicted dimensions and the single best box for shipping it.
4.  **`generate_analysis.py`**: This script is used to validate the packaging logic. It runs the optimization algorithm on the *entire* dataset (using the known "ground-truth" dimensions) to calculate the `packaging_optimization_analysis.csv` report, showing potential savings across all products.

## How to Use This Project

### Prerequisites

You must have Python and the following libraries installed:
* TensorFlow (`pip install tensorflow`)
* Pandas (`pip install pandas`)
* Scikit-learn (`pip install scikit-learn`)
* Pillow (`pip install pillow`)

### 1. Training the Model

(Note: The data and model files are not included in this repository due to their large size, as specified in the `.gitignore`.)

1.  Place your `MASTER_product_dataset.csv` and your image dataset in the root folder.
2.  Prepare the data:
    ```bash
    python prepare_data.py
    ```
3.  Train the AI model:
    ```bash
    python train_model.py
    ```

### 2. Predicting a New Product

Once you have the `product_dimension_model.h5` file, you can predict any new image:

```bash
python predict_packaging.py "path/to/your/new_image.jpg"
```

#### Example Output:
```
Analyzing image: D:\AI\pythonProject3\download.jpeg

--- AI Model Prediction ---
Predicted Length: 4.66 inches
Predicted Width:  0.28 inches
Predicted Height: 4.60 inches

--- Packaging Recommendation ---
Recommended Box (L, W, H): (10, 8, 1)
Estimated Wasted Space: 10.59%
```
