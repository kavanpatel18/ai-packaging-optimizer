# 📦 AI Packaging Optimizer

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

An intelligent packaging optimization tool powered by Convolutional Neural Networks (MobileNetV2) and OpenCV. This system automatically analyzes product images, extracts physical dimension features (Color Histograms & HOG), and predicts the most optimal standard box size (e.g., Flat mailer, 6x4x4, 12x12x8) to minimize shipping void space and waste.

## 🚀 Key Features
- **Computer Vision Extraction**: Calculates Histogram of Oriented Gradients (HOG) and color properties from raw images.
- **Deep Learning Prediction**: Uses a fine-tuned MobileNetV2 architecture with custom regression heads.
- **Data Pipelines**: Batch compresses, cleans, and augments e-commerce dataset JSON/CSV structures (ABO datasets).
- **Physical Bounding Math**: Computes custom padding offsets and optimal 3D rotational fits for standard packaging.

## 🛠️ Getting Started
```bash
python prepare_data.py
python extract_features.py
python train_model.py
python predict_packaging.py
```
