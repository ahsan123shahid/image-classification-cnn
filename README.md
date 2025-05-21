# 🧠 Retina Image Classification using CNN (AlexNet Style)

This project implements an AlexNet-inspired Convolutional Neural Network to classify retina images into 5 diagnostic categories. It was built using TensorFlow/Keras and trained on real medical image data.

## 📌 Features
- Preprocesses high-resolution retina images
- Uses a custom AlexNet-like CNN model
- Trains with categorical labels and dropout regularization
- Saves and loads the model for inference
- Predicts and visualizes classification results

## 🛠️ Tech Stack
- Python
- TensorFlow & Keras
- NumPy, Pandas
- Matplotlib
- Scikit-learn

## 📊 Dataset
The dataset contains retina images labeled by diagnosis class (0–4).  
> Note: Full dataset not included due to size. Use your own dataset or request access.

## 📈 How to Run

```bash
pip install -r requirements.txt
python cnn_model.py
