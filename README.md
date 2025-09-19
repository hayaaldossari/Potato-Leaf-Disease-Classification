

# Potato Leaf Disease Classification Using CNN

## Project Overview
This project builds a Convolutional Neural Network (CNN) model to classify potato leaf images into Healthy or Late Blight categories. The goal is to automate leaf disease detection, helping farmers monitor crop health and prevent losses.

## Dataset
The dataset used is Potato Leaf Healthy and Late Blight from Kaggle. It contains images of healthy and diseased potato leaves.

Dataset link: [Potato Leaf Healthy and Late Blight](https://www.kaggle.com/datasets/nirmalsankalana/potato-leaf-healthy-and-late-blight)

The dataset contains two main folders: Healthy – images of healthy leaves, Late Blight – images of leaves affected by late blight.

## Libraries and Tools
Python: NumPy, Pandas, Matplotlib, Seaborn, OpenCV, PIL  
Deep Learning: TensorFlow, Keras  
Machine Learning: scikit-learn

## Steps

1. Data Preparation
- Download and unzip the dataset.
- Create directories for training and testing datasets.
- Split dataset: 80% training, 20% testing.
- Organize images by class.

2. Data Augmentation
- Used ImageDataGenerator for training data.
- Applied rotation, shear, zoom, and horizontal flip.
- Normalized all images by scaling pixel values to [0,1].

3. Model Architecture
- CNN model built using Keras Sequential API.
- Layers include multiple Conv2D layers with ReLU activation, MaxPooling2D layers, BatchNormalization layers, Flatten and Dense layers, and output layer with sigmoid activation for binary classification.

4. Model Training
- Loss function: binary_crossentropy
- Optimizer: Adam
- Trained for 15 epochs
- Used EarlyStopping to prevent overfitting

5. Evaluation
- Evaluated on training and test datasets.
- Plotted loss and accuracy curves.
- Generated classification report (precision, recall, F1-score).


