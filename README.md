# MNIST Digit Recognition with Convolutional Neural Networks

This project demonstrates how to build a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset using TensorFlow and Keras. The project is implemented in a Google Colab notebook.

## Overview

The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits (0-9). The goal is to train a model that can accurately classify these digits.

## Features

- Data loading and preprocessing
- CNN model architecture with Batch Normalization and Dropout
- Data augmentation for improved generalization
- Training with learning rate scheduling and early stopping
- Model evaluation and visualization of results

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SimranRojia/handwritten_digit_recognition.git
   cd handwritten_digit_recognition
   ```


2.   **Install dependencies**:
Ensure you have Python and pip installed. Then, install the required packages:

```bash

pip install -r requirements.txt
```
3. **Open the Colab notebook:**

Go to Google Colab.
Click on File > Upload notebook.
Select the mnist_cnn_colab.ipynb file from the cloned repository and upload it.
Run the notebook:

Follow the instructions in the notebook to run each cell sequentially.


4. **Usage Instructions:**
Data Loading and Preprocessing:

The notebook includes code to load the MNIST dataset directly from TensorFlow datasets and preprocess it for training.
Model Definition:

The notebook defines a Convolutional Neural Network (CNN) architecture with layers for convolution, pooling, batch normalization, and dropout.
Training the Model:

The model is trained using the training data with data augmentation, learning rate scheduling, and early stopping to prevent overfitting.
Evaluating the Model:

The notebook evaluates the model on the test data and displays the accuracy.
Visualizing Predictions:

The notebook includes code to visualize the predicted labels on test images.
Results
The model achieves an accuracy of approximately 99.17% on the test set.
Training and validation loss/accuracy plots are provided for visual inspection.
Sample images with predicted labels are displayed.
