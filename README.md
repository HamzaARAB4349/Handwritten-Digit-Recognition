# Simple Handwritten Digit Recognition Model

This project implements a basic neural network model for recognizing handwritten digits. The model architecture consists of a simple feedforward neural network with one hidden layer containing 15 units and an output layer with 10 units. The input to the model is a 28x28 pixel grayscale image representing a handwritten digit.

## Dataset

The dataset used for training and testing the model is sourced from keras mnist. It contains a collection of handwritten digit images along with their corresponding labels.

## Preprocessing

Before training the model, the dataset undergoes minimal preprocessing steps to prepare it for training. This includes tasks such as:

- Reshaping images to a flat vector of size 784 (28x28).
- Scaling pixel values to a range between 0 and 1.
- Splitting the data into training and testing sets.

## Model Architecture

The architecture of the neural network model used in this project is as follows:

- Input Layer: 784 units (corresponding to a flattened 28x28 image)
- Hidden Layer: 15 units (fully connected layer with ReLU activation function)
- Output Layer: 10 units (corresponding to 10 digits, with softmax activation function)

## Training

The model is trained using the training data. During training, the model learns to map input images to their corresponding digit labels by minimizing the categorical cross-entropy loss function. The training process involves iterating over the training dataset for multiple epochs using gradient descent optimization.

## Evaluation

Once training is complete, the trained model is evaluated using the testing data to assess its performance. The evaluation metric used is accuracy, which measures the percentage of correctly classified digits.

## Usage

To use the trained model for predicting handwritten digits, follow these steps:

1. Load the trained model.
2. Preprocess input images (flatten and scale pixel values).
3. Use the model to make predictions on the preprocessed images.

## Dependencies

- Python 
- TensorFlow 
- NumPy 


