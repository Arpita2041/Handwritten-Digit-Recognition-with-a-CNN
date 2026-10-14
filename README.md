# Handwritten Digit Recognition with a CNN

This project is about recognizing handwritten digit with a CNN from Scratch. The main objective is to develop a Deep Learning model capable of recognizing handwritten digits from 0 to 9.

## Setup

For this project, I used the Scikit-learn Digits Dataset, which contains 1,797 grayscale images, each having a resolution of 8 × 8 pixels. 
I implemented a Convolutional Neural Network from scratch using Python and NumPy. 
The CNN uses convolutional layers to extract image features, max-pooling layers to reduce feature dimensions, and fully connected layers to classify digits and implemented backpropagation and Adam optimization to train the model. 

To evaluate performance, I compared my custom CNN with Logistic Regression and RBF Support Vector Machine models using the same dataset partitions.

## Result
The CNN achieved approximately 97.04% accuracy on clean test images and 95.19% accuracy under Gaussian noise.
