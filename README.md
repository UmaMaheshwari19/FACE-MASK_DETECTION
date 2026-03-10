# FACE-MASK_DETECTION

# Face Mask Detection using Deep Learning

## Project Overview

This project implements a Face Mask Detection system using Deep Learning and Computer Vision techniques. The system is capable of detecting whether a person is wearing a face mask or not in real time using a webcam. It uses a Convolutional Neural Network (CNN) model trained on a dataset containing images of people with and without masks.

## Objective

The main objective of this project is to develop an automated system that can identify mask usage to help improve public safety in environments such as hospitals, airports, offices, and public areas.

## Dataset

The dataset used for training the model is downloaded from Kaggle and contains two classes:

* with_mask
* without_mask

These images are preprocessed and resized before being used for training the deep learning model.

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy

## Methodology

1. The dataset is downloaded and loaded into the system.
2. Images are preprocessed by resizing and normalizing pixel values.
3. A Convolutional Neural Network (CNN) model is built using TensorFlow and Keras.
4. The model is trained on the dataset to classify images into mask and no-mask categories.
5. The trained model is saved as "mask_model.h5".
6. Using OpenCV, real-time face detection is performed through a webcam and the model predicts whether the detected face is wearing a mask.

## Features

* Automatic dataset loading
* Image preprocessing
* Deep learning model training
* Real-time face mask detection using webcam
* Classification of mask and no-mask faces

## Applications

* Public safety monitoring
* Hospitals and healthcare facilities
* Airports and transportation hubs
* Workplace safety systems

## Output

The system detects faces through the webcam and displays:

* Green box for "Mask"
* Red box for "No Mask"

## Future Improvements

* Improve model accuracy with larger datasets
* Deploy the model as a web application using Streamlit
* Integrate with surveillance camera systems
