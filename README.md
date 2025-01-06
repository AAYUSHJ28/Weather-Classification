# Weather Classification using Deep Learning

## Overview

This project implements a deep learning-based solution for multi-class classification of weather conditions using image data. The model classifies images into one of four categories: Cloudy, Rain, Shine, and Sunrise.

## Features

* Data Handling: Utilizes Pandas and NumPy for efficient data manipulation.

* Visualization: Includes data visualizations using Matplotlib and Seaborn.

* Deep Learning Model: Implements a convolutional neural network (CNN) using TensorFlow and Keras.

* Augmentation: Applies data augmentation using ImageDataGenerator to enhance model robustness.

## Technologies Used

* Programming Language: Python

* Libraries:

 ** Data handling: Pandas, NumPy

 * Visualization: Seaborn, Matplotlib

 * Machine Learning: Scikit-learn

 * Deep Learning: TensorFlow, Keras

## Project Structure

* The key components of the project include:

 * Dataset Preparation:

 * Image file paths and labels are extracted and organized into a DataFrame.

 * A utility function constructs full paths for specific weather conditions.

* Model Definition:

 * A CNN is built using layers such as Conv2D, MaxPooling2D, Flatten, Dense, and Dropout.

 * The model is compiled using optimizers like Adam and loss functions suitable for multi-class classification.

* Training & Evaluation:

 * The dataset is split into training and validation sets.

 * Performance metrics, such as accuracy and confusion matrices, are computed and visualized.
   
Dataset download-https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset/code



## Results

The model achieves classification of weather images with a high accuracy.

Visualization of confusion matrices and classification reports provides insights into the model's performance.
