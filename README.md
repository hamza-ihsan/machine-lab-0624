Machine Learning Lab - Lab 2
This repository contains the second lab assignment for the Machine Learning course.

Lab Overview
Lab 2 focuses on:

Understanding and applying key machine learning concepts

Performing data preprocessing

Training and evaluating machine learning models

Analyzing and interpreting results

Contents
The Jupyter notebook with detailed explanations and results

Requirements
To run the notebook, you need:

Python

Jupyter Notebook

Common Python libraries for data science (e.g., NumPy, pandas, scikit-learn, matplotlib)
----------------------------------------------------------------------------------------------
Lab 3: NumPy Fundamentals
This repository contains a Jupyter Notebook that explores the basic features of NumPy, a popular Python library used for numerical computations. The notebook is designed for educational purposes and is part of a series of lab exercises.

Contents

A Jupyter Notebook that introduces:

Creating and working with NumPy arrays

Indexing and slicing arrays

Reshaping arrays

Performing basic arithmetic and statistical operations

Getting Started
To use this notebook, you will need:

Python installed on your system

Jupyter Notebook or JupyterLab environment

NumPy library

Once everything is set up, open the notebook and follow the instructions and examples to learn how NumPy works.

Learning Goals
This lab will help you:

Understand how NumPy handles array data

Learn common array operations and techniques

Practice reshaping and manipulating data structures

Gain familiarity with NumPy's vectorized operations

Prerequisites
Basic knowledge of Python programming

Familiarity with Jupyter Notebook
--------------------------------------------------------------------------------------------------
1. Lab 4.1: Handling Missing Data
This notebook focuses on identifying and managing missing values in the dataset. Techniques include visualizing missing data, time-based imputation, and using methods like forward fill, backward fill, and interpolation.

2. Lab 4.2: Outlier Identification and Treatment
In this notebook, outliers are detected and addressed using the Interquartile Range (IQR) method. Visual tools are used to better understand anomalies, and strategies are applied to fill or cap extreme values responsibly.

3. Lab 4.3: Introducing Holidays as Features
This notebook enriches the dataset by introducing holiday-based features. It includes marking major holidays, creating binary flags or indicators for holidays, and analyzing the impact of holidays on electricity demand patterns.

Getting Started

To explore the notebooks:

Clone or download the repository

Open the notebooks using Jupyter Notebook or JupyterLab

Ensure required libraries like pandas, numpy, matplotlib, seaborn, and holidays are installed

Dependencies

This project uses standard data analysis libraries including:

pandas

numpy

matplotlib

seaborn

holidays

-------------------------------------------------------------------------------------------------------

Lab 5.1 – Feature Extraction
This notebook focuses on deriving useful features from the raw energy consumption data. These include time-based attributes such as day of the week, hour of the day, weekends vs. weekdays, and other temporal patterns. The goal is to enhance the dataset with meaningful inputs for predictive models.

Lab 5.2 – Correlation Analysis
This notebook explores the relationships between different features using statistical correlation techniques. Visual tools such as heatmaps are used to highlight how strongly different variables relate to each other, helping to identify redundant or influential factors for model design.

Objectives

Enrich raw data with relevant time-based features

Understand the strength and direction of relationships between variables

Support data-driven decision-making for future forecasting models

Tools and Libraries Used

Jupyter Notebook

Python

pandas

numpy

matplotlib

seaborn

About Me

My name is [Abdul Haseeb], and I created this repository as part of my data science work focused on time-series and energy analytics. The notebooks here reflect my interest in using clean, structured data to build smarter, more accurate models.

-------------------------------------------------------------------------------------------------

Lab 6 – Normalization, One-Hot Encoding, and Cyclic Features
This notebook focuses on essential preprocessing steps to improve the quality and usability of time-series energy data. The lab includes:

Normalization: Scaling numerical features to a standard range to improve model performance and convergence.

One-Hot Encoding: Transforming categorical variables (such as days of the week or months) into a format suitable for machine learning algorithms.

Cyclic Transformation: Converting cyclical features (e.g., hours, days) into sine and cosine components to reflect their circular nature and preserve time-related patterns.

These steps are key in building robust forecasting models and ensuring the dataset is properly structured for downstream tasks.

Objectives

Standardize numeric data for model input

Transform categorical and cyclical time features appropriately

Prepare a clean and fully encoded dataset ready for predictive modeling

Tools and Libraries Used

Jupyter Notebook

Python

pandas

numpy

matplotlib

seaborn

scikit-learn

About Me

My name is [Abdul Haseeb]. I am passionate about data preprocessing and time-series modeling. This repository showcases part of my work in transforming raw energy data into a machine-learning-ready format.

-----------------------------------------------------------------------------------------------------------------

Lab 7 - Multi-Layer Perceptron (MLP)
This repository contains the materials for Lab 7, which focuses on the implementation of a Multi-Layer Perceptron (MLP) model for classification tasks. The main file is a Jupyter Notebook titled "Lab 7 MLP.ipynb".

Overview
This lab introduces the structure and training process of an MLP, a foundational type of neural network used in machine learning. The notebook guides users through:

Preparing data for model training

Designing a simple MLP architecture

Implementing forward and backward propagation

Evaluating the model’s performance using metrics like accuracy

Objectives
Understand the basics of neural networks

Build a working MLP model from scratch or using a framework

Train the model on a dataset and evaluate its accuracy

Visualize the model’s performance

Requirements
To run the notebook successfully, you should have Python and some libraries installed. These may include:

NumPy

Matplotlib

Scikit-learn

Optionally, a deep learning framework such as PyTorch or TensorFlow

How to Use
Clone this repository to your local machine.

Open the Jupyter Notebook file (Lab 7 MLP.ipynb) in Jupyter Lab or Jupyter Notebook.

Run each cell step by step, following the instructions and explanations in the notebook.

-----------------------------------------------------------------------------------------------------------------------

Lab 8 – 1D Convolutional Neural Network (CNN)
Welcome to this repository, which contains the materials for Lab 8: 1D Convolutional Neural Network. This lab is designed to help students or practitioners understand how convolutional neural networks can be applied to 1-dimensional data, such as time series or signal data.

Description
This lab focuses on building a simple 1D CNN model. It includes steps like preprocessing data, constructing the model architecture, training the model, and evaluating its performance. The notebook is structured to guide users through each stage of a typical machine learning workflow.

Objectives
Learn the fundamentals of 1D CNNs and how they work.

Understand how to process 1D data for deep learning tasks.

Gain practical experience implementing and training a CNN model.

Evaluate model accuracy and performance using common metrics.

Contents of This Repository
Lab 8 1D CNN.ipynb – The main Jupyter Notebook for the lab, containing explanations, implementations, and results.

Supporting data files or notes if needed for context.

Topics Covered
Convolutional layers and their role in pattern recognition

Pooling and activation functions

Model training and validation

Application to time series or signal-based datasets

Requirements
To make use of this lab, you should have:

Python installed (preferably version 3.x)

A working Jupyter Notebook environment

Libraries such as NumPy, TensorFlow or PyTorch, Matplotlib, and any others referenced in the notebook

Intended Use
This lab is ideal for:

Students learning about deep learning and CNNs

Researchers or developers exploring 1D data problems

Anyone interested in time series classification or signal processing with neural networks

-------------------------------------------------------------------------------------------------------------------

Lab 9: Recurrent Neural Networks (RNN)
Overview:
This repository contains a Jupyter Notebook focused on Recurrent Neural Networks (RNNs). It is designed as part of a lab exercise or educational module for learning how RNNs work with sequential data such as time series, text, or signals.

What You’ll Find
A structured walkthrough of RNN concepts

Step-by-step examples and explanations

Implementation of an RNN model (depending on the framework used)

Insights into how RNNs process sequences compared to traditional neural networks

Objectives
Understand the architecture and purpose of RNNs

Learn how to apply RNNs to sequential problems

Gain hands-on experience through guided experimentation in a Jupyter Notebook

Getting Started
To use this notebook:

Download or clone the repository.

Open the .ipynb file in Jupyter Notebook or JupyterLab.

Follow the explanations and execute the steps as needed.

Requirements
Basic understanding of the following will be helpful:

Python

Neural Networks

Jupyter Notebooks

ML libraries (e.g., TensorFlow or PyTorch, depending on what's used)

------------------------------------------------------------------------------------------------------

Lab 10 - LSTM Neural Network
This repository contains a Jupyter Notebook demonstrating the use of an LSTM (Long Short-Term Memory) model, developed as part of a lab assignment focused on sequential data processing and deep learning.

Overview
LSTM models are a type of Recurrent Neural Network (RNN) capable of learning long-term dependencies. This notebook walks through building an LSTM to process time-series or sequential input data. It covers data preprocessing, model design, training, and evaluation.

Features
Sequential data preprocessing

LSTM model construction

Model training and evaluation

Visualization of predictions and loss trends

Tools & Libraries Used
Python and Jupyter Notebook

NumPy and Pandas for data handling

Matplotlib for visualization

PyTorch or TensorFlow for deep learning (depending on the notebook)

How to Use
To view and run this notebook, you’ll need a Python environment with the required libraries installed. You can open it in Jupyter Notebook or use an online platform like Google Colab.

File Included
Lab 10 LSTM.ipynb: The main notebook containing all the work and explanations.

--------------------------------------------------------------------------------------------------------------------

Lab 11: Image Dataset Creation
Welcome to the repository for Lab 11: Image Dataset Making. This project focuses on creating and organizing an image dataset, typically used in machine learning or computer vision applications.

Overview
This repository includes a Jupyter Notebook that walks through the process of building an image dataset from scratch. It covers steps such as collecting images, organizing them into directories, and preparing them for model training.

Contents
Lab 11 image dataset Making.ipynb – Main notebook demonstrating dataset creation and processing.

(Optional) Additional folders or data files related to the dataset can be included here.

Features
Image gathering and organization

Dataset folder structure for training and validation

Basic image preprocessing techniques

Easy-to-follow instructions for reproducibility

Requirements
To run the notebook successfully, ensure you have a Python environment with libraries commonly used in image processing and Jupyter Notebook execution.

Suggested libraries include:

OpenCV

NumPy

Matplotlib

OS and Shutil (for file management)

You can install these libraries using a Python package manager like pip.

How to Use
Clone or download this repository.

Open the notebook in Jupyter Notebook or JupyterLab.

Follow the step-by-step instructions in the notebook to create your image dataset.

Dataset Notes
If you're working with a specific dataset (e.g., self-collected images or from online sources), be sure to describe its structure and content here, such as the number of classes, total images, and any preprocessing applied.

-------------------------------------------------------------------------------------------------------------------------------

Custom CNN Project – Lab 12
Welcome to the Lab 12 project repository, where I designed and evaluated a custom Convolutional Neural Network (CNN) for image classification. This lab explores how deep learning models can be built from scratch and trained on visual data to recognize patterns and make accurate predictions.

Project Overview
In this lab, I developed a custom CNN using a popular image dataset. The project includes the entire workflow of building a machine learning model, including:

Preparing the dataset

Designing a custom CNN architecture

Training the model

Evaluating performance

Visualizing results

This project demonstrates the power and flexibility of CNNs for computer vision tasks.

What’s Included
A Jupyter Notebook titled "Lab 12 Designed your own CNN"

Explanations and commentary for each major step

Graphs and visualizations showing how the model performed

Goals
Learn how CNNs work

Build a model without using pre-built architectures

Practice training and validating a neural network

Interpret and present model results

Tools Used
Python

Jupyter Notebook

TensorFlow / Keras (for model creation and training)

Matplotlib (for visualizing performance)

How to Use This Project
To explore or replicate this project:

Open the notebook file in Jupyter.

Follow along with the explanations and visualizations.

Experiment with modifying the model or dataset.

Whether you're a student, a beginner in deep learning, or just curious about CNNs, this project is a hands-on demonstration of building an image classification model from scratch.

----------------------------------------------------------------------------------------------------------------------------------

Lab 13: Data Augmentation with Keras ImageDataGenerator
This repository contains the notebook and related materials for Lab 13, which focuses on using image data augmentation techniques through Keras' ImageDataGenerator. These techniques help improve the performance and generalization of image classification models.

Contents
A Jupyter Notebook demonstrating various augmentation techniques.

Visualizations of how images are transformed.

Training and evaluation of a CNN model with and without data augmentation.

Objectives
To understand why data augmentation is important in deep learning.

To apply various transformations like rescaling, rotating, flipping, shifting, and zooming using Keras.

To observe the impact of augmentation on model performance.

Requirements
To run the notebook, you should have the following installed:

Python 3

TensorFlow 2

Jupyter Notebook

NumPy

Matplotlib

These libraries are commonly used for machine learning and data science tasks.

Augmentation Examples
The notebook includes examples of how one image can be transformed into multiple variations using augmentation techniques. These help the model become more robust to changes and prevent overfitting.

Model Training
A convolutional neural network (CNN) is trained using both the original dataset and the augmented dataset. The performance is compared to illustrate the benefits of augmentation.

----------------------------------------------------------------------------------------------------------------------------
