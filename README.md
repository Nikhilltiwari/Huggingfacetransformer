# Huggingfacetransformer
Sentiment Analysis with DistilBERT
This repository contains a project for training, evaluating, and deploying a sentiment analysis model using DistilBERT. The project involves tokenizing text data, training a classification model, evaluating its performance, making predictions, and saving the model.

Table of Contents
Project Overview
Installation
Dataset
Usage
Data Preparation
Model Training
Model Evaluation
Making Predictions
Saving the Model
Results
License
Project Overview
The goal of this project is to perform sentiment analysis using the DistilBERT model from the Hugging Face transformers library. The project demonstrates the entire pipeline from data preparation, model training, evaluation, and prediction to model saving.

Installation
To run this project, you need to install the following dependencies:
Copy code
pip install tensorflow transformers scikit-learn pandas
Dataset
The dataset used in this project is assumed to be a tab-separated file named SMSSpamCollection, which contains text messages labeled as 'ham' (not spam) or 'spam'. You need to place this file in the same directory as your script.
