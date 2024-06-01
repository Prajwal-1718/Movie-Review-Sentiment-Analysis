# Movie Review Sentiment Analysis

This repository contains code and resources for performing sentiment analysis on movie reviews using natural language processing (NLP) techniques.

## Overview

This project focuses on sentiment analysis of movie reviews using **natural language processing (NLP)** techniques. Sentiment analysis, also known as **opinion mining**, aims to determine the sentiment expressed in a piece of text, whether it is positive, negative, or neutral. In this project, we specifically target movie reviews and aim to classify them as either **positive or negative** based on the sentiment conveyed in the text.

## Dataset

The dataset used for this project is the IMDB dataset, which contains a large collection of movie reviews labeled as positive or negative.

You can download the dataset from [[link to dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)].

## Project Structure

The project is organized as follows:

- `Models/`: This directory contains saved model files after training and the standard vocabulary file.
- `IMDB-Dataset.csv`: This file contains the dataset used for training and testing the model.
- `Movie Review Sentiment Analysis.ipynb`: This notebook contains code used for data exploration, preprocessing, model training, and evaluation.
- `Token Count Distribution.png`: This image contains the token count distribution of the entire dataset.


## Dependencies

The project requires the following Python libraries:

- numpy
- pandas
- scikit-learn
- nltk
- matplotlib
- seaborn

## Steps Involved In Analysis

1. **Problem** : Classify the reviews of movies into either a positive sentiment or a negative sentiment review.

2. **Import the required packages and Load the data:** We utilize the IMDB dataset, which contains a large collection of movie reviews labeled with their corresponding sentiment (positive or negative). We import the necessary packages such as numpy, pandas, scikit-learn models and metrics, nltk.

2. **EDA and Preprocess the data:** We preprocess the text data by tokenizing the reviews into words, removing stopwords and punctuation, and performing stemming to reduce words to their base form.

3. **Feature Engineering:** We extract features from the preprocessed text data using techniques such as bag-of-words and countVectorizer.

4. **Model Selection and Training:** We train a machine learning model, specifically a Naive Bayes classifiers such as Gaussian, Multinominal and Bernouli models, on the extracted features to learn patterns in the text data and classify movie reviews into positive or negative sentiment categories.

5. **Evaluate the Model Performance:** We evaluate the performance of the trained model using metrics such as accuracy, precision, and recall on a separate test set of movie reviews.

By accurately predicting the sentiment of movie reviews, this project has various practical applications such as movie recommendation systems, customer feedback analysis for movie producers and distributors, and sentiment analysis of social media discussions about movies.


## Plot

Distribution of Token Counts of the Corpus 

![Token Count Distribution](https://github.com/Prajwal-1718/Movie-Review-Sentiment-Analysis/assets/68771962/df64f0fb-d3aa-4438-b8ab-87aad438a4d7)

## Confusion Matrix

1. **Gaussian NB Model**

   ![image](https://github.com/Prajwal-1718/Movie-Review-Sentiment-Analysis/assets/68771962/ef0847d9-422b-4973-9e57-d43edd7a31b3)

2. **Multinominal NB Model**

   ![image](https://github.com/Prajwal-1718/Movie-Review-Sentiment-Analysis/assets/68771962/9743f765-652b-4247-a8b4-180aefec387f)

3. **Bernouli NB Model**

   ![image](https://github.com/Prajwal-1718/Movie-Review-Sentiment-Analysis/assets/68771962/2f2f839a-1c04-42be-b98b-2f290616833b)


