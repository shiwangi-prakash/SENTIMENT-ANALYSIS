SENTIMENT ANALYSIS

This project focuses on sentiment analysis for Twitter data, specifically classifying tweets as positive (0) or negative (1) using machine learning models. It involves techniques such as text mining, text analysis, data analysis, and data visualization.

Table of Contents
	Introduction
	Data
	Requirements
	Model Training
	Evaluation
	Results

Introduction
Sentiment analysis is a Natural Language Processing (NLP) technique used to determine the sentiment or emotional tone of a given piece of text. In this project, we focus on sentiment analysis for tweets collected from Twitter. The goal is to classify tweets as positive or negative using various machine learning models.

Data
The dataset used for sentiment analysis of tweets is expected to be in CSV format, with specific columns and formatting requirements. Training dataset in csv file contained ‘id’, ‘label’, ‘tweet’, where the ‘id’ is a unique integer identifying the tweet sentiment is either 0 or 1, and ‘tweet’ is enclosed in “ “ similarly the testing dataset is a csv file of type ‘tweet_id’, ‘tweet’.  

![image](https://github.com/shivani-hibare-123/Sentiment-Analysis/assets/122072816/605c27ef-aa2d-4798-a5b0-110402d36b5a)


 
 Requirements
There are some general library requirements for the project and some which are specific to individual methods. The general requirements are as follows.
	Numpy
	Scikit-learn
	Scipy
	Nltk
The library requirements specific to some methods are:
	Keras
	Tensorflow
	Transformers
	Xgboost

Model Training
Preprocessing
Preprocessing Twitter data involves several steps, including:
	Removing URLs, mentions, and hashtags
	Handling special characters, punctuation, and emoticons
	Tokenization: Splitting text into individual words or tokens
	Removing stopwords: Eliminating common words that carry little sentiment information
	Stemming or lemmatization: Reducing words to their base form

![image](https://github.com/shivani-hibare-123/Sentiment-Analysis/assets/122072816/27c5c3e3-090c-4f7c-a7c0-5f53cdd1d300)

 
Popular NLP libraries such as NLTK (Natural Language Toolkit)  can be utilized for efficient preprocessing.
Feature Extraction
Various features were extracted from the preprocessed text, such as bag-of-words, TF-IDF, or word embeddings.
Model selection and training
Different machine learning algorithms were experimented with, including Logistic Regression, XGBoost , Decision tree, Recurrent Neural Networks, BERT . The models were trained on the labeled dataset.

Model Evaluation
The trained models were evaluated using appropriate evaluation metrics such as Confusion matrix, accuracy, precision, recall, and F1-score. 
CONFUSION MATRIX 

 ![image](https://github.com/shivani-hibare-123/Sentiment-Analysis/assets/122072816/b79035de-d4d0-4c64-8e1e-6a46e1d8d98d)


CLASSIFICATION REPORT
 
 ![image](https://github.com/shivani-hibare-123/Sentiment-Analysis/assets/122072816/a1165d3d-3f04-417e-9b14-d61f8c0c582e)



Results
The trained sentiment analysis model achieved an accuracy of 96% on the evaluation dataset. However, the performance may vary depending on the specific dataset and the choice of machine learning algorithms.







