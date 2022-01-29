[![Python application test with Github Actions](https://github.com/BokaiZZZ/sentiment_analysis/actions/workflows/main.yml/badge.svg)](https://github.com/BokaiZZZ/sentiment_analysis/actions/workflows/main.yml)

# Sentiment Analysis Microservice
This is the first individual project of IDS721 in Duke University.

## Project introduction 
This project is to achieve Cloud Continuous Delivery of Microservice. The microservice is built by Flask and deployed using Google App Engine. Continuous delivery is implemented by Cloud Bulid in GCP and Continuous Integration is reliazed through Github Actions. 

## Function demonstration
### Home page
Go to the [deployed website](http://sentiment-analysis-339601.ue.r.appspot.com/) and the home page will be shown as the figure below. 

![home_page](https://user-images.githubusercontent.com/97444802/151588859-82ee0b0c-cf6e-4a06-aa55-7b6024eff2ad.png)

### Sentiment analysis

Type "/predict/text" after the URL, where "text" is what the user want to predict. After several seconds, the predicted sentimental trend result will be returned on the page. Figures below show the prediction example for postive and negative sentimental trends respectively. 
- postive   
![postive](https://user-images.githubusercontent.com/97444802/151588434-f83d230a-2a7c-4922-9761-e2d60c9d36ec.png)
- negative
![negative](https://user-images.githubusercontent.com/97444802/151588461-4a7a4293-60c5-4d73-b600-3d09ee617cdd.png)

The demo video is uploaded on [Youtube](https://youtu.be/eYdy20-Yg8U). 

## Function introduction 

### Overview
The microservice is to predict sentimental trend for an input text and return a json whose format is {"text":input text, "trend": predicted trend}. The model is trained on [Sentiment140](http://help.sentiment140.com/home) dataset, which contains comments about a brand, product, or topic on Twitter. The dataset also has some infomation about user, which is not used in this project. Only the text and sentimental polarity of the tweet is kept to train the model. This is a binary classification task since the training set only has text with positive and negative sentiment trend. 

### Model Construct
![Sentiment Analysis Model](https://user-images.githubusercontent.com/97444802/151578663-9c7bcdda-6646-42ab-816c-b2f7feeb0497.png)

As figure shown above, the tweet text would pass through a TF-IDF vectorizer after preprocessed. Then, model can be built by combining the predictions from three different models using majority voting ensemble. 

#### 1. Preprocess
All the preprocess procedures are accomplished using [nltk](https://www.nltk.org/) package in Python. 
- Tokenize

  Tokenization is essentially splitting a phrase, sentence, paragraph, or an entire text document into smaller units, such as individual words or terms. Each of these smaller units are called tokens. The tokens could be words, numbers or punctuation marks. Before processing a natural language, we need to identify the words that constitute a string of characters. That’s why tokenization is the most basic step to proceed with NLP (text data). This is important because the meaning of the text could easily be interpreted by analyzing the words present in the text.

- Stop words  
  
  The words which are generally filtered out before processing a natural language are called stop words. These are actually the most common words in any language (like articles, prepositions, pronouns, conjunctions, etc) and does not add much information to the text. Examples of a few stop words in English are “the”, “a”, “an”, “so”, “what”.

  Stop words are available in abundance in any human language. By removing these words, we remove the low-level information from our text in order to give more focus to the important information. In order words, we can say that the removal of such words does not show any negative consequences on the model we train for our task. Removal of stop words definitely reduces the dataset size and thus reduces the training time due to the fewer number of tokens involved in the training.

- Stemming  
  
  Stemming is to remove morphological affixes from words, leaving only the word stem. Stemming algorithms aim to remove those affixes required for eg. grammatical role, tense, derivational morphology leaving only the stem of the word. This is a difficult problem due to irregular words (eg. common verbs in English), complicated morphological rules, and part-of-speech and sense ambiguities (eg. ceil- is not the stem of ceiling).

#### 2. TF-IDF

TF-IDF which means Term Frequency and Inverse Document Frequency, is a scoring measure widely used in information retrieval (IR) or summarization. TF-IDF is intended to reflect how relevant a term is in a given document. TF-IDF for a word in a document is calculated by multiplying two different metrics:

- The term frequency of a word in a document. There are several ways of calculating this frequency, with the simplest being a raw count of instances a word appears in a document. Then, there are ways to adjust the frequency, by length of a document, or by the raw frequency of the most frequent word in a document.
- The inverse document frequency of the word across a set of documents. This means, how common or rare a word is in the entire document set. The closer it is to 0, the more common a word is. This metric can be calculated by taking the total number of documents, dividing it by the number of documents that contain a word, and calculating the logarithm.

#### 3. Model ensemble

All the models are built using [scikit-learn](https://scikit-learn.org/stable/) package in Python. The ensemble strategy is majority voting ensemble.

- Naive Bayes

  Naive Bayes Classifier Algorithm is a family of probabilistic algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of a feature. Naive Bayes are mostly used in natural language processing (NLP) problems which predict the tag of a text. They calculate the probability of each tag for a given text and then output the tag with the highest one. 
  
- Logistic Regression

  Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable. It is widely used in simple Machine Learning models. 

- SGD

  Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions. SGD has been successfully applied to large-scale and sparse machine learning problems often encountered in text classification and natural language processing. 
 
