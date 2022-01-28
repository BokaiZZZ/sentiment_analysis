import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib

# preprocess 
def preprocessing(text):
    #分词
    token_words = word_tokenize(text)
    #去除停用词
    stop_words = stopwords.words('english')
    fileter_words = [word for word in token_words if word not in stop_words]
    #stemmer
    stemmer = PorterStemmer()
    fileterStem_words = [stemmer.stem(word) for word in fileter_words]
    
    return ' '.join(fileterStem_words)

def predict_label(text):
    proc_sent = preprocessing(text)
    tfidf = joblib.load('tfidf.pkl') 
    sent_tfidf = tfidf.transform([proc_sent])
    
    # predict 
    clf_NB = joblib.load('NB.joblib') 
    clf_LR = joblib.load('LR.joblib') 
    clf_SGD = joblib.load('SGD.joblib')

    y_nb = clf_NB.predict(sent_tfidf)
    y_lr = clf_LR.predict(sent_tfidf)
    y_sgd = clf_SGD.predict(sent_tfidf)
    y_sum = y_nb + y_lr + y_sgd
    y_pred = [1 if y_sum > 1 else 0]
    label_dict = { 0:'negative', 1:'positive'}
    predict_sentiment = label_dict[y_pred[0]]
    return predict_sentiment