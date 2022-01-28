import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier as SGD
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from joblib import dump
nltk.download('punkt')
nltk.download('stopwords')

def preprocessing(text):
    #分词
    token_words = word_tokenize(text)
    #去除停用词
    stop_words = stopwords.words('english')
    fileter_words = [word for word in token_words if word not in stop_words]
    #stemmer
    stemmer = PorterStemmer()
    fileterStem_words = [stemmer.stem(word) for word in fileter_words]
    
    return ' '.join(fileterStem_words) #返回一个字符串  以空格间隔

# data cleaning 
df_raw = pd.read_csv('training.1600000.processed.noemoticon.csv')
df = pd.DataFrame(columns = ['text','label'])
df['label'] = df_raw['0']
df['text'] = df_raw.iloc[:,-1]
df['text'] = df['text'].apply(preprocessing)

# Get feature matrix and target label
dataset = df.to_numpy() 
target = dataset[:,1]
features = dataset[:,0]
tfidf = TfidfVectorizer()
X_processed = tfidf.fit_transform(features)
dump(tfidf, 'tfidf.pkl') 
le = LabelEncoder()
target_processed = le.fit_transform(target) 

# Train the model 
X_train, X_test, y_train, y_test = train_test_split(X_processed, target_processed, test_size=0.2, random_state=42)
# Naive Bayes
model_NB = MultinomialNB().fit(X_train, y_train)
y_pred = model_NB.predict(X_test)
NB_score = model_NB.score(X_test, y_test)
dump(model_NB, 'NB.joblib') 

# Logistic Regression 
grid= {"C":np.logspace(-1,1,3)} # Decide which settings you want for the grid search. 
model_LR = GridSearchCV(LogisticRegression(max_iter = 2000),grid, cv = 5) 
y_pred = model_LR.fit(X_train, y_train).predict(X_test) # Fit the model.
LR_score = model_LR.score(X_test, y_test)
dump(model_LR, 'LR.joblib') 

# SGD
sgd_params = {'alpha': [0.00006, 0.00007, 0.00008, 0.0001, 0.0005]} # Regularization parameter
model_SGD = GridSearchCV(SGD(random_state = 0, shuffle = True, loss = 'modified_huber'), 
                        sgd_params, scoring = 'roc_auc', cv = 5) # Find out which regularization parameter works the best. 
y_pred = model_SGD.fit(X_train, y_train).predict(X_test) # Fit the model.
model_SGD.score(X_test, y_test)
dump(model_SGD, 'SGD.joblib') 