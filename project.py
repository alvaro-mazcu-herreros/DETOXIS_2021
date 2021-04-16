#%%
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import preprocessor as p

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import re
import string
import numpy as np

#%%

stop_words = set(stopwords.words('spanish'))

data = pd.read_csv('/Users/mazcu/Documents/UPV/3º/2º Cuatrimestre/LNR/DATASET_DETOXIS.csv')

def read_data(data):
    text = list(data['comment'])
    t1_label = list(data['toxicity'])
    t2_label = list(data['toxicity_level'])
    return text, t1_label, t2_label

def preprocess_tweet_text(tweet):
    tweet = tweet.lower()
    # REMOVE URL
    tweet = re.sub(r"http\S+|www\S+|https\S+", "" ,tweet , flags=re.MULTILINE)
    # REMOVE @ AND #
    tweet = re.sub(r"\@\w+|\#", "", tweet)
    # REMOVE PUNCTUATIONS
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    #REMOVE STOPWORDS
    tweet_tokens = word_tokenize(tweet)
    
    
    
    
    
    

#%%
stop_words
#%% Data path.

data = pd.read_csv('/Users/mazcu/Documents/UPV/3º/2º Cuatrimestre/LNR/DATASET_DETOXIS.csv') 
#%%

data.head()
#%%
cols = ['comment', 'toxicity', 'toxicity_level']
df = pd.DataFrame()
for col in cols:
    df[col] = data[col]
# %%
df['comment'][46].lower()

#%%

import nltk

nltk.download('stopwords')