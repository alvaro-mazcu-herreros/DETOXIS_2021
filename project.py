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
    # REMOVE EMOJIS AND EMOTICONES
    tweet = re.sub(r"[\U00010000-\U0010ffff]|:\)|:\(|XD|xD|;\)|:,\(|:D|D:", "", tweet)
    # REMOVE PUNCTUATIONS
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    #REMOVE STOPWORDS
    tweet_tokens = word_tokenize(tweet)
    filtered = [w for w in tweet_tokens if not w in set(stopwords.words('spanish'))]
    
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    return " ".join(lemma_words) #NO TOKENIZADO
    
texts, t1_label, t2_label = read_data(data)
tweets_cleaned = [preprocess_tweet_text(tweet) for tweet in texts]

#%% BAG OF WORDS
# El bag of words no necesita el texto tokenizado, lo hace él

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(tweets_cleaned)

# PARA SABER QUÉ FORMA EL VOCABULARIO vectorizer.get_feature_names()

X_bag_of_words = vectorizer.transform(tweets_cleaned)
#%% N-GRAMAS

ngram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(3, 3))
counts = ngram_vectorizer.fit_transform(tweets_cleaned)
counts.toarray().astype(int)

# %% TD-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(tweets_cleaned)

#tfidf_vectorizer.vocabulary_ #TO KNOW THE IMPORTANCE OF EACH WORD

#%% WORDS EMBEDDINGS

def preprocess_tweet_text(tweet):
    tweet = tweet.lower()
    # REMOVE URL
    tweet = re.sub(r"http\S+|www\S+|https\S+", "" ,tweet , flags=re.MULTILINE)
    # REMOVE @ AND #
    tweet = re.sub(r"\@\w+|\#", "", tweet)
    # REMOVE EMOJIS AND EMOTICONES
    tweet = re.sub(r"[\U00010000-\U0010ffff]|:\)|:\(|XD|xD|;\)|:,\(|:D|D:", "", tweet)
    # REMOVE PUNCTUATIONS
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    #REMOVE STOPWORDS
    tweet_tokens = word_tokenize(tweet)
    filtered = [w for w in tweet_tokens if not w in set(stopwords.words('spanish'))]
    
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    return lemma_words #TOKENIZADO
    
texts, t1_label, t2_label = read_data(data)
tweets_cleaned = [preprocess_tweet_text(tweet) for tweet in texts]


#   WORD2VEC

from gensim.models import Word2Vec

model = Word2Vec(
        tweets_cleaned,
        vector_size=30,
        min_count=5)

model.wv.most_similar('africa')

# %%

