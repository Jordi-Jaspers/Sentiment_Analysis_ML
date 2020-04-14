#imports
import pandas
import re
import nltk
import matplotlib.pyplot as plot

from MessageProcessor import *
from math import exp

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer

# Read data from file 'filename.csv' 
colnames = ['Emotion', 'ID', 'Date', 'Receiver', 'User', 'Message']
data = pandas.read_csv("traindata_With_Emoticons.csv", names=colnames)

#--------------------------------------Functions----------------------------------------------#

#using the methods made in the MessageProcessor file
def processMessages(messages):
    messages_preprocess = preprocess_messages(messages)
    messages_tokenized  = tokenize_messages(messages_preprocess)
    messages_normalized = normalize_words(messages_tokenized)
    return messages_normalized
        
#--------------------------------------END Functions----------------------------------------------#

# Filtering negative and positive words
neg_messages = data.Message[data.Emotion == 0]
neutral_messages = data.Message[data.Emotion == 2]
pos_messages = data.Message[data.Emotion == 4]

# Processing the message to usable values 
clean_pos_messages = processMessages(neg_messages)
pos_words = []
for message in clean_pos_messages:
    words = message.split()
    for word in words:
        pos_words.append(word)        

clean_neg_messages = processMessages(neg_messages)
neg_words = []
for message in clean_neg_messages:
    words = message.split()
    for word in words:
        neg_words.append(word)

# In order for this data to make sense to our machine learning algorithm
# Convert each message to a numeric representation of the frequency of neg. or pos. words.
# TF = Term frequency
clean_messages = processMessages(data.Message)

cvector = CountVectorizer(stop_words='english',max_features=10000)
cvector.fit_transform(clean_messages)

neg_matrix = cvector.transform(data.Message[data.Emotion == 0])
neu_matrix = cvector.transform(data.Message[data.Emotion == 2])
pos_matrix = cvector.transform(data.Message[data.Emotion == 4])

# Getting the frequency of the negative words
neg_words = neg_matrix.sum(axis=0)
neg_words_freq = [(word, neg_words[0, idx]) for word, idx in cvector.vocabulary_.items()]
neg_tf = pandas.DataFrame(list(sorted(neg_words_freq, key = lambda x: x[1], reverse=True)),columns=['Terms','negative'])
neg_tf_df = neg_tf.set_index('Terms')
neg_tf_df.head()

# Getting the frequency of the neutral words
neu_words = neu_matrix.sum(axis=0)
neu_words_freq = [(word, neu_words[0, idx]) for word, idx in cvector.vocabulary_.items()]
neu_words_tf = pandas.DataFrame(list(sorted(neu_words_freq, key = lambda x: x[1], reverse=True)),columns=['Terms','neutral'])
neu_words_tf_df = neu_words_tf.set_index('Terms')
neu_words_tf_df.head()

# Getting the frequency of the positive words
pos_words = pos_matrix.sum(axis=0)
pos_words_freq = [(word, pos_words[0, idx]) for word, idx in cvector.vocabulary_.items()]
pos_words_tf = pandas.DataFrame(list(sorted(pos_words_freq, key = lambda x: x[1], reverse=True)),columns=['Terms','positive'])
pos_words_tf_df = pos_words_tf.set_index('Terms')
pos_words_tf_df.head()

#Creating a table of the full word frequency
term_freq_df = pandas.concat([neg_tf_df,neu_words_tf_df,pos_words_tf_df],axis=1)
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['neutral'] +  term_freq_df['positive'] 
term_freq_df.sort_values(by='total', ascending=False).head(20)

# Printing word frequency table
print(term_freq_df)

# --- Classifier problem: Logistic Regression --- #
# Logistic Regression is a good baseline model for us to use for several reasons: 
# (1) They’re easy to interpret, 
# (2) linear models tend to perform well on sparse datasets like this one, 
# (3) they learn very fast compared to other algorithms.














    






