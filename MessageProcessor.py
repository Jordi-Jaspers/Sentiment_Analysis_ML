#imports
import pandas
import re
import nltk
import matplotlib.pyplot as plot

from math import exp

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer

#downloading libraries if not already up-to-date
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

#Define replacementents
stop_words = set(stopwords.words('english'))

REPLACE_NO_SPACE = re.compile('[|.|;|:|!|\'|?|,|\"|(|)|\|[|\|]|]|@|$')
REPLACE_WITH_SPACE = re.compile('(<br\s*/><br\s*/>)|(\-)|(\/)|#')
REPLACE_WITH_AND = re.compile('&')
REPLACE_URL = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

#preprocess messages by substituting signs with spaces
def preprocess_messages(messages):
    messages_preprocess = []
    for message in messages:
        message = message.lower()
        message = REPLACE_URL.sub('', message)
        message = REPLACE_WITH_AND.sub( 'and', message)
        message = REPLACE_NO_SPACE.sub('', message)
        message = REPLACE_WITH_SPACE.sub( ' ', message)
        messages_preprocess.append(message)  
    return messages_preprocess

#Separate every word of the sentence and delete stopwords
def tokenize_messages(messages): 
    messages_tokenized = []
    message_filtered = []

    for message in messages:
        message_tokens = word_tokenize(message) 
        message_noSW = [word for word in message_tokens if not word in stop_words] 

        for word in message_tokens: 
            if word not in stop_words: 
                message_filtered.append(word) 

        messages_tokenized.append(message_noSW)         
    
    return messages_tokenized    

# Words have different forms—for instance, “ran”, “runs”, and “running” are various forms of the same verb, “run”. 
# Depending on the requirement of your analysis, all of these versions may need to be converted to the same form, “run”.
def normalize_words(messages):
    messages_normalized = []

    for words in messages:
        words_normalized = []
        for word in words:
            porter = PorterStemmer()
            word = porter.stem(word)
            words_normalized.append(word)
        messages_normalized.append(" ".join(words_normalized))    

    return messages_normalized
