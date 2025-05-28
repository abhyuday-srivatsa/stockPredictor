import pandas as pd
import nltk
import re
nltk.download('punkt')
from nltk import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

STOPWORD_SET = set(stopwords.words('english'))
JUNKWORD_SET = set(["'s", "'", " ",'"', "``", "''", ",", ".", "!", "?", ";", ":", "-", "_", "(", ")", "[", "]", "{", "}"])

def remove_stopwords(sentence):
    filtered = []
    for word in sentence:
        if word not in STOPWORD_SET:
            filtered.append(word)
    return filtered

def stem(sentence):
    stemmer = PorterStemmer()
    stemmed = []
    for word in sentence:
        stemmed.append(stemmer.stem(word))
    return stemmed

def remove_junk_characters(sentence):
    year_pattern = re.compile(r'^(19|20)\d{2}$')
    filtered = []
    for word in sentence:
        if word not in JUNKWORD_SET and not year_pattern.match(word):
            filtered.append(word)
    return filtered


def preprocess_data(df):
    sentences = df.Sentence.values
    lowercased = [sentence.lower() for sentence in sentences]
    tokenized = [sentence.split() for sentence in lowercased]
    cleaned = [remove_stopwords(sentence) for sentence in tokenized]
    stemmed = [stem(sentence) for sentence in cleaned]
    junk_cleaned = [remove_junk_characters((sentence)) for sentence in stemmed]
    #lemmatization
    df['Sentence'] = junk_cleaned
    return df

train_data = pd.read_csv("finance_train.csv")
test_data = pd.read_csv("finance_test.csv")

train_data = preprocess_data(train_data)
print(train_data)