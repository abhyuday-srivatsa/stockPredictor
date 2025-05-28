import pandas as pd
import nltk
import re
import sklearn

nltk.download('punkt', quiet=True)
from nltk import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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

    df['Sentence'] = junk_cleaned
    return df['Sentence'], df['Label']

data = pd.read_csv("data.csv")
sentiment_map = { 'negative': 0, 'neutral': 1, 'positive': 2 }
data['Label'] = data["Sentiment"].map(sentiment_map)
data = data.drop(columns=["Sentiment"])
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Label'])

train_sentences, train_labels = preprocess_data(train_data)
test_seentences, test_labels = preprocess_data(test_data)

def train_model(train_sentences, train_labels):
    train_sentences = [" ".join(t) for t in train_sentences]
    train_labels = [l for l in train_labels]

    vectorizer = CountVectorizer()

    vectorizer.fit(train_sentences)
    train_vect = vectorizer.transform(train_sentences)

    model = LogisticRegression()

    model.fit(train_vect, train_labels)

    return model, vectorizer

train_model, vectorizer = train_model(train_sentences, train_labels)
def predict(model, vectorizer, test_sentences, test_labels):
    test_sentences = [" ".join(t) for t in test_sentences]
    test_vect = vectorizer.transform(test_sentences)
    predictions = model.predict(test_vect)
    accuracy = sklearn.metrics.accuracy_score(test_labels, predictions)
    return predictions, accuracy

manual_entry = ["after the financial crisis of 2008 many companies were forced to restructure their debts and operations to survive, while many banks needed gavernment bailouts to avoid bankruptcy", "after a record high third quarter, AKAM is expected to grow its revenue in the cloud computing sector by over 20% within the next year", "after many experts came out and brought up points of how quantumn computing is not expected to be a sustainable industry for next few decades, IonQ is expected to drop significantly in the next few months"]
manual_entry_labels = [0,2,0]
manual_entry_df = pd.DataFrame({'Sentence': manual_entry, 'Label': manual_entry_labels})
manual_entry_sentences, manual_entry_labels = preprocess_data(manual_entry_df)
predictions, accuracy = predict(train_model, vectorizer, test_seentences, test_labels)
print("Accuracy:", accuracy)