import pandas as pd
import nltk
import re
import sklearn

nltk.download('punkt', quiet=True)
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords', quiet=True)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

STOPWORD_SET = set(stopwords.words('english'))
JUNKWORD_SET = set(["'s", "'", " ",'"', "``", "''", ",", ".", "!", "?", ";", ":", "-", "_", "(", ")", "[", "]", "{", "}"])

def remove_stopwords(sentence):
    filtered = []
    for word in sentence:
        if word not in STOPWORD_SET:
            filtered.append(word)
    return filtered

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize(sentence):
    sentence = pos_tag(sentence)
    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    for word, tag in sentence:
        lemmatized.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))
    return lemmatized

def remove_junk_characters(sentence):
    year_pattern = re.compile(r'^(19|20)\d{2}$')
    filtered = []
    for word in sentence:
        if word not in JUNKWORD_SET and not year_pattern.match(word):
            filtered.append(word)
    return filtered


def preprocess_data(df):
    sentences = df.Sentence.values
    lowercased = [str(sentence).lower() for sentence in sentences if pd.notnull(sentence)]
    tokenized = [word_tokenize(sentence) for sentence in lowercased]
    cleaned = [remove_stopwords(sentence) for sentence in tokenized]
    lemmatized = [lemmatize(sentence) for sentence in cleaned]
    junk_cleaned = [remove_junk_characters((sentence)) for sentence in lemmatized]

    if 'Label' in df.columns:
        return junk_cleaned, df['Label'].tolist()
    else:
        return junk_cleaned

df1 = pd.read_csv("newData.csv")
df2 = pd.read_csv("data.csv")
df3 = pd.read_csv("all-data.csv")

sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2 }

df2['Label'] = df2['Sentiment'].map(sentiment_map)
df3['Label'] = df3['Sentiment'].map(sentiment_map)
df3 = df3.drop(columns=['Sentiment'])

df1 = df1.rename(columns={'Label': 'Label_df1'})
data_1 = pd.merge(df1[['Sentence', 'Label_df1']], df2[['Sentence', 'Label']], on='Sentence', how='inner')
data_1 = data_1.drop(columns=['Label'])

data = pd.merge(data_1[['Sentence', 'Label_df1']], df3[['Sentence', 'Label']], on='Sentence', how='inner')
data = data.drop(columns=['Label_df1'])

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Label'])

train_sentences, train_labels = preprocess_data(train_data)
test_sentences, test_labels = preprocess_data(test_data)

def train_model(train_sentences, train_labels):
    train_sentences = [" ".join(t) for t in train_sentences]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_sentences)
    train_vect = vectorizer.transform(train_sentences)

    # Define individual models
    log_clf = LogisticRegression()
    nb_clf = MultinomialNB()
    svc_clf = LinearSVC()
    rf_clf = RandomForestClassifier()

    # Ensemble model
    ensemble = VotingClassifier(
        estimators=[
            ('lr', log_clf),
            ('nb', nb_clf),
            #('svc', svc_clf),
            #('rf', rf_clf)
        ],
        voting='soft'  # Use 'soft' if all models support predict_proba
    )

    ensemble.fit(train_vect, train_labels)

    return ensemble, vectorizer

model, vectorizer = train_model(train_sentences, train_labels)


def predict(model, vectorizer, test_sentences, test_labels):
    test_sentences = [" ".join(t) for t in test_sentences]
    test_vect = vectorizer.transform(test_sentences)

    predictions = model.predict(test_vect)
    accuracy = sklearn.metrics.accuracy_score(test_labels, predictions)

    return predictions, accuracy

predictions, accuracy = predict(model, vectorizer, test_sentences, test_labels)
print(f"Accuracy: {accuracy}")