import pandas as pd

def preprocess_data(df):
    sentences = df.Sentence.values
    lowercased = [sentence.lower() for sentence in sentences]
    tokenized = [sentence.split() for sentence in lowercased]
    #Stop words
    # stemming
    #lemmatization
    df['Sentence'] = tokenized
    return df

train_data = pd.read_csv("finance_train.csv")
test_data = pd.read_csv("finance_test.csv")

train_data = preprocess_data(train_data)
print(train_data.head())