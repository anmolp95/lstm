import tensorflow as tf
import sys
import keras
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import *
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import os


def tokensizer(text):
    print(text)
    tk = Tokenizer(lower = True)
    tk.fit_on_texts(text)
    X_seq = tk.texts_to_sequences(text)
    X_pad = pad_sequences(X_seq, maxlen=100, padding='post')
    print("Tokenizer Dimension",X_pad.shape)
    return (X_pad,len(tk.word_counts.keys()))

def get_model(vocab_size):
    vocabulary_size = vocab_size + 1
    max_words = 100
    embedding_size = 32
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
    model.add(LSTM(200))
    model.add(Dense(5, activation='sigmoid'))
    return model

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    print(tf.__version__)
    print(sys.executable)
    print (sys.path)
    filename = 'train.tsv'
    dataframe = pd.read_csv(filename, sep='\t')
    dataframe.head()
    for col in dataframe.columns:
        print(col)
    print(dataframe['PhraseId'].values)
    (X,vocab_length) = tokensizer(dataframe['Phrase'].values)
    print("Tokenised Output",X)
    Y = to_categorical(dataframe["Sentiment"])
    print("Output Labels",Y)
    print("Output Label Dimension",Y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    model = get_model(vocab_length)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X, Y,
                 batch_size=30,
                 shuffle=True,
                 epochs=1,
                 validation_split=0.1)



