import numpy as np
import pandas as pd
import matplotlib as plt
import cPickle as pickle
import json
import bcolz
import re
import os, sys


import keras
from keras.utils.data_utils import get_file
from keras.preprocessing import sequence
from keras.layers import Dense, Input, Flatten,Dropout, Lambda
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential, Model
from keras.datasets import imdb


# path = get_file('imdb_full.pkl',origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl',
#                md5_hash='d091312047c43cf9e4e38fef92437263')



current_dir = os.getcwd()
PROJECT_HOME_DIR = current_dir
DATA_HOME_DIR = current_dir+'/data/'


datafile = open(DATA_HOME_DIR+ 'imdb/imdb_full.pkl', 'rb')
(x_train, labels_train), (x_test, labels_test) = pickle.load(datafile)


vocab = imdb.get_word_index()
#filename = DATA_HOME_DIR+'imdb/imdb_word_index.json'
#files = open(filename)
#vocab =json.load(files)
vocab_len = len(vocab)

idx_arr = sorted(vocab, key=vocab.get)
idx2word = {v: k for k, v in vocab.iteritems()}

#vocab_size = 5000
#train  = [np.array([i if i < vocab_size-1 else vocab_size-1 for i in s]) for s in x_train]
#test  = [np.array([i if i < vocab_size-1 else vocab_size-1 for i in s]) for s in x_test]

# No resize vocabulary
train = [np.array([i for i in s]) for s in x_train]
test = [np.array([i for i in s]) for s in x_test]



input_len =500
train = sequence.pad_sequences(train, maxlen=input_len, value=0)
test = sequence.pad_sequences(test, maxlen=input_len, value=0)


embeddings_index = {}
f = open(os.path.join(DATA_HOME_DIR +'glove.6B/', 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


print('Found %s word vectors.' % len(embeddings_index))



embed_matrix = np.zeros((vocab_len, 50))

for word, i in vocab.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embed_matrix[i] = embedding_vector
        
        
        
embedding_layer = Embedding(vocab_len, 50,
                            weights=[embed_matrix],
                            input_length=input_len,
                            trainable=False)


####################################### Models #################################
embedding_layer = Embedding(vocab_len, 50,
                            weights=[embed_matrix],
                            input_length=input_len, dropout =0.2,
                            trainable=False)
seq_input = Input(shape=(input_len,), dtype='int32')
x = embedding_layer(seq_input)
x = Dropout(0.25)(x)
x = Conv1D(64, 5, border_mode='same', activation='relu')(x)
x = Dropout(0.25)(x)
x = MaxPooling1D()(x)
x = Flatten()(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.7)(x)
labels = Dense(1, activation='sigmoid')(x)

model2= Model(seq_input, labels)

model2.summary()

model2.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])






















