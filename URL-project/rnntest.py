from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import CSVLogger
import keras
import keras.preprocessing.text
import itertools
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import callbacks
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer

trainlabels = pd.read_csv('data/testlabel.csv', header=None)
trainlabel = trainlabels.iloc[:,0:1]
testlabels = pd.read_csv('data/testlabel.csv', header=None)
testlabel = testlabels.iloc[:,0:1]



train = pd.read_csv('data/test.txt', header=None)
test = pd.read_csv('data/test.txt', header=None)



X = train.values.tolist()
X = list(itertools.chain(*X))


T = test.values.tolist()
T = list(itertools.chain(*T))




# Generate a dictionary of valid characters
valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X+T)))}

max_features = len(valid_chars) + 1

maxlen = np.max([len(x) for x in X])
print(max_features)
print(maxlen)


# Convert characters to int and pad
X1 = [[valid_chars[y] for y in x] for x in X]

T1 = [[valid_chars[y] for y in x] for x in T]


X_train = sequence.pad_sequences(X1, maxlen=maxlen)

X_test = sequence.pad_sequences(T1, maxlen=maxlen)

y_train = np.array(trainlabel)
y_test = np.array(testlabel)


embedding_vecor_length = 128

model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(SimpleRNN(128))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))


print("--------------------------------------")

unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)).T)

unique, counts = np.unique(y_test, return_counts=True)
print(np.asarray((unique, counts)).T)


score = []
name = []
import os
for file in os.listdir("logs/rnn/"):
  print(file)
  model.load_weights("logs/rnn/"+file)
  y_pred = model.predict_classes(X_test)
  y_proba = model.predict_proba(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred , average="binary")
  precision = precision_score(y_test, y_pred , average="binary")
  f1 = f1_score(y_test, y_pred, average="binary")

  print("confusion matrix")
  print("----------------------------------------------")
  print("accuracy")
  print("%.3f" %accuracy)
  print("racall")
  print("%.3f" %recall)
  print("precision")
  print("%.3f" %precision)
  print("f1score")
  print("%.3f" %f1)
  score.append(accuracy)
  name.append(file)


print(name[score.index(max(score))])






'''
#model.load_weights("logs/rnn/" +name[score.index(min(score))])

model.load_weights("logs/rnn/checkpoint-01.hdf5")
y_pred = model.predict_classes(X_test)
y_proba = model.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred , average="binary")
precision = precision_score(y_test, y_pred , average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("racall")
print("%.3f" %recall)
print("precision")
print("%.3f" %precision)
print("f1score")
print("%.3f" %f1)
'''
