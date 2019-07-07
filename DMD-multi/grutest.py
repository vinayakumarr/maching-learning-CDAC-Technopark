from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
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
from keras.utils import np_utils
import os

trainlabels = pd.read_csv('dgcorrect/trainlabel.csv', header=None)

trainlabel = trainlabels.iloc[:,0:1]

testlabels = pd.read_csv('dgcorrect/test1label.csv', header=None)

testlabel = testlabels.iloc[:,0:1]

testlabels1 = pd.read_csv('dgcorrect/test2label.csv', header=None)

testlabel1 = testlabels1.iloc[:,0:1]


train = pd.read_csv('dgcorrect/train.txt', header=None)
test = pd.read_csv('dgcorrect/test1.txt', header=None)
test1 = pd.read_csv('dgcorrect/test2.txt', header=None)

X = train.values.tolist()
X = list(itertools.chain(*X))


T = test.values.tolist()
T = list(itertools.chain(*T))

T1 = test1.values.tolist()
T1 = list(itertools.chain(*T1))

# Generate a dictionary of valid characters
valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X+T+T1)))}

max_features = len(valid_chars) + 1

maxlen = np.max([len(x) for x in X])
print(maxlen)


# Convert characters to int and pad
X1 = [[valid_chars[y] for y in x] for x in X]

T11 = [[valid_chars[y] for y in x] for x in T]

T12 = [[valid_chars[y] for y in x] for x in T1]


X_train = sequence.pad_sequences(X1, maxlen=maxlen)

X_test = sequence.pad_sequences(T11, maxlen=maxlen)

X_test1 = sequence.pad_sequences(T12, maxlen=maxlen)

y_trainn = np.array(trainlabel)
y_testn = np.array(testlabel)
y_test1n = np.array(testlabel1)

y_train= to_categorical(y_trainn)
y_test= to_categorical(y_testn)
y_test1= to_categorical(y_test1n)

embedding_vecor_length = 128

model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(GRU(128))
model.add(Dropout(0.1))
model.add(Dense(21))
model.add(Activation('softmax'))



# try using different optimizers and different optimizer configs
model.load_weights("logs/gru/checkpoint-12.hdf5")

y_pred = model.predict_classes(X_test)
np.savetxt("res/gru1.txt",y_pred, fmt="%01d")


y_pred = model.predict_classes(X_test1)
np.savetxt("res/gru2.txt",y_pred, fmt="%01d")



'''
y_pred = model.predict_classes(X_test)
accuracy = accuracy_score(y_testn, y_pred)
recall = recall_score(y_testn, y_pred , average="weighted")
precision = precision_score(y_testn, y_pred , average="weighted")
f1 = f1_score(y_testn, y_pred, average="weighted")

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


cm = metrics.confusion_matrix(y_test, y_pred)
print("==============================================")
print(cm)
tp = cm[0][0]
fp = cm[0][1]
tn = cm[1][1]
fn = cm[1][0]
print("tp")
print(tp)
print("fp")
print(fp)
print("tn")
print(tn)
print("fn")
print(fn)

print("LSTM acc")
Acc = float(tp+tn)/float(tp+fp+tn+fn)
print(Acc)
print("precision")
prec = float(tp)/float(tp+fp)
print(prec)
print("recall")
rec = float(tp)/float(tp+fn)
print(rec)
print("F-score")
fs = float(2*tp)/float((2*tp)+fp+fn)
print(fs)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))


for file in os.listdir("logs/gru/"):
  model.load_weights("logs/gru/"+file)
  y_pred = model.predict_classes(X_test)
  accuracy = accuracy_score(y_testn, y_pred)
  recall = recall_score(y_testn, y_pred , average="weighted")
  precision = precision_score(y_testn, y_pred , average="weighted")
  f1 = f1_score(y_testn, y_pred, average="weighted")

  print(file)
  print("----------------------------------------------")
  print("accuracy")
  print("%.3f" %accuracy)
  print("racall")
  print("%.3f" %recall)
  print("precision")
  print("%.3f" %precision)
  print("f1score")
  print("%.3f" %f1)

  
  y_pred = model.predict_classes(X_test1)
  accuracy = accuracy_score(y_test1n, y_pred)
  recall = recall_score(y_test1n, y_pred , average="weighted")
  precision = precision_score(y_test1n, y_pred , average="weighted")
  f1 = f1_score(y_test1n, y_pred, average="weighted")

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



# try using different optimizers and different optimizer configs
model.load_weights("logs/gru/checkpoint-01.hdf5")

y_pred = model.predict_classes(X_test)
np.savetxt("cs-res/gru1.txt",y_pred, fmt="%01d")

model.load_weights("logs/gru/checkpoint-03.hdf5")


y_pred = model.predict_classes(X_test1)
np.savetxt("cs-res/gru2.txt",y_pred, fmt="%01d")


