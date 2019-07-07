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
from keras.layers import Convolution1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.layers import Convolution1D, GlobalMaxPooling1D

trainlabels = pd.read_csv('data/trainlabel.csv', header=None)
trainlabel = trainlabels.iloc[:,0:1]
testlabels = pd.read_csv('data/testlabel.csv', header=None)
testlabel = testlabels.iloc[:,0:1]



train = pd.read_csv('data/train.txt', header=None)
test = pd.read_csv('data/test.txt', header=None)



X = train.values.tolist()
X = list(itertools.chain(*X))


T = test.values.tolist()
T = list(itertools.chain(*T))




# Generate a dictionary of valid characters
valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X+T)))}

max_features = len(valid_chars) + 1

maxlen = np.max([len(x) for x in X])
print(maxlen)


# Convert characters to int and pad
X1 = [[valid_chars[y] for y in x] for x in X]

T1 = [[valid_chars[y] for y in x] for x in T]


X_train = sequence.pad_sequences(X1, maxlen=maxlen)

X_test = sequence.pad_sequences(T1, maxlen=maxlen)




y_train = np.array(trainlabel)
y_test = np.array(testlabel)


hidden_dims = 128
nb_filter = 128
filter_length = 2 
embedding_vecor_length = 128
pool_length = 2
lstm_output_size = 70


model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
#model.add(MaxPooling1D(pool_length=pool_length))
#model.add(LSTM(lstm_output_size))
model.add(GlobalMaxPooling1D())
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/cnn/checkpoint-{epoch:02d}.hdf5", save_best_only=True, monitor='val_acc', mode='max')
csv_logger = CSVLogger('logs/cnn/cnnlstmanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=32, nb_epoch=1000,validation_split=0.2, shuffle=True,callbacks=[checkpointer,csv_logger])
model.save("logs/cnn/completemodel.hdf5")



