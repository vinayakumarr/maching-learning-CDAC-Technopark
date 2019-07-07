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
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils

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


hidden_dims=128
nb_filter = 64
filter_length =5 
embedding_vecor_length = 128
pool_length=4
lstm_output_size = 70
model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(21))
model.add(Activation('softmax'))
print(model.summary())



'''
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/cnnlstm/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('logs/cnnlstm/training_set_lstmanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=32, nb_epoch=1000,validation_split=0.33, shuffle=True,callbacks=[checkpointer,csv_logger])
score, acc = model.evaluate(X_test, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
'''

# try using different optimizers and different optimizer configs
model.load_weights("logs/cnnlstm/checkpoint-43.hdf5")


from keras.models import Model

model_feat = Model(inputs=model.input,outputs=model.get_layer('lstm_1').output)

feat_train = model_feat.predict(X_train)
print(feat_train.shape)


feat_test1 = model_feat.predict(X_test)
print(feat_test1.shape)

feat_test2 = model_feat.predict(X_test1)
print(feat_test2.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm




traindata = feat_train
trainlabel = y_trainn
trainlabel= np.int64(trainlabel)

testdata = feat_test1
testlabel = y_testn
testlabel = np.int64(testlabel)
expected = testlabel


testdata1 = feat_test2
testlabel1 = y_test1n
testlabel1 = np.int64(testlabel1)
expected1 = testlabel1



print("-----------------------------------------LR---------------------------------")
model = LogisticRegression()
model.fit(traindata, trainlabel)


# make predictions
expected = testlabel
predicted = model.predict(testdata)
predicted1 = model.predict(testdata1)


y_train1 = expected
y_pred = predicted

y_testv = expected1
y_pred1 = predicted1


accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="weighted")
precision = precision_score(y_train1, y_pred , average="weighted")
f1 = f1_score(y_train1, y_pred, average="weighted")
np.savetxt('clcna/expected.txt', y_train1, fmt='%01d')
np.savetxt('clcna/LR-predicted.txt', y_pred, fmt='%01d')

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)

print("***********************************************************************")


accuracy1 = accuracy_score(y_testv, y_pred1)
recall1 = recall_score(y_testv, y_pred1, average="weighted")
precision1 = precision_score(y_testv, y_pred1, average="weighted")
f11 = f1_score(y_testv, y_pred1, average="weighted")
np.savetxt('clcna/expected1.txt', y_testv, fmt='%01d')
np.savetxt('clcna/LR-predicted1.txt', y_pred1, fmt='%01d')


print("accuracy")
print("%.3f" %accuracy1)
print("precision")
print("%.3f" %precision1)
print("racall")
print("%.3f" %recall1)
print("f1score")
print("%.3f" %f11)




# fit a Naive Bayes model to the data
print("-----------------------------------------NB---------------------------------")
model = GaussianNB()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
predicted1 = model.predict(testdata1)


y_train1 = expected
y_pred = predicted

y_testv = expected1
y_pred1 = predicted1

accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="weighted")
precision = precision_score(y_train1, y_pred , average="weighted")
f1 = f1_score(y_train1, y_pred, average="weighted")
np.savetxt('clcna/NB-predicted.txt', y_pred, fmt='%01d')

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)


print("***********************************************************************")

accuracy = accuracy_score(y_testv, y_pred1)
recall = recall_score(y_testv, y_pred1 , average="weighted")
precision = precision_score(y_testv, y_pred1 , average="weighted")
f1 = f1_score(y_testv, y_pred1, average="weighted")
np.savetxt('clcna/NB-predicted1.txt', y_pred1, fmt='%01d')

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)



# fit a k-nearest neighbor model to the data
print("-----------------------------------------KNN---------------------------------")
model = KNeighborsClassifier()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
predicted1 = model.predict(testdata1)

# summarize the fit of the model

y_train1 = expected
y_pred = predicted

y_testv = expected1
y_pred1 = predicted1

accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="weighted")
precision = precision_score(y_train1, y_pred , average="weighted")
f1 = f1_score(y_train1, y_pred, average="weighted")
np.savetxt('clcna/KNN-predicted.txt', y_pred, fmt='%01d')

print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)


print("***********************************************************************")


accuracy1 = accuracy_score(y_testv, y_pred1)
recall1 = recall_score(y_testv, y_pred1, average="weighted")
precision1 = precision_score(y_testv, y_pred1, average="weighted")
f11 = f1_score(y_testv, y_pred1, average="weighted")
np.savetxt('clcna/KNN-predicted1.txt', y_pred1, fmt='%01d')


print("accuracy")
print("%.3f" %accuracy1)
print("precision")
print("%.3f" %precision1)
print("racall")
print("%.3f" %recall1)
print("f1score")
print("%.3f" %f11)




print("-----------------------------------------DT---------------------------------")

model = DecisionTreeClassifier()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
predicted1 = model.predict(testdata1)

# summarize the fit of the model

y_train1 = expected
y_pred = predicted

y_testv = expected1
y_pred1 = predicted1

accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="weighted")
precision = precision_score(y_train1, y_pred , average="weighted")
f1 = f1_score(y_train1, y_pred, average="weighted")
np.savetxt('clcna/DT-predicted.txt', y_pred, fmt='%01d')

print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)


print("***********************************************************************")


accuracy1 = accuracy_score(y_testv, y_pred1)
recall1 = recall_score(y_testv, y_pred1, average="weighted")
precision1 = precision_score(y_testv, y_pred1, average="weighted")
f11 = f1_score(y_testv, y_pred1, average="weighted")
np.savetxt('clcna/DT-predicted1.txt', y_pred1, fmt='%01d')


print("accuracy")
print("%.3f" %accuracy1)
print("precision")
print("%.3f" %precision1)
print("racall")
print("%.3f" %recall1)
print("f1score")
print("%.3f" %f11)




model = RandomForestClassifier()
model = model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
predicted = model.predict(testdata)
predicted1 = model.predict(testdata1)

# summarize the fit of the model

print("--------------------------------------RF--------------------------------------")

y_train1 = expected
y_pred = predicted

y_testv = expected1
y_pred1 = predicted1

accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="weighted")
precision = precision_score(y_train1, y_pred , average="weighted")
f1 = f1_score(y_train1, y_pred, average="weighted")
np.savetxt('clcna/RF-predicted.txt', y_pred, fmt='%01d')

print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)


print("***********************************************************************")


accuracy1 = accuracy_score(y_testv, y_pred1)
recall1 = recall_score(y_testv, y_pred1, average="weighted")
precision1 = precision_score(y_testv, y_pred1, average="weighted")
f11 = f1_score(y_testv, y_pred1, average="weighted")
np.savetxt('clcna/RF-predicted1.txt', y_pred1, fmt='%01d')


print("accuracy")
print("%.3f" %accuracy1)
print("precision")
print("%.3f" %precision1)
print("racall")
print("%.3f" %recall1)
print("f1score")
print("%.3f" %f11)






model = svm.SVC(kernel='rbf')
model = model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
predicted = model.predict(testdata)
predicted1 = model.predict(testdata1)

print("--------------------------------------SVMrbf--------------------------------------")
y_train1 = expected
y_pred = predicted

y_testv = expected1
y_pred1 = predicted1

accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="weighted")
precision = precision_score(y_train1, y_pred , average="weighted")
f1 = f1_score(y_train1, y_pred, average="weighted")
np.savetxt('clcna/SVM-rbf-predicted.txt', y_pred, fmt='%01d')

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)


print("***********************************************************************")


accuracy1 = accuracy_score(y_testv, y_pred1)
recall1 = recall_score(y_testv, y_pred1, average="weighted")
precision1 = precision_score(y_testv, y_pred1, average="weighted")
f11 = f1_score(y_testv, y_pred1, average="weighted")
np.savetxt('clcna/SVM-rbf-predicted1.txt', y_pred1, fmt='%01d')


print("accuracy")
print("%.3f" %accuracy1)
print("precision")
print("%.3f" %precision1)
print("racall")
print("%.3f" %recall1)
print("f1score")
print("%.3f" %f11)







model = svm.SVC(kernel='linear')
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
predicted1 = model.predict(testdata1)

# summarize the fit of the model
print("--------------------------------------SVM linear--------------------------------------")
y_train1 = expected
y_pred = predicted

y_testv = expected1
y_pred1 = predicted1

accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="weighted")
precision = precision_score(y_train1, y_pred , average="weighted")
f1 = f1_score(y_train1, y_pred, average="weighted")
np.savetxt('clcna/SVM-linear-predicted.txt', y_pred, fmt='%01d')

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)




print("***********************************************************************")


accuracy1 = accuracy_score(y_testv, y_pred1)
recall1 = recall_score(y_testv, y_pred1, average="weighted")
precision1 = precision_score(y_testv, y_pred1, average="weighted")
f11 = f1_score(y_testv, y_pred1, average="weighted")
np.savetxt('clcna/SVM-linear-predicted1.txt', y_pred1, fmt='%01d')


print("accuracy")
print("%.3f" %accuracy1)
print("precision")
print("%.3f" %precision1)
print("racall")
print("%.3f" %recall1)
print("f1score")
print("%.3f" %f11)





'''
y_pred = model.predict_classes(X_test)
np.savetxt("res/cnnlstm1.txt",y_pred, fmt="%01d")


y_pred = model.predict_classes(X_test1)
np.savetxt("res/cnnlstm2.txt",y_pred, fmt="%01d")




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

'''
