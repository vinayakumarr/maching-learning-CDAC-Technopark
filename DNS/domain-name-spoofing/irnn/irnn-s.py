"""Sample code for siamese neural net for detecting spoofing attacks"""
from __future__ import with_statement

import matplotlib
matplotlib.use('Agg')

import cPickle as pickle
import editdistance
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import strSimilarity
import itertools
from keras.layers import Dense, Input, Lambda, Flatten, Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, model_from_json
from keras import backend as K
from keras.optimizers import RMSprop
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU



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
from keras import initializers
from keras.optimizers import RMSprop
print("hi----------------------------------")

learning_rate = 1e-6

isFast = False # If True, then it runs on a very small dataset (and results won't be that great)

#dataset_type = 'process'
dataset_type = 'domain'

OUTPUT_DIR = 'output'

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

if dataset_type == 'domain':
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'domain_results.pkl')
    INPUT_FILE = os.path.join('data', 'domains_spoof.pkl')
    IMAGE_FILE = os.path.join(OUTPUT_DIR, 'domains_roc_curve.png')
    OUTPUT_NAME = 'Domain Spoofing'
elif dataset_type == 'process':
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'process_results.pkl')
    INPUT_FILE = os.path.join('data', 'process_spoof.pkl')
    IMAGE_FILE = os.path.join(OUTPUT_DIR, 'process_roc_curve.png')
    OUTPUT_NAME = 'Process Spoofing'
else:
    raise Exception('Unknown dataset type: %s' % (dataset_type,))

def generate_imgs(strings, font_location, font_size, image_size, text_location):
    font = ImageFont.truetype(font_location, font_size)

    str_imgs = []

    for st in strings:
        # Create a single channel image of floats
        img1 = Image.new('F', image_size)
        dimg = ImageDraw.Draw(img1)
        dimg.text(text_location, st.lower(), font=font)
        
        img1 = np.expand_dims(img1, axis=0)

        str_imgs.append(img1)

    return np.array(str_imgs, dtype=np.float32)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y)+np.random.rand()*.0001, axis=1, keepdims=True))
    #return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)), axis=-1, keepdims=False)

def build_model(data_shape):                                                                                                                   
    model = Sequential()

    #model.add(Convolution2D(128, 5, 5, input_shape=data_shape))
    #model.add(LeakyReLU(alpha=.1))                                                                                                                                                          

    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Convolution2D(64, 3, 3))
    #model.add(LeakyReLU(alpha=.1))

    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Flatten())                                                                                                                                                                 
    print("9999999999999999999999999999999999999")
    print(max_features)
    print(maxlen)
   
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(SimpleRNN(128,kernel_initializer=initializers.RandomNormal(stddev=0.001),recurrent_initializer=initializers.Identity(gain=1.0),activation='relu'))

    #model.add(Dense(32))

    input_a = Input(shape=(maxlen,))
    input_b = Input(shape=(maxlen,))

    print(input_a)

    processed_a = model(input_a)
    processed_b = model(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)

    # # train
    rms = RMSprop(lr=learning_rate)
    model.compile(loss=contrastive_loss, optimizer=rms) 

    return model

def initialize_encoder(self):
    """Initialize encoder for translating images to features."""
    # Set locations of models, weights, and feature parameters
    model_file = os.path.join(OUTPUT_DIR, dataset_type + '_cnn.json')
    weight_file = os.path.join(OUTPUT_DIR, dataset_type + '_cnn.h5')

    # Load model
    with open(model_file) as f:
        model = model_from_json(f.read())
    model.load_weights(weight_file)

    # Set up encoder to convert images to features
    encoder = self._tm.layers[2]
    input_shape = tuple(model.get_layer(model.layers[0].name).input_shape[1:])
    input_a = Input(shape=input_shape)
    encoder(input_a)

    return encoder

if not os.path.isfile(OUTPUT_FILE):
    font_location = "/home/sachin/vinay/other/mydga/homoglyph/modified1/arial.ttf"
    font_size = 10
    image_size = (150, 12)
    text_location = (0, 0)
    max_epochs = 25
    print("fdffsdfsdfs")
    with open(INPUT_FILE) as f:
        data = pickle.load(f)

    if isFast:
        data['train'] = random.sample(data['train'], 20000)
        data['validate'] = random.sample(data['validate'], 10000)
        data['test'] = random.sample(data['test'], 10000)
        max_epochs = 10
    print("================")    
    # organize data and translate from th to tf image ordering via .transpose( (0,2,3,1) )
    #X1_train = generate_imgs([x[0] for x in data['train']], font_location, font_size, image_size, text_location).transpose( (0,2,3,1) ) 
    #X2_train = generate_imgs([x[1] for x in data['train']], font_location, font_size, image_size, text_location).transpose( (0,2,3,1) )
    #y_train = [x[2] for x in data['train']]

    #X1_valid = generate_imgs([x[0] for x in data['validate']], font_location, font_size, image_size, text_location).transpose( (0,2,3,1) )
    #X2_valid = generate_imgs([x[1] for x in data['validate']], font_location, font_size, image_size, text_location).transpose( (0,2,3,1) )
    #y_valid = [x[2] for x in data['validate']]
    #print(y_valid)
    #X1_test = generate_imgs([x[0] for x in data['test']], font_location, font_size, image_size, text_location).transpose( (0,2,3,1) )
    #X2_test = generate_imgs([x[1] for x in data['test']], font_location, font_size, image_size, text_location).transpose( (0,2,3,1) )
    #y_test = [x[2] for x in data['test']]

    #y_train = np.array(y_train)
    #y_test = np.array(y_test)
    #y_valid = np.array(y_valid)
    
    X1_train = []
    X1_valid = []
    X1_test = []
    X2_train = []
    X2_valid = []
    X2_test = []
    y_train = []
    y_test = []
    y_valid = []
    for x in data['train']:
        X1_train.append(x[0])
        X2_train.append(x[1])
        y_train.append(x[2])        

    for x in data['validate']:
        X1_valid.append(x[0])
        X2_valid.append(x[1])
        y_valid.append(x[2])

    for x in data['test']:
        X1_test.append(x[0])
        X2_test.append(x[1])
        y_test.append(x[2])

    
    print()


    from keras.preprocessing.text import Tokenizer
    tok = Tokenizer(char_level=True, lower=True)
    t = tok.fit_on_texts(X1_train)
    t1 = tok.fit_on_texts(X2_train)
    
    # summarize what was learned
    #print(t.word_counts)
    #print(t.document_count)
    #print(t.word_index)
    #print(t.word_docs)
    
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    print(t)

    v = tok.fit_on_texts(X1_valid)
    v1 = tok.fit_on_texts(X2_valid)
    
    te = tok.fit_on_texts(X1_test)
    te1 = tok.fit_on_texts(X2_test)

    t = tok.texts_to_sequences(X1_train)
    t1 = tok.texts_to_sequences(X2_train)

    v = tok.texts_to_sequences(X1_valid)
    v1 = tok.texts_to_sequences(X2_valid)

    te = tok.texts_to_sequences(X1_test)
    te1 = tok.texts_to_sequences(X2_test)
    
    #valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X11+X21+Xv11+Xv21+Xt11+Xt21)))}

    max_features = 200

    maxlen = 100

    t = np.array(t)
    t1 = np.array(t1)
    v = np.array(v)
    v1 = np.array(v1)
    te = np.array(te)
    te1 = np.array(te1)
    print(t)
    X1_train = sequence.pad_sequences(t, maxlen=maxlen)
    X2_train = sequence.pad_sequences(t1, maxlen=maxlen)

    X1_valid = sequence.pad_sequences(v, maxlen=maxlen)
    X2_valid = sequence.pad_sequences(v1, maxlen=maxlen)

    X1_test = sequence.pad_sequences(te, maxlen=maxlen)
    X2_test = sequence.pad_sequences(te1, maxlen=maxlen)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_valid = np.array(y_valid)







    '''
    print("77777777777777777777777777777777777777777777777")
    print(len(X1_train))
    print(len(X2_train))
    
    print(type(X1_train))
    print(len(X1_train))
    #print(X1_train)
    #X1 = X1_train.tolist()
    #X11 = list(itertools.chain(*X1_train))
    #print(X11)
    #X2 = X2_train.tolist()
    #X21 = list(itertools.chain(*X2_train))

    #Xv1 = X1_valid.tolist()
    #Xv11 = list(itertools.chain(*X1_valid))
    
    #Xv2 = X2_valid.tolist()
    #Xv21 = list(itertools.chain(*X2_valid))

    #Xt1 = X1_test.tolist()
    #Xt11 = list(itertools.chain(*X1_test))

    #Xt2 = X2_test.tolist()
    #Xt21 = list(itertools.chain(*X2_test))
    #print(len(X11))
    print("1111111111111111111111")
    X11 = X1_train
    X21 = X2_train
    Xv11 = X1_valid
    Xv21 = X2_valid
    Xt11 = X1_test
    Xt21 = X2_test
    valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X11+X21+Xv11+Xv21+Xt11+Xt21)))}
    print(valid_chars)
    max_features = len(valid_chars) + 1

    maxlen = 100

    X1 = [[valid_chars[y] for y in x] for x in X11]
    X2 = [[valid_chars[y] for y in x] for x in X21]

    Xv1 = [[valid_chars[y] for y in x] for x in Xv11]
    Xv2 = [[valid_chars[y] for y in x] for x in Xv21]

    Xt1 = [[valid_chars[y] for y in x] for x in Xt11]
    Xt2 = [[valid_chars[y] for y in x] for x in Xt21]

    print("??????????????????????????????????")
    print(len(X1))
    print(len(X2))
    print(len(Xv1))
    print(len(Xv2))
    print(len(Xt1))
    print(len(Xt2))

    X1_train = sequence.pad_sequences(X1, maxlen=maxlen)
    X2_train = sequence.pad_sequences(X2, maxlen=maxlen)

    X1_valid = sequence.pad_sequences(Xv1, maxlen=maxlen)
    X2_valid = sequence.pad_sequences(Xv2, maxlen=maxlen)

    X1_test = sequence.pad_sequences(Xt1, maxlen=maxlen)
    X2_test = sequence.pad_sequences(Xt2, maxlen=maxlen)
   
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_valid = np.array(y_valid)

    print("====================================================") 
    print(X1_train.shape)
    print(y_train.shape)
    print(X2_train.shape)

    print(X1_test.shape)
    print(y_test.shape)
    print(X2_test.shape)

    print(X1_valid.shape)
    print(y_valid.shape)
    print(X2_valid.shape)

    print("====================================================")
   

    print("vin")
    '''

    
    model = build_model((12, 150, 1))

    # First figure out how many epochs we need
    max_auc = 0
    max_idx = 0
    for i in range(max_epochs):
        model.fit([X1_train, X2_train], y_train, batch_size=8, nb_epoch=1)
        scores = [-x[0] for x in model.predict([X1_valid, X2_valid])]

        t_auc = roc_auc_score(y_valid, scores)
        if t_auc > max_auc:
            print('Updated best AUC from %f to %f' % (max_auc, t_auc))
            max_auc = t_auc
            max_idx = i+1


    print(X1_train.shape)
    print(y_train.shape)
    print(X2_train.shape)
    # Train on the correct number of epochs
    model = build_model((12, 150, 1))
    model.fit([X1_train, X2_train], y_train, batch_size=8, nb_epoch=max_idx)

    # Save the NN
    json_string = model.to_json()
    model.save_weights(os.path.join(OUTPUT_DIR, dataset_type + '_cnn.h5'), overwrite=True)
    with open(os.path.join(OUTPUT_DIR, dataset_type + '_cnn.json'), 'wb') as f:
        f.write(json_string)

    scores = [-x[0] for x in model.predict([X1_test, X2_test])]
    fpr_siamese, tpr_siamese, _ = roc_curve(y_test, scores)
    roc_auc_siamese = auc(fpr_siamese, tpr_siamese)

    #
    # Run Edit distance
    #
    scores = [(editdistance.eval(x[0].lower(), x[1].lower()), len(x[0]), 1.0-x[2]) for x in data['test']]

    y_percent_score = [float(x[0])/x[1] for x in scores]

    y_score, _, y_test = zip(*scores)
    fpr_ed, tpr_ed, _ = roc_curve(y_test, y_score)
    roc_auc_ed = auc(fpr_ed, tpr_ed)

    fpr_ps, tpr_ps, _ = roc_curve(y_test, y_percent_score)
    roc_auc_ps = auc(fpr_ps, tpr_ps)

    #
    # Run editdistance visual similarity
    # 
    scores = [(strSimilarity.howConfusableAre(x[0].lower(), x[1].lower()), 1.0-x[2]) for x in data['test']]

    y_score, y_test = zip(*scores)
    fpr_edvs, tpr_edvs, _ = roc_curve(y_test, [-x for x in y_score])
    roc_auc_edvs = auc(fpr_edvs, tpr_edvs)

    #
    # Store results
    #
    results = {}
    results['editdistance_vs'] = {'fpr': fpr_edvs, 'tpr': tpr_edvs, 'auc':roc_auc_edvs}
    results['editdistance'] = {'fpr': fpr_ed, 'tpr': tpr_ed, 'auc':roc_auc_ed}
    results['editdistance_percent'] = {'fpr': fpr_ps, 'tpr': tpr_ps, 'auc':roc_auc_ps}
    results['siamese'] = {'fpr': fpr_siamese, 'tpr': tpr_siamese, 'auc':roc_auc_siamese}

    with open(OUTPUT_FILE, 'w') as f:
        pickle.dump(results, f)

with open(OUTPUT_FILE) as f:
    results = pickle.load(f)
#
# Make Figures
#
fig = plt.figure()
plt.plot(results['siamese']['fpr'], results['siamese']['tpr'],
         label='Siamese LSTM (AUC=%0.2f)' % results['siamese']['auc'])
plt.plot(results['editdistance_vs']['fpr'], results['editdistance_vs']['tpr'],
         label='Visual edit distance (AUC=%0.2f)' % results['editdistance_vs']['auc'])
plt.plot(results['editdistance']['fpr'], results['editdistance']['tpr'],
         label='Edit distance (AUC=%0.2f)' % results['editdistance']['auc'])
plt.plot(results['editdistance_percent']['fpr'], results['editdistance_percent']['tpr'],
         label='Percent edit distance (AUC=%0.2f)' % results['editdistance_percent']['auc'])
plt.plot([0, 1], [0, 1], 'k', lw=3, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('{} - Receiver Operating Characteristic'.format(OUTPUT_NAME))
plt.legend(loc="lower right")
fig.savefig(IMAGE_FILE)



