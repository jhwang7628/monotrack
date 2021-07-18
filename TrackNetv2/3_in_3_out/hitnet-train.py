#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys, getopt
import os
from glob import glob
import piexif
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from keras.models import *
from keras.layers import *
from HitNet import HitNet
from TrackNet3 import TrackNet3
import keras.backend as K
from keras import optimizers
from keras.activations import *
import tensorflow as tf
import cv2
import math

print('Done importing.')

# In[2]:


BATCH_SIZE=2
HEIGHT=288
WIDTH=512
mag = 1
sigma = 2.5

#Return the values of accuracy, precision and recall
def evaluation(y_pred, y_true, tol):
    (TP, TN, FP1, FP2, FN) = outcome(y_pred, y_true, tol)
    try:
        accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except:
        accuracy = 0
    try:
        precision = TP / (TP + FP1 + FP2)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0
    return (accuracy, precision, recall)


# In[3]:


def custom_loss(y_true, y_pred):
    loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
    return K.mean(loss)


# In[4]:


load_weights = './model906_30'
save_weights = '/home/groups/djames/prj-sports-video/hitnet_weights_3'

print('Loading tracknet...')
tracknet_model = load_model(load_weights, custom_objects={'custom_loss':custom_loss})

model = HitNet(HEIGHT, WIDTH)

for l, layer in enumerate(tracknet_model.layers):
    if l >= len(model.layers)-model.offset:
        break
    model.layers[l].set_weights(layer.get_weights())
    model.layers[l].trainable = False


# In[5]:

print('Compiling model...')
opt = optimizers.Adam(lr=1e-8)
model.compile(
    optimizer='adam', 
    loss="binary_crossentropy",
    metrics=['accuracy']
)
model.summary()


# In[ ]:


from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import random

epochs = 10
dataDir = '/home/groups/djames/prj-sports-video/npy'

r = os.path.abspath(os.path.join(dataDir))
path = glob(os.path.join(r, '*.npy'))
num = len(path) / 2
idx = np.arange(num, dtype='int') + 1
print('Beginning training......')
for i in range(epochs):
    print('============epoch', i+1, '================')
    np.random.shuffle(idx)
    for j in idx:
        x_train = np.load(os.path.abspath(os.path.join(dataDir, 'x_data_' + str(j) + '.npy')))
        y_train = np.load(os.path.abspath(os.path.join(dataDir, 'y_data_' + str(j) + '.npy')))
        y_train = np.sum(y_train, axis=1)

        p = float(sum(y_train == 1)) / sum(y_train == 0)
        print(p)
        subsample = []
        for i in range(x_train.shape[0]):
            if random.random() < p or y_train[i]:
                subsample.append(i)

        x_sub = x_train[subsample]
        y_sub = y_train[subsample]
        print(y_sub)
        cw = class_weight.compute_class_weight('balanced', [0,1], y_train)
        model.fit(
            x_sub, y_sub, 
            batch_size=1, 
            epochs=15,
            class_weight={0: cw[0], 1: cw[1]},
            shuffle=True)
            #validation_split=0.1)
        
        y_pred = np.round(model.predict(x_train) > 0.5)
        print(classification_report(y_train, y_pred))
        del x_train
        del y_train
        
    #Save intermediate weights during training
    if (i + 1) % 1 == 0:
        model.save(save_weights + '_' + str(i + 1))

print('Saving weights......')
model.save(save_weights)
print('Done......')


# In[ ]:




