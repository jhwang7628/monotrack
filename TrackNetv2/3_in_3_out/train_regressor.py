import numpy as np
import sys
import getopt
import os
import piexif

from glob import glob
from tqdm import tqdm

from tracknet_improved import *
from tensorflow.keras import optimizers
from tensorflow.keras.activations import *
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

from tensorflow.keras import mixed_precision
from tensorflow.keras.activations import *

import tensorflow.keras.backend as K
import tensorflow as tf

import cv2
import math
import gc
import random

from utils import *
from constants import *
import dask.array as da

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
# if use_mp:
#     policy = mixed_precision.experimental.Policy('mixed_float16')
#     mixed_precision.experimental.set_policy(policy)

#     print('Compute dtype: %s' % policy.compute_dtype)
#     print('Variable dtype: %s' % policy.variable_dtype)

try:
    (opts, args) = getopt.getopt(sys.argv[1:], '', [
        'load_weights=',
        'save_weights=',
        'dataDir=',
        'epochs=',
    ])
    if len(opts) < 3:
        raise ''
except:
    print('usage: python3 train_TrackNet3.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs>')
    print('argument --load_weights is required only if you want to retrain the model')
    exit(1)

paramCount={
    'load_weights': 0,
    'save_weights': 0,
    'dataDir': 0,
    'epochs': 0,
}

for (opt, arg) in opts:
    if opt == '--load_weights':
        paramCount['load_weights'] += 1
        load_weights = arg
    elif opt == '--save_weights':
        paramCount['save_weights'] += 1
        save_weights = arg
    elif opt == '--dataDir':
        paramCount['dataDir'] += 1
        dataDir = arg
        if dataDir[-1] == '/':
            dataDir = dataDir[:-1]
    elif opt == '--epochs':
        paramCount['epochs'] += 1
        epochs = int(arg)
    else:
        print('usage: python3 train_TrackNet3.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs>')
        print('argument --load_weights is required only if you want to retrain the model')
        exit(1)

if paramCount['save_weights'] == 0 or paramCount['dataDir'] == 0 or paramCount['epochs'] == 0:
    print('usage: python3 train_TrackNet3.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs>')
    print('argument --load_weights is required only if you want to retrain the model')
    exit(1)

# losses = {
#     "exists": "binary_crossentropy",
#     "coords": "mse",
# }
# loss_weights = {
#     "exists": 0.0,
#     "coords": 1.0,
# }
# metrics = {
#     "exists": "accuracy",
#     "coords": [],
# }
strategy = tf.distribute.MirroredStrategy()#["GPU:1", "GPU:2", "GPU:3"])
with strategy.scope():
    OPT = optimizers.Adadelta(lr=1.0)
    #     OPT = optimizers.SGD(1e-4)
#     OPT = optimizers.Adam()
    # Training for the first time
    if paramCount['load_weights'] == 0:
        model = TrackNetRegressor(HEIGHT, WIDTH, NUM_CONSEC, grayscale)
        model.compile(loss=regressor_loss, optimizer=OPT, metrics=regressor_metric)
#         model.compile(loss=losses, loss_weights=loss_weights, optimizer=OPT, metrics=metrics)
    # Retraining
    else:
        model = tf.keras.models.load_model(load_weights, custom_objects={'PReLU': PReLU})
#         model.compile(loss='mse', optimizer=OPT, metrics='mse')

r = os.path.abspath(os.path.join(dataDir))
path = glob(os.path.join(r, '*.npy'))
num = len(path) // 2
    
print('Preloading training data...')
x_data = []
y_data = []

data_indices = list(range(1, num+1))
random.shuffle(data_indices)
for i in tqdm(data_indices):
    x_path = os.path.abspath(os.path.join(dataDir, 'x_data_' + str(i) + '.npy'))
    y_path = os.path.abspath(os.path.join(dataDir, 'y_data_' + str(i) + '.npy'))
    
    x_train = np.load(x_path, mmap_mode='r')
    y_train = np.load(y_path, mmap_mode='r')
    y_processed = []
    for j in range(y_train.shape[0]):
        if np.amax(y_train[j]) == 0:
            y_processed.append([0, 0, 1])
        else:
            cx, cy = get_coordinates(y_train[j])
            y_processed.append([cx / WIDTH, cy / HEIGHT, 0])

    x_data.append(x_train)
    y_data.append(np.array(y_processed))

    del x_train, y_train, y_processed
    gc.collect()
print('Loaded!')

def slice_windows(X, Y):
    index = ( np.expand_dims(np.arange(NUM_CONSEC), 0) + 
              np.expand_dims(np.arange(len(X)-NUM_CONSEC), 0).T )
    return X[index], Y[index]

x_data = da.concatenate(x_data, axis=0)
y_data = da.concatenate(y_data, axis=0)

print('Dataset size:', x_data.shape, y_data.shape)

def fetch_data(X=x_data, Y=y_data, num_consec=NUM_CONSEC, batch_size=BATCH_SIZE):
    print('Initialized generator stream!')
    L = X.shape[0] - num_consec + 1
    index = 0
    
    x_inp = []
    y_inp = []
    while True:
        x_inp.append(X[index].compute().astype('float32') / 255)
        y_inp.append(Y[index].compute().astype('float32'))
        index += 1
        if len(x_inp) < num_consec + batch_size:
            continue
        
        x_dat, y_dat = slice_windows(np.array(x_inp), np.array(y_inp))
        yield x_dat, y_dat
        
        del x_inp[0], y_inp[0]
        index = (index + num_consec) % L

# Read test dataset
print('Loading test dataset..')

testDir = dataDir + '-test'
r = os.path.abspath(os.path.join(testDir))
test_path = glob(os.path.join(r, '*.npy'))
test_num = len(test_path) // 2
test_data_indices = list(range(1, test_num + 1))

x_test = []
y_test = []
for i in tqdm(test_data_indices):
    # Prune test set
    if i % 10 != 0:
        continue
        
    x_path = os.path.abspath(os.path.join(testDir, 'x_data_' + str(i) + '.npy'))
    y_path = os.path.abspath(os.path.join(testDir, 'y_data_' + str(i) + '.npy'))
    
    x_train = np.load(x_path, mmap_mode='r')
    y_train = np.load(y_path, mmap_mode='r')
    y_processed = []
    for j in range(y_train.shape[0]):
        if np.amax(y_train[j]) == 0:
            y_processed.append([0, 0, 1])
        else:
            cx, cy = get_coordinates(y_train[j])
            y_processed.append([cx / WIDTH, cy / HEIGHT, 0])

    x_test.append(x_train)
    y_test.append(np.array(y_processed))
    del x_train, y_train, y_processed
    gc.collect()

print('Loaded!')

x_test = da.concatenate(x_test, axis=0)
y_test = da.concatenate(y_test, axis=0)

print('Test dataset size:', x_test.shape, y_test.shape)

# Create dataset generator
if grayscale:
    xshape = (BATCH_SIZE, NUM_CONSEC, HEIGHT, WIDTH)
else:
    xshape = (BATCH_SIZE, NUM_CONSEC, HEIGHT, WIDTH, 3)
yshape = (BATCH_SIZE, NUM_CONSEC, 3)

data_generator = fetch_data(x_data, y_data, NUM_CONSEC, BATCH_SIZE)
test_data_generator = fetch_data(x_test, y_test, NUM_CONSEC, BATCH_SIZE)
dataset = tf.data.Dataset.from_generator(
    lambda: data_generator,
    output_shapes=(xshape, yshape),
    output_types=(tf.float32, tf.float32),
).prefetch(tf.data.experimental.AUTOTUNE)

test_data_generator = fetch_data(x_test, y_test, NUM_CONSEC, BATCH_SIZE)
test_dataset = tf.data.Dataset.from_generator(
    lambda: test_data_generator,
    output_shapes=(xshape, yshape),
    output_types=(tf.float32, tf.float32),
).prefetch(tf.data.experimental.AUTOTUNE)

for i in range(epochs):
    print('============epoch', i+1, '================')

    history = model.fit(
        dataset, 
        epochs=1, 
        steps_per_epoch=x_data.shape[0] // BATCH_SIZE,
        validation_data=test_dataset,
        validation_steps=x_test.shape[0] // BATCH_SIZE
    )
    loss = sum(history.history['loss'])
    
    # Save intermediate weights during training
    if (i + 1) % 10 == 0:
        model.save(save_weights + '_' + str(i + 1), save_format='h5')

print('Saving weights......')
model.save(save_weights)
print('Done......')
