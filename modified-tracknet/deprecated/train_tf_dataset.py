import numpy as np
import sys
import getopt
import os

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

dataDir = 'npy-color-augmented'
testDir = 'npy-color-small-test'
tol = 4
epochs = 100
try:
    (opts, args) = getopt.getopt(sys.argv[1:], '', [
        'load_weights=',
        'save_weights=',
    ])
    if len(opts) < 1:
        raise ''
except:
    print('usage: python3 train_faster.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath>')
    print('argument --load_weights is required only if you want to retrain the model')
    exit(1)

paramCount={
    'load_weights': 0,
    'save_weights': 0,
}

for (opt, arg) in opts:
    if opt == '--load_weights':
        paramCount['load_weights'] += 1
        load_weights = arg
    elif opt == '--save_weights':
        paramCount['save_weights'] += 1
        save_weights = arg

if paramCount['save_weights'] == 0:
    print('usage: python3 train_faster.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath>')
    print('argument --load_weights is required only if you want to retrain the model')
    exit(1)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    OPT = tf.keras.optimizers.Adadelta(lr=1.0)
    # Training for the first time
    if paramCount['load_weights'] == 0:
        model = TrackNetImproved(HEIGHT, WIDTH, NUM_CONSEC, grayscale)
        model.compile(loss=custom_loss, optimizer=OPT, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    # Retraining
    else:
        model = tf.keras.models.load_model(load_weights, custom_objects={'custom_loss': custom_loss})
            
print('Preloading training data...')
x_data = []
y_data = []
files = []

r = os.path.abspath(dataDir)
path = glob(os.path.join(r, '*.npy'))
num = len(path) // 2

data_indices = list(range(1, num+1))
for i in tqdm(data_indices):
    x_path = os.path.abspath(os.path.join(dataDir, 'x_data_' + str(i) + '.npy'))
    y_path = os.path.abspath(os.path.join(dataDir, 'y_data_' + str(i) + '.npy'))

    x_train = np.load(x_path, mmap_mode='r')
    y_train = np.load(y_path, mmap_mode='r')

    x_data.append(x_train)
    y_data.append(y_train)

    del x_train, y_train
    gc.collect()
print('Loaded!')

x_data = da.concatenate(x_data, axis=0)
y_data = da.concatenate(y_data, axis=0)

print('Dataset size:', x_data.shape, y_data.shape)

# Read test dataset
print('Loading test dataset..')
x_test = []
y_test = []

r = os.path.abspath(os.path.join(testDir))
test_path = glob(os.path.join(r, '*.npy'))
test_num = len(test_path) // 2
test_data_indices = list(range(1, test_num + 1))
for i in tqdm(test_data_indices):
    x_path = os.path.abspath(os.path.join(testDir, 'x_data_' + str(i) + '.npy'))
    y_path = os.path.abspath(os.path.join(testDir, 'y_data_' + str(i) + '.npy'))
    
    x_train = np.load(x_path, mmap_mode='r')
    y_train = np.load(y_path, mmap_mode='r')

    x_test.append(x_train)
    y_test.append(y_train)
    del x_train, y_train
    gc.collect()
print('Loaded!')

x_test = da.concatenate(x_test, axis=0)
y_test = da.concatenate(y_test, axis=0)

print('Test dataset size:', x_test.shape, y_test.shape)

def create_dataset(X, Y):
    x_list, y_list = [], []
    for i in range(NUM_CONSEC):
        x_list.append(X[i:-NUM_CONSEC-1+i])
        y_list.append(Y[i:-NUM_CONSEC-1+i])
    return da.stack(x_list, axis=1), da.stack(y_list, axis=1)

x_data, y_data = create_dataset(x_data, y_data)

# x_test, y_test = create_dataset(x_test, y_test)

# Fetching data with random restarts to simulate random data shuffling
def fetch_helper(index):
    index = index.numpy()
    return x_data[index].compute().astype('float32') / 255, \
           y_data[index].compute().astype('float32')

@tf.function
def fetch_data(index):
    X, Y = tf.py_function(func=fetch_helper, inp=[index], Tout=(tf.float32, tf.float32))
    X.set_shape(x_data.shape[1:])
    Y.set_shape(y_data.shape[1:])
    return X, Y


# options = tf.data.Options()
# options.experimental_deterministic = False
# options.experimental_optimization.autotune = True
# options.experimental_optimization.map_parallelization = True
# options.experimental_optimization.map_vectorization.enabled = True
# options.experimental_optimization.noop_elimination = True
# options.experimental_optimization.parallel_batch = True

data_length = x_data.shape[0]
dataset = (tf.data.Dataset.range(data_length)
                          .shuffle(buffer_size=data_length)
                          .map(fetch_data, num_parallel_calls=8, deterministic=False)
                          .batch(BATCH_SIZE)
                          .prefetch(buffer_size=1024))
# dataset = dataset.with_options(options)
# print(dataset)
for i in range(epochs):
    print('============epoch', i+1, '================')

    model.fit(
        dataset, 
        epochs=1,
    )
    gc.collect()
    
    print('Estimating test performance...')
    sTP = sTN = sFP1 = sFP2 = sFN = 0
    # Test performance on random frames
    num_samples = 50
    sample_len = 16
    length = x_test.shape[0]
    for j in tqdm(range(num_samples)):
        index = random.randint(0, length - NUM_CONSEC - sample_len)
        x_raw = x_test[index:index + NUM_CONSEC + sample_len].compute().astype('float32') / 255
        y_raw = y_test[index:index + NUM_CONSEC + sample_len].compute().astype('float32')

        x_val, y_val = slice_windows(x_raw, y_raw)
            
        y_pred = model.predict(x_val, batch_size=BATCH_SIZE)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype('float32')
        
        (tp, tn, fp1, fp2, fn) = outcome(y_pred, y_val, tol)
        sTP += tp
        sTN += tn
        sFP1 += fp1
        sFP2 += fp2
        sFN += fn
        
        del x_raw, y_raw, x_val, y_val, y_pred
    
    gc.collect()
    
    print("Outcome of training data of epoch " + str(i+1) + ":")
    print("Number of true positive:", sTP)
    print("Number of true negative:", sTN)
    print("Number of false positive FP1:", sFP1)
    print("Number of false positive FP2:", sFP2)
    print("Number of false negative:", sFN)
    
    try:
        print("Accuracy:", (sTP + sTN) / (sTP + sTN + sFP1 + sFP2 + sFN))
        print("Precision:", sTP / (sTP + sFP1 + sFP2))
        print("Recall:", sTP / (sTP + sFN))
    except:
        pass
    
    # Save intermediate weights during training
    if (i + 1) % 3 == 0:
        model.save(save_weights + '_' + str(i + 1), save_format='h5')

print('Saving weights......')
model.save(save_weights)
print('Done......')
