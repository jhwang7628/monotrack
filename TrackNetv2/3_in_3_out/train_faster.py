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

dataDir = 'npy-color-small'
testDir = 'npy-color-small-test'
elasDir = 'npy-color-mada'
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

strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1', 'GPU:2', 'GPU:3'])
with strategy.scope():
#     OPT = optimizers.Adadelta(lr=1.0)
    OPT = optimizers.Adam(lr=5e-6)
    # Training for the first time
    if paramCount['load_weights'] == 0:
        model = TrackNetRegressor(HEIGHT, WIDTH, NUM_CONSEC, grayscale)
        model.compile(loss=custom_loss, optimizer=OPT, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    # Retraining
    else:
        model = tf.keras.models.load_model(load_weights, custom_objects={'custom_loss': custom_loss})
#         imgs_input = Input(shape=(NUM_CONSEC, HEIGHT, WIDTH, 3))
#         x = K.permute_dimensions(imgs_input, (0, 1, 4, 2, 3))
#         imgs_output = Reshape(target_shape=(NUM_CONSEC * 3, HEIGHT, WIDTH))(x)
#         model = Model(imgs_input, model(imgs_output))

#         model.compile(loss=custom_loss, optimizer=OPT, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

# Fetching data with random restarts to simulate random data shuffling
def fetch_data(X, Y, num_consec=NUM_CONSEC, batch_size=BATCH_SIZE):
    print('Initialized generator stream!')
    L = X.shape[0] - num_consec - 1
    T = 420
    index = 0
    
    x_inp = []
    y_inp = []
    while True:
        x_inp.append(X[index].compute().astype('float32') / 255)
        y_inp.append(Y[index].compute().astype('float32'))
        index = (index + 1) % L
        if len(x_inp) < num_consec + batch_size:
            continue
        
        x_dat, y_dat = slice_windows(np.array(x_inp), np.array(y_inp))
        yield x_dat, y_dat
        
        del x_inp[0], y_inp[0]
        del x_dat, y_dat
        T -= 1
        if T == 0:
            T = 420
            index = random.randint(0, L-1)
            del x_inp, y_inp
            x_inp, y_inp = [], []
            
print('Preloading training data...')
x_data = []
y_data = []
files = []

r = os.path.abspath(dataDir)
path = glob(os.path.join(r, '*.npy'))
num = len(path) // 2

data_indices = list(range(1, num+1))
for i in data_indices:
    x_path = os.path.abspath(os.path.join(dataDir, 'x_data_' + str(i) + '.npy'))
    y_path = os.path.abspath(os.path.join(dataDir, 'y_data_' + str(i) + '.npy'))
    files.append((x_path, y_path))

r = os.path.abspath(elasDir)
path = glob(os.path.join(r, '*.npy'))
num = len(path) // 2
    
data_indices = list(range(1, num+1))
for i in data_indices:
    x_path = os.path.abspath(os.path.join(elasDir, 'x_data_' + str(i) + '.npy'))
    y_path = os.path.abspath(os.path.join(elasDir, 'y_data_' + str(i) + '.npy'))
    files.append((x_path, y_path))
    

random.shuffle(files)
for x_path, y_path in tqdm(files):
    x_train = np.load(x_path, mmap_mode='r')
    y_train = np.load(y_path, mmap_mode='r')
    if not x_train.shape[0] or not y_train.shape[0]:
        continue
        
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
random.shuffle(test_data_indices)

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

# Create dataset generator
if grayscale:
    xshape = (BATCH_SIZE, NUM_CONSEC, HEIGHT, WIDTH)
else:
    xshape = (BATCH_SIZE, NUM_CONSEC, HEIGHT, WIDTH, 3)
yshape = (BATCH_SIZE, NUM_CONSEC, HEIGHT, WIDTH)

data_generator = fetch_data(x_data, y_data, NUM_CONSEC, BATCH_SIZE)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = tf.data.Dataset.from_generator(
    lambda: data_generator,
    output_shapes=(xshape, yshape),
    output_types=(tf.float32, tf.float32),
).with_options(options).shuffle(buffer_size=200).prefetch(tf.data.experimental.AUTOTUNE)


for i in range(epochs):
    print('============epoch', i+1, '================')

    model.fit(
        dataset, 
        epochs=1, 
        steps_per_epoch=x_data.shape[0] // BATCH_SIZE
    )
    
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
