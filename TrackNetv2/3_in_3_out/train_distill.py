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
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from tensorflow.keras import mixed_precision

import tensorflow.keras.backend as K
import tensorflow as tf

import cv2
import math
import gc
import random

from utils import *
from constants import *
import dask.array as da

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1})
# sess = tf.compat.v1.Session(config=config)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

tracknet_weights='model906_30'
# student_weights='attention_distilled_60'
student_weights='tracknet_improved_42'
save_weights='tracknet_improved'
#dataDir='npy-color-small'
#dataDir='npy-color-augmented-elastic'
dataDir='npy-color-no-augmentation'
epochs=10
tol=4

#strategy = tf.distribute.MirroredStrategy(["GPU:1", "GPU:2", "GPU:3"])
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    OPT = optimizers.Adadelta(lr=1.0)

    tracknet = load_model(tracknet_weights, custom_objects={'custom_loss': custom_loss})
    imgs_input = Input(shape=(NUM_CONSEC, HEIGHT, WIDTH, 3))
    x = K.permute_dimensions(imgs_input, (0, 1, 4, 2, 3))
    imgs_output = Reshape(target_shape=(NUM_CONSEC * 3, HEIGHT, WIDTH))(x)
    teacher = Model(imgs_input, tracknet(imgs_output))

#     student_net = load_model(student_weights, custom_objects={'custom_loss': custom_loss})
#     imgs_input = Input(shape=(NUM_CONSEC, HEIGHT, WIDTH, 3))
#     imgs_output = K.mean(imgs_input, axis=-1)
#     student = Model(imgs_input, student_net(imgs_output))
    #student = load_model(student_weights, custom_objects={'custom_loss': custom_loss})
    student = TrackNetImproved(HEIGHT, WIDTH, NUM_CONSEC, grayscale)

    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=OPT,
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)],
        student_loss_fn=custom_loss,
        distillation_loss_fn=tf.keras.losses.MeanSquaredError(reduction="sum"),
        alpha=0.9,
        temperature=1,
    )

    teacher.summary()
    student.summary()

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

    x_data.append(x_train)
    y_data.append(y_train)

    del x_train, y_train
    gc.collect()
print('Loaded!')

def slice_windows(X, Y):
    index = ( np.expand_dims(np.arange(NUM_CONSEC), 0) +
              np.expand_dims(np.arange(len(X)-NUM_CONSEC), 0).T )
    return X[index], Y[index]

x_data = da.concatenate(x_data, axis=0)
y_data = da.concatenate(y_data, axis=0)

print('Dataset size:', x_data.shape)

def fetch_data(X=x_data, Y=y_data, num_consec=NUM_CONSEC, batch_size=BATCH_SIZE):
    print('Initialized generator stream!')
    L = x_data.shape[0] - num_consec + 1
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

r = os.path.abspath(os.path.join(dataDir + '-test'))
test_path = glob(os.path.join(r, '*.npy'))
test_num = len(test_path) // 2
test_data_indices = list(range(1, test_num + 1))

#x_test = []
#y_test = []
#for i in tqdm(test_data_indices):
#    x_path = os.path.abspath(os.path.join(dataDir + '-test', 'x_data_' + str(i) + '.npy'))
#    y_path = os.path.abspath(os.path.join(dataDir + '-test', 'y_data_' + str(i) + '.npy'))
#
#    x_train = np.load(x_path, mmap_mode='r')
#    y_train = np.load(y_path, mmap_mode='r')
#
#    x_test.append(x_train)
#    y_test.append(y_train)
#
#    del x_train, y_train
#    gc.collect()
#
#print('Loaded!')
#
#x_test = da.concatenate(x_test, axis=0)
#y_test = da.concatenate(y_test, axis=0)
#
#print('Test dataset size:', x_test.shape)

# Create dataset generator
if grayscale:
    xshape = (BATCH_SIZE, NUM_CONSEC, HEIGHT, WIDTH)
else:
    xshape = (BATCH_SIZE, NUM_CONSEC, HEIGHT, WIDTH, 3)
yshape = (BATCH_SIZE, NUM_CONSEC, HEIGHT, WIDTH)

data_generator = fetch_data(x_data, y_data, NUM_CONSEC, BATCH_SIZE)
dataset = tf.data.Dataset.from_generator(
    lambda: data_generator,
    output_shapes=(xshape, yshape),
    output_types=(tf.float32, tf.float32)
).shuffle(buffer_size=256).prefetch(tf.data.experimental.AUTOTUNE)

# test_data_generator = fetch_data(x_test, y_test, NUM_CONSEC, BATCH_SIZE)
# test_dataset = tf.data.Dataset.from_generator(
#     lambda: test_data_generator,
#     output_shapes=(xshape, yshape),
#     output_types=(tf.float16, tf.float16)
# ).prefetch(tf.data.experimental.AUTOTUNE)

for i in range(epochs):
    print('============epoch', i+1, '================')

    # Distill teacher to student
    distiller.fit(dataset, epochs=1, steps_per_epoch=x_data.shape[0] // BATCH_SIZE)

    # Show the outcome of training data so long
    print('Estimating student and teacher test performance...')
    sTP = sTN = sFP1 = sFP2 = sFN = 0
    tTP = tTN = tFP1 = tFP2 = tFN = 0
    # Test performance on 1000 random frames
    num_samples = 50
    sample_len = 16
    length = x_data.shape[0]
    for j in tqdm(range(num_samples)):
        index = random.randint(0, length - NUM_CONSEC - sample_len)
        x_raw = x_data[index:index + NUM_CONSEC + sample_len].compute().astype('float32') / 255
        y_raw = y_data[index:index + NUM_CONSEC + sample_len].compute().astype('float32')

        x_val, y_val = slice_windows(x_raw, y_raw)

        y_pred = student.predict(x_val, batch_size=BATCH_SIZE)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype('float32')

        (tp, tn, fp1, fp2, fn) = outcome(y_pred, y_val, tol)
        sTP += tp
        sTN += tn
        sFP1 += fp1
        sFP2 += fp2
        sFN += fn

        y_pred = teacher.predict(x_val, batch_size=BATCH_SIZE)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype('float32')

        (tp, tn, fp1, fp2, fn) = outcome(y_pred, y_val, tol)
        tTP += tp
        tTN += tn
        tFP1 += fp1
        tFP2 += fp2
        tFN += fn

        del x_raw, y_raw, x_val, y_val, y_pred

    gc.collect()

    print("Outcome of training data of epoch " + str(i+1) + ":")
    print("Number of true positive:", sTP, tTP)
    print("Number of true negative:", sTN, tTN)
    print("Number of false positive FP1:", sFP1, tFP1)
    print("Number of false positive FP2:", sFP2, tFP2)
    print("Number of false negative:", sFN, tFN)

    try:
        print("Accuracy:", (sTP + sTN) / (sTP + sTN + sFP1 + sFP2 + sFN), (tTP + tTN) / (tTP + tTN + tFP1 + tFP2 + tFN))
        print("Precision:", sTP / (sTP + sFP1 + sFP2), tTP / (tTP + tFP1 + tFP2))
        print("Recall:", sTP / (sTP + sFN), tTP / (tTP + tFN))
    except:
        pass

    # Save intermediate weights during training
    if (i + 1) % 10 == 0:
        student.save(save_weights + '_' + str(i + 1), save_format='h5')

print('Saving weights......')
student.save(save_weights)
print('Done......')
