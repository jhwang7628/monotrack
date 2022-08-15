import numpy as np
import shutil
import os
import cv2
import random

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage.transform import resize
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras import optimizers
from glob import glob

from constants import *
from utils import *
from tracknet_improved import *

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
if use_mp:
    policy = mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.experimental.set_policy(policy)

    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

# Create training data frames and label frames

save_weights = 'test_model'
epochs = 1
augment_prob = 0.5
train_games = ['/home/code-base/scratch_space/data/match' + str(i) for i in range(1, 24)]
test_games = ['/home/code-base/scratch_space/data/test_match' + str(i) for i in range(1, 4)]

def slice_windows(X):
    index = ( np.expand_dims(np.arange(NUM_CONSEC), 0) + 
              np.expand_dims(np.arange(len(X)-NUM_CONSEC), 0).T )
    return X[index]

def create_meta(games):
    data = []
    for game in games:
        gameDir = os.path.join('/home/code-base/scratch_space/data/', game)
        frameDir = os.path.join(gameDir, 'frame')
        labelDir = os.path.join(gameDir, 'ball_trajectory')
        for rally in os.listdir(frameDir):
            if '.' in rally:
                continue
            labelFile = os.path.join(labelDir, rally + '_ball.csv')
            df = pd.read_csv(labelFile)
            df['file'] = [os.path.join(frameDir, rally, str(f) + '.jpg') for f in df['Frame']]
            df.drop(columns=['Frame'], inplace=True)
            data.append(df)
    meta = np.vstack(data)
    return slice_windows(meta)

train_data = create_meta(train_games)
test_data = create_meta(test_games)

augmenter = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

def as_string(tf_string):
    return tf_string.numpy().decode('ascii')

def augment(image, params):
    image = augmenter.apply_transform(image, params)
    return image

def parse_record(labels, filenames):
    images = [img_to_array(load_img(name)) for name in filenames]
    W, H = images[0].shape[1], images[0].shape[0]
    mask_gen = lambda vis, xy: genHeatMap(W, H, xy[0], xy[1], 2*sigma) if vis else np.zeros((H, W))
    
    # Factor of 2 to compensate for eventual rescaling
    masks = [np.expand_dims(mask_gen(labels[i, 0], labels[i, 1:3]), axis=-1) for i in range(NUM_CONSEC)]
    
    # Generate a random augmentation with some probability
    if random.random() < augment_prob:
        params = augmenter.get_random_transform(images[0].shape)
        images = np.array([augment(image, params) for image in images])
        masks = np.array([augment(mask, params) for mask in masks])
    
    # Resize the images
    images = [resize(image, (HEIGHT, WIDTH)) / 255 for image in images]
    masks = [np.squeeze(resize(mask, (HEIGHT, WIDTH))) for mask in masks]
    if grayscale:
        images = [np.average(image, axis=-1) for image in images]
    return np.array(images), np.array(masks)

def fetch_data(data):
    np.random.shuffle(data)
    batchX, batchY = [], []
    for row in data:
        X, Y = parse_record(row[:, :3], row[:, 3])
        batchX.append(X)
        batchY.append(Y)
        if len(batchX) == BATCH_SIZE:
            yield np.array(batchX), np.array(batchY)
            del batchX, batchY
            batchX, batchY = [], []
        
if grayscale:
    xshape = (BATCH_SIZE, NUM_CONSEC, HEIGHT, WIDTH)
else:
    xshape = (BATCH_SIZE, NUM_CONSEC, HEIGHT, WIDTH, 3)
yshape = (BATCH_SIZE, NUM_CONSEC, HEIGHT, WIDTH)

dataset = (tf.data.Dataset.from_generator(lambda: fetch_data(train_data),
                                          output_types=(tf.float32, tf.float32),
                                          output_shapes=(xshape, yshape))
                          .prefetch(tf.data.experimental.AUTOTUNE))

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    OPT = optimizers.Adadelta(lr=1.0)
    model = TrackNetImproved(HEIGHT, WIDTH, NUM_CONSEC, grayscale)
    model.compile(loss=custom_loss, optimizer=OPT, metrics=[tf.keras.metrics.MeanIoU(num_classes=2), 'accuracy'])
#     model = tf.keras.models.load_model(load_weights, custom_objects={'custom_loss': custom_loss})

for i in range(epochs):
    print('============epoch', i+1, '================')
    
    model.fit(
        dataset, 
        epochs=1,
        steps_per_epoch=train_data.shape[0] // BATCH_SIZE
#         validation_data=test_dataset,
    )
    
    # Save intermediate weights during training
    if (i + 1) % 10 == 0:
        model.save(save_weights + '_' + str(i + 1), save_format='h5')

print('Saving weights......')
model.save(save_weights)
print('Done......')
