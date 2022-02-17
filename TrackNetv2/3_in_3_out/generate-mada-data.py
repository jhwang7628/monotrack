import numpy as np
import os
import piexif

from tqdm import tqdm
from glob import glob
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import *
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from constants import *
from utils import *

import random
import shutil
import pandas as pd
import cv2
import json

gen_frames = False
root_folder = '/home/code-base/scratch_space/mada_videos'
game_list = []
for folder in ['behind-the-court', 'bleacher', 'side']:
    game_folder = f'{root_folder}/{folder}'
    print(folder)
    for match in tqdm(os.listdir(game_folder)):
        match_folder = f'{game_folder}/{match}'
        for rally in os.listdir(match_folder):
            if '.' not in rally:
                continue
            rallyName, ext = os.path.splitext(rally)
            rallyPrefix = f'{match_folder}/{rallyName}'
            game_list.append((rallyPrefix, folder, match, rallyName))
            
            if gen_frames:
                frameDir = os.path.join(rallyPrefix, 'frame')
                os.makedirs(frameDir, exist_ok=True)

                vidcap = cv2.VideoCapture(f'{match_folder}/{rally}')
                success, image = vidcap.read()
                count = 0
                while success:
                  cv2.imwrite(os.path.join(frameDir, "%d.jpg" % count), image)     # save frame as JPEG file      
                  success,image = vidcap.read()
                  count += 1
            
dataDir = 'npy-color-mada'
# if os.path.exists(dataDir):
#     shutil.rmtree(dataDir)
os.makedirs(dataDir, exist_ok=True)
            
def augment(image, params):
#     image = elastic_transform(image, image.shape[1] * 2, image.shape[1] * 0.08, image.shape[1] * 0.08, params['seed'])
    image = augmenter.apply_transform(image, params)
    return image

augmenter = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest')

def create_data(image, params):
    scale = 2
    W, H = image.shape[1] // scale, image.shape[0] // scale
    image = resize(image, (H, W))
    
    # Factor of 2 to compensate for eventual rescaling
    mask = genHeatMap(W, H, params['xy'][0] / scale, params['xy'][1] / scale, sigma) if params['vis'] else np.zeros((H, W))
    mask = np.expand_dims(mask, axis=-1)
    
    # Generate a random augmentation
    if params['augment']:
        result = augment(np.concatenate([image, mask], axis=-1), params)
        image, mask = result[:, :, :3], result[:, :, 3]
    
    # Resize the images
    image = resize(image, (HEIGHT, WIDTH))
    mask = np.squeeze(resize(mask, (HEIGHT, WIDTH)))
    if grayscale:
        image = np.average(image, axis=-1)
    return image, mask

count = 1

game_list = list(set(game_list))
for game in tqdm(game_list):
    train_path = glob(os.path.join(game[0], 'frame', '*.jpg'))
    no = list(range(len(train_path)))    
    images = [load_img(path) for path in train_path]
    
    x_data_tmp, y_data_tmp = [], []
    for i, image in enumerate(images):
        subpath = '%s/%s/%s/%s_%d.json' % (game[1], game[2], game[3], game[3], i+1)
        try:
            labels = json.load(open(f'/home/code-base/scratch_space/mada_data/{subpath}', 'r'))
            labels = labels['points']

            ratiox, ratioy = image.size[0] / WIDTH, image.size[1] / HEIGHT
            params = {}
            params['vis'] = any('球中心' in point['label'] for point in labels)
            params['augment'] = False
            if params['vis']:
                point = None
                for p in labels:
                    if '球中心' in p['label']:
                        point = p['position']
                        break
                params['xy'] = tuple(point)
            else:
                params['xy'] = (0, 0)

            image = img_to_array(image)
            a, b = create_data(image, params)

            x_data_tmp.append(a)
            y_data_tmp.append(b)

            del a
            del b
        except:
            print(f'Cannot find {subpath}.')

    x_data = np.asarray(x_data_tmp).astype('uint8')
    y_data = np.asarray(y_data_tmp).astype('bool')

    np.save(os.path.join(dataDir, 'x_data_' + str(count) + '.npy'), x_data)
    np.save(os.path.join(dataDir, 'y_data_' + str(count) + '.npy'), y_data)

#     print('============================')
#     print(count)
#     print(game)
#     print(x_data.shape)
#     print(y_data.shape)
#     print('============================')
    count += 1

    del x_data_tmp
    del y_data_tmp
    del x_data
    del y_data
    del images