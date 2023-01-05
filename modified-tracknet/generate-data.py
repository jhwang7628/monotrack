import numpy as np
import os

from tqdm import tqdm
from glob import glob
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras import optimizers
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from constants import *
from utils import *

import random
import shutil
import pandas as pd
import cv2
import getopt
import sys

try:
    (opts, args) = getopt.getopt(sys.argv[1:], '', [
        'type=',
    ])
    if len(opts) < 1:
        raise ''
except:
    print('usage: python3 generate-data.py --type=["TRAIN", "DISTILLATION", or "TEST"]')
    exit(1)
    
data_dir = '/home/work_space/data'
gen_type = None
for (opt, arg) in opts:
    if opt == '--type':
        gen_type = arg
        
if gen_type == "TRAIN":
    NUM_AUGMENTATIONS = 6
    outputDir = 'npy-color-augmented'
    game_list = [f'{data_dir}/match' + str(i) for i in range(1, 27)]
elif gen_type == "DISTILLATION":
    NUM_AUGMENTATIONS = 1
    outputDir = 'npy-color-no-augmented'
    game_list = [f'{data_dir}/match' + str(i) for i in range(1, 27)]
elif gen_type == "TEST":
    NUM_AUGMENTATIONS = 1
    outputDir = 'npy-color-test'
    game_list = [f'{data_dir}/test_match' + str(i) for i in range(1, 4)]
else:
    print('usage: python3 generate-data.py --type=["TRAIN",  "DISTILLATION", or "TEST"]')
    exit(1)
    

print(game_list)

os.makedirs(outputDir, exist_ok=True)

gen_frames = True
if gen_frames:
    # First generate the frames
    for game in game_list:
        print(game)
        gameDir = os.path.join(data_dir, game)
        frameDir = os.path.join(gameDir, 'frame')
        rallyDir = os.path.join(gameDir, 'rally_video')
        os.makedirs(frameDir, exist_ok=True)
        for vidfile in os.listdir(rallyDir):
            vidname, _ = os.path.splitext(vidfile)
            vidDir = os.path.join(frameDir, vidname)
            os.makedirs(vidDir, exist_ok=True)

            print(os.path.join(rallyDir, vidfile))
            vidcap = cv2.VideoCapture(os.path.join(rallyDir, vidfile))
            success, image = vidcap.read()
            count = 0
            while success:
              cv2.imwrite(os.path.join(vidDir, "%d.jpg" % count), image)     # save frame as JPEG file      
              success,image = vidcap.read()
              count += 1

            
def augment(image, params):
    if gen_type == "TEST" or gen_type == "DISTILLATION":
        return image
    return augmenter.apply_transform(image, params)

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

for game in game_list:
    train_path = glob(os.path.join(game, 'frame', '*'))
    
    for i in range(len(train_path)):
        train_path[i] = train_path[i][len(os.path.join(game, 'frame')) + 1:]
    for p in train_path:
        print('Processing frames...')
        labelPath = os.path.join(game, 'ball_trajectory', p + '_ball.csv')
        data = pd.read_csv(labelPath)
        no = data['Frame'].values
        v = data['Visibility'].values
        x = data['X'].values
        y = data['Y'].values
        r = os.path.join(game, 'frame', p)
        num = no.shape[0]
        images = [load_img(os.path.join(r, str(i) + '.jpg')) for i in no]
        ratiox, ratioy = images[0].size[0] / WIDTH, images[0].size[1] / HEIGHT
        print('Done loading frames...')
        
        for t in range(NUM_AUGMENTATIONS):
            x_data_tmp = []
            y_data_tmp = []

            params = augmenter.get_random_transform((images[0].size[1], images[0].size[0]))
            params['augment'] = t
            params['seed'] = random.randint(1, 100)
            for i in tqdm(range(num)):
                params['vis'] = v[i]
                params['xy'] = (x[i], y[i])
                image = img_to_array(images[no[i]])
                a, b = create_data(image, params)

                x_data_tmp.append(a)
                y_data_tmp.append(b)

                del a, image
                del b

            x_data = np.asarray(x_data_tmp).astype('uint8')
            y_data = np.asarray(y_data_tmp).astype('bool')

            np.save(os.path.join(outputDir, 'x_data_' + str(count) + '.npy'), x_data)
            np.save(os.path.join(outputDir, 'y_data_' + str(count) + '.npy'), y_data)

            print('============================')
            print(count)
            print(game, p)
            print(x_data.shape)
            print(y_data.shape)
            print('============================')
            count += 1

            del x_data_tmp
            del y_data_tmp
            del x_data
            del y_data
        del images
