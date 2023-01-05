import numpy as np
import os
from glob import glob
import piexif
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
import keras.backend as K
from keras import optimizers
import tensorflow as tf
import random
import shutil
import cv2

BATCH_SIZE=3
HEIGHT=288
WIDTH=512
mag = 1
sigma = 2.5

#game_list = ['TaiTzuYing_vs_AkaneYAMAGUCHI_2018AllEnglandOpenMatch_Final', 'TaiTzuYing_vs_ChenYufei_2018AllEnglandOpenMatch_semiFinal']
#p = os.path.join(game_list[0], 'frame', '0_0_0', '1.png')
game_list = ['match1', 'match2', 'match3', 'match4', 'match5', 'match6', 'match7', 'match8', 'match9', 'match10', 'match11', 'match12', 'match13', 'match14', 'match15']

# First generate the frames
for game in game_list:
    gameDir = os.path.join('/home/code-base/scratch_space/data/', game)
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

exit()
p = os.path.join('/home/code-base/scratch_space/data/', game_list[0], 'frame', '1_01_00', '1.jpg')
a = img_to_array(load_img(p))
ratio = a.shape[0] / HEIGHT
dataDir = 'npy-hitnet'

if os.path.exists(dataDir):
    shutil.rmtree(dataDir)
os.makedirs(dataDir)

count = 1

for game in game_list:
    all_path = glob(os.path.join('/home/code-base/scratch_space/data/', game, 'frame', '*'))
    train_path = all_path[:int(len(all_path)*0.8)]
    # train_path = all_path[int(len(all_path)*0.8):]
    for i in range(len(train_path)):
        train_path[i] = train_path[i][len(os.path.join('/home/code-base/scratch_space/data/', game, 'frame')) + 1:]
    for p in train_path:
        labelPath = os.path.join('/home/code-base/scratch_space/data/', game, 'shot', p + '_hit.csv')
        data = pd.read_csv(labelPath)
        no = data['frame'].values
        h = data['hit'].values
        # Add a hit for the bird landing on the ground
        # 7 for good luck
        h[-7] = 1
        num = no.shape[0]
        r = os.path.join('/home/code-base/scratch_space/data/', game, 'frame', p)
        x_data_tmp = []
        y_data_tmp = []
        for i in range(num-2):
            unit = []
            for j in range(BATCH_SIZE):
                target=str(no[i+j])+'.jpg'
                png_path = os.path.join(r, target)
                a = load_img(png_path)
                a = np.moveaxis(img_to_array(a.resize(size=(WIDTH, HEIGHT))), -1, 0)
                unit.append(a[0])
                unit.append(a[1])
                unit.append(a[2])
                del a
            x_data_tmp.append(unit)
            del unit

            unit = []
            for j in range(BATCH_SIZE):
                unit.append(h[i+j])
            y_data_tmp.append(unit) 
            del unit

        x_data_tmp2 = np.asarray(x_data_tmp)
        del x_data_tmp
        x_data = x_data_tmp2.astype('float32')
        del x_data_tmp2
        x_data=(x_data/255)

        y_data=np.asarray(y_data_tmp)
        del y_data_tmp
        np.save(os.path.join(dataDir, 'x_data_' + str(count) + '.npy'), x_data)
        np.save(os.path.join(dataDir, 'y_data_' + str(count) + '.npy'), y_data)
        print('============================')
        print(count)
        print(game, p)
        print(x_data.shape)
        print(y_data.shape)
        print('============================')
        del x_data
        del y_data
        count += 1
