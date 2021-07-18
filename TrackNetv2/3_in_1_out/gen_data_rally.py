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
BATCH_SIZE=2
HEIGHT=288
WIDTH=512
mag = 1
sigma = 2.5

def genHeatMap(w, h, cx, cy, r, mag):
	if cx < 0 or cy < 0:
		return np.zeros((h, w))
	x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
	heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
	heatmap[heatmap <= r**2] = 1
	heatmap[heatmap > r**2] = 0
	return heatmap*mag

#game_list = ['TaiTzuYing_vs_AkaneYAMAGUCHI_2018AllEnglandOpenMatch_Final', 'TaiTzuYing_vs_ChenYufei_2018AllEnglandOpenMatch_semiFinal']
#p = os.path.join(game_list[0], 'frame', '0_0_0', '1.png')
game_list = ['match1', 'match2', 'match3', 'match4', 'match5', 'match6', 'match7', 'match8', 'match9', 'match10', 'match11', 'match12', 'match13', 'match14', 'match15']
p = os.path.join(game_list[0], 'frame', '1_01_00', '1.png')
a = img_to_array(load_img(p))
ratio = a.shape[0] / HEIGHT

dataDir = 'npy'

if os.path.exists(dataDir):
    shutil.rmtree(dataDir)
os.makedirs(dataDir)

count = 1

for game in game_list:
	all_path = glob(os.path.join(game, 'frame', '*'))
	train_path = all_path[:int(len(all_path)*0.8)]
	#train_path = all_path[int(len(all_path)*0.8):]
	for i in range(len(train_path)):
		train_path[i] = train_path[i][len(os.path.join(game, 'frame')) + 1:]
	for p in train_path:
		labelPath = os.path.join(game, 'ball_trajectory', p + '_ball.csv')
		data = pd.read_csv(labelPath)
		no = data['Frame'].values
		v = data['Visibility'].values
		x = data['X'].values
		y = data['Y'].values
		num = no.shape[0]
		r = os.path.join(game, 'frame', p)
		x_data_tmp = []
		y_data_tmp = []
		for i in range(num-2):
			unit = []
			for j in range(3):
				target=str(no[i+j])+'.png'
				png_path = os.path.join(r, target)
				a = load_img(png_path)
				a = np.moveaxis(img_to_array(a.resize(size=(WIDTH, HEIGHT))), -1, 0)
				unit.append(a[0])
				unit.append(a[1])
				unit.append(a[2])
				del a
			x_data_tmp.append(unit)
			del unit
			if v[i+2] == 0:
				y_data_tmp.append(genHeatMap(WIDTH, HEIGHT, -1, -1, sigma, mag))
			else:
				y_data_tmp.append(genHeatMap(WIDTH, HEIGHT, int(x[i+2]/ratio), int(y[i+2]/ratio), sigma, mag))
        
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
