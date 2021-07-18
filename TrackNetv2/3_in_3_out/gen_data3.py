import sys, getopt
import shutil
import numpy as np
import os
from glob import glob
import piexif
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split
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

try:
	(opts, args) = getopt.getopt(sys.argv[1:], '', [
		'batch=',
		'label=',
		'frameDir=',
		'dataDir='
	])
	if len(opts) != 4:
		raise ''
except:
	print('usage: python3 gen_data3.py --batch=<batchSize> --label=<csvFile> --frameDir=<frameDirectory> --dataDir=<npyDataDirectory>')
	exit(1)

batch = 500
labelPath = ''
frameDir = ''
dataDir = ''

for (opt, arg) in opts:
	if opt == '--batch':
		batch = int(arg)
	elif opt == '--label':
		labelPath = arg
	elif opt == '--frameDir':
		frameDir = arg
	elif opt == '--dataDir':
		dataDir = arg
	else:
		print('usage: python3 gen_data3.py --batch=<batchSize> --label=<csvFile> --frameDir=<frameDirectory> --dataDir=<npyDataDirectory>')
		exit(1)

p = os.path.abspath(os.path.join(frameDir, '1.png'))
img = img_to_array(load_img(p))
ratio = img.shape[0] / HEIGHT

if os.path.exists(dataDir):
	shutil.rmtree(dataDir)
os.makedirs(dataDir)

data = pd.read_csv(labelPath)
no = data['Frame'].values
v = data['Visibility'].values
x = data['X'].values
y = data['Y'].values
num = no.shape[0]		
r = os.path.abspath(os.path.join(frameDir))

i = 0
ptr = 0
count = 1

#generate data and save to the disk in the format of .npy
print('Generating data......')
print('==========================================================')
while ptr < num-2:
	x_data_tmp = []
	y_data_tmp = []
	while (i < ptr+batch) and (i < num-2):
		if no[i]+2 != no[i+2]:
			i += 1
			continue
		unit = []
		for j in range(3):
			target=str(no[i+j])+'.png'
			png_path=os.path.join(r, target)
			a=load_img(png_path)
			a=np.moveaxis(img_to_array(a.resize(size=(WIDTH, HEIGHT))), -1, 0)
			#a:(3, HEIGHT, WIDTH) nparray
			unit.append(a[0])
			unit.append(a[1])
			unit.append(a[2])
			del a
		#unit:(9, HEIGHT, WIDTH) 
		x_data_tmp.append(unit)
		del unit
		
		unit = []
		for j in range(3):
			if v[i+j] == 0:
				unit.append(genHeatMap(WIDTH, HEIGHT, -1, -1, sigma, mag))
			else:
				unit.append(genHeatMap(WIDTH, HEIGHT, int(x[i+j]/ratio), int(y[i+j]/ratio), sigma, mag))
		#unit:(3, HEIGHT, WIDTH)
		y_data_tmp.append(unit)
		del unit

		i += 1

	x_data_tmp2 = np.asarray(x_data_tmp)
	del x_data_tmp
	x_data = x_data_tmp2.astype('float32')
	del x_data_tmp2
	x_data /= 255

	y_data=np.asarray(y_data_tmp)
	del y_data_tmp
	np.save(os.path.abspath(os.path.join(dataDir, 'x_data_' + str(count) + '.npy')), x_data)
	print('Finish generating x_data_' + str(count) + ' (shape:' + str(x_data.shape) + ')')
	np.save(os.path.abspath(os.path.join(dataDir, 'y_data_' + str(count) + '.npy')), y_data)
	print('Finish generating y_data_' + str(count) + ' (shape:' + str(y_data.shape) + ')')
	count += 1
	del x_data
	del y_data
	ptr = i

print('==========================================================')
print('Done......')
