import sys
import getopt
import numpy as np
import os
from glob import glob
import piexif
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from TrackNet import TrackNet
import keras.backend as K
from keras import optimizers
import tensorflow as tf
import cv2
from os.path import isfile, join
from PIL import Image
import time
BATCH_SIZE=1
HEIGHT=288
WIDTH=512
sigma=2.5
mag=1

def genHeatMap(w, h, cx, cy, r, mag):
	if cx < 0 or cy < 0:
		return np.zeros((h, w))
	x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
	heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
	heatmap[heatmap <= r**2] = 1
	heatmap[heatmap > r**2] = 0
	return heatmap*mag

#time: in milliseconds
def custom_time(time):
	remain = int(time / 1000)
	ms = (time / 1000) - remain
	s = remain % 60
	s += ms
	remain = int(remain / 60)
	m = remain % 60
	remain = int(remain / 60)
	h = remain
	#Generate custom time string
	cts = ''
	if len(str(h)) >= 2:
		cts += str(h)
	else:
		for i in range(2 - len(str(h))):
			cts += '0'
		cts += str(h)
	
	cts += ':'

	if len(str(m)) >= 2:
		cts += str(m)
	else:
		for i in range(2 - len(str(m))):
			cts += '0'
		cts += str(m)

	cts += ':'

	if len(str(int(s))) == 1:
		cts += '0'
	cts += str(s)

	return cts


try:
	(opts, args) = getopt.getopt(sys.argv[1:], '', [
		'video_name=',
		'load_weights='
	])
	if len(opts) != 2:
		raise ''
except:
	print('usage: python3 predict.py --video_name=<videoPath> --load_weights=<weightPath>')
	exit(1)

for (opt, arg) in opts:
	if opt == '--video_name':
		videoName = arg
	elif opt == '--load_weights':
		load_weights = arg
	else:
		print('usage: python3 predict.py --video_name=<videoPath> --load_weights=<weightPath>')
		exit(1)

#Loss function
def custom_loss(y_true, y_pred):
	loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
	return K.mean(loss)

model = load_model(load_weights, custom_objects={'custom_loss':custom_loss})

print('Beginning predicting......')

start = time.time()

f = open(videoName[:-4]+'_predict.csv', 'w')
f.write('Frame,Visibility,X,Y,Time\n')

cap = cv2.VideoCapture(videoName)

success, image1 = cap.read()
success, image2 = cap.read()
success, image3 = cap.read()

ratio = image1.shape[0] / HEIGHT

size = (int(WIDTH*ratio), int(HEIGHT*ratio))
fps = 30

if videoName[-3:] == 'avi':
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
elif videoName[-3:] == 'mp4':
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
else:
	print('usage: video type can only be .avi or .mp4')
	exit(1)

out = cv2.VideoWriter(videoName[:-4]+'_predict'+videoName[-4:], fourcc, fps, size)

out.write(image1)
out.write(image2)

count = 2

while success:
	unit = []
	#Adjust BGR format (cv2) to RGB format (PIL)
	x1 = image1[...,::-1]
	x2 = image2[...,::-1]
	x3 = image3[...,::-1]
	#Convert np arrays to PIL images
	x1 = array_to_img(x1)
	x2 = array_to_img(x2)
	x3 = array_to_img(x3)
	#Resize the images
	x1 = x1.resize(size = (WIDTH, HEIGHT))
	x2 = x2.resize(size = (WIDTH, HEIGHT))
	x3 = x3.resize(size = (WIDTH, HEIGHT))
	#Convert images to np arrays and adjust to channels first
	x1 = np.moveaxis(img_to_array(x1), -1, 0)		
	x2 = np.moveaxis(img_to_array(x2), -1, 0)		
	x3 = np.moveaxis(img_to_array(x3), -1, 0)
	#Create data
	unit.append(x1[0])
	unit.append(x1[1])
	unit.append(x1[2])
	unit.append(x2[0])
	unit.append(x2[1])
	unit.append(x2[2])
	unit.append(x3[0])
	unit.append(x3[1])
	unit.append(x3[2])
	unit=np.asarray(unit)	
	unit = unit.reshape((1, 9, HEIGHT, WIDTH))
	unit = unit.astype('float32')
	unit /= 255
	y_pred = model.predict(unit, batch_size=BATCH_SIZE)
	y_pred = y_pred > 0.5
	y_pred = y_pred.astype('float32')
	h_pred = y_pred[0]*255
	h_pred = h_pred.astype('uint8')
	frame_time = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
	if np.amax(h_pred) <= 0:
		f.write(str(count)+',0,0,0,'+frame_time+'\n')
		out.write(image3)
	else:	
		#h_pred
		(cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		rects = [cv2.boundingRect(ctr) for ctr in cnts]
		max_area_idx = 0
		max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
		for i in range(len(rects)):
			area = rects[i][2] * rects[i][3]
			if area > max_area:
				max_area_idx = i
				max_area = area
		target = rects[max_area_idx]
		(cx_pred, cy_pred) = (int(ratio*(target[0] + target[2] / 2)), int(ratio*(target[1] + target[3] / 2)))

		f.write(str(count)+',1,'+str(cx_pred)+','+str(cy_pred)+','+frame_time+'\n')
		image3_cp = np.copy(image3)
		cv2.circle(image3_cp, (cx_pred, cy_pred), 5, (0,0,255), -1)
		out.write(image3_cp)
	image1 = image2
	image2 = image3
	success, image3 = cap.read()
	count += 1

f.close()
out.release()
end = time.time()
print('Prediction time:', end-start, 'secs')
print('Done......')
