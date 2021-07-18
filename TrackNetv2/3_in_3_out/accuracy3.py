import sys
import getopt
import numpy as np
import math
from keras.models import *
from keras.layers import *
import os
from glob import glob
import piexif
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from TrackNet3 import TrackNet3
import keras.backend as K
from keras import optimizers
from keras.activations import *
import tensorflow as tf
import cv2
#Size of y_pred and y_true: batch*288*512
BATCH_SIZE=1
HEIGHT=288
WIDTH=512
sigma = 2.5
mag = 1

def genHeatMap(w, h, cx, cy, r, mag):
	if cx < 0 or cy < 0:
		return np.zeros((h, w))
	x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
	heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
	heatmap[heatmap <= r**2] = 1
	heatmap[heatmap > r**2] = 0
	return heatmap*mag

#Return the numbers of true positive, true negative, false positive and false negative
def outcome(y_pred, y_true, tol):
	n = y_pred.shape[0]
	i = 0
	TP = TN = FP1 = FP2 = FN = 0
	while i < n:
		for j in range(3):
			if np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) == 0:
				TN += 1
			elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) == 0:
				FP2 += 1
			elif np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) > 0:
				FN += 1
			elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) > 0:
				h_pred = y_pred[i][j] * 255
				h_true = y_true[i][j] * 255
				h_pred = h_pred.astype('uint8')
				h_true = h_true.astype('uint8')
				#h_pred
				(cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				rects = [cv2.boundingRect(ctr) for ctr in cnts]
				max_area_idx = 0
				max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
				for j in range(len(rects)):
					area = rects[j][2] * rects[j][3]
					if area > max_area:
						max_area_idx = j
						max_area = area
				target = rects[max_area_idx]
				(cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

				#h_true
				(cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				rects = [cv2.boundingRect(ctr) for ctr in cnts]
				max_area_idx = 0
				max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
				for j in range(len(rects)):
					area = rects[j][2] * rects[j][3]
					if area > max_area:
						max_area_idx = j
						max_area = area
				target = rects[max_area_idx]
				(cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
				dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
				if dist > tol:
					FP1 += 1
				else:
					TP += 1
		i += 1
	return (TP, TN, FP1, FP2, FN)

#Return the values of accuracy, precision and recall
def evaluation(y_pred, y_true, tol):
	(TP, TN, FP1, FP2, FN) = outcome(y_pred, y_true, tol)
	try:
		accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
	except:
		accuracy = 0
	try:
		precision = TP / (TP + FP1 + FP2)
	except:
		precision = 0
	try:
		recall = TP / (TP + FN)
	except:
		recall = 0
	return (accuracy, precision, recall)

try:
	(opts, args) = getopt.getopt(sys.argv[1:], '', [
		'load_weights=',
		'dataDir=',
		'tol='
	])
	if len(opts) != 3:
		raise ''
except:
	print('usage: python3 accuracy3.py --load_weights=<weightPath> --dataDir=<npyDataDirectory> --tol=<toleranceValue>')
	exit(1)

load_weights = ''
dataDir = ''
tol = 4

for (opt, arg) in opts:
	if opt == '--load_weights':
		load_weights = arg
	elif opt == '--dataDir':
		dataDir = arg
	elif opt == '--tol':
		tol = int(arg)
	else:
		print('usage: python3 accuracy3.py --load_weights=<weightPath> --dataDir=<npyDataDirectory> --tol=<toleranceValue>')
		exit(1)

#Loss function
def custom_loss(y_true, y_pred):
	loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
	return K.mean(loss)


model = load_model(load_weights, custom_objects={'custom_loss':custom_loss})


r = os.path.abspath(os.path.join(dataDir))
path = glob(os.path.join(r, '*.npy'))
num = len(path) / 2
i = 1
TP = TN = FP1 = FP2 = FN = 0
print('Beginning evaluating......')
print('==========================================================')
while i <= num:
	x = np.load(os.path.abspath(os.path.join(dataDir, 'x_data_' + str(i) + '.npy')))
	y = np.load(os.path.abspath(os.path.join(dataDir, 'y_data_' + str(i) + '.npy')))
	y_pred = model.predict(x, batch_size=BATCH_SIZE)
	y_pred = y_pred > 0.5
	y_pred = y_pred.astype('float32')
	(tp, tn, fp1, fp2, fn) = outcome(y_pred, y, tol)
	print('Finish evaluating data' + str(i) + ':(TP, TN, FP1, FP2, FN)=' + str((tp, tn, fp1, fp2, fn)))
	TP += tp
	TN += tn
	FP1 += fp1
	FP2 += fp2
	FN += fn
	del x
	del y
	del y_pred
	i += 1
print('==========================================================')

try:
	accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
except:
	accuracy = 0
try:
	precision = TP / (TP + FP1 + FP2)
except:
	precision = 0
try:
	recall = TP / (TP + FN)
except:
	recall = 0

print("Number of true positive:", TP)
print("Number of true negative:", TN)
print("Number of false positive FP1:", FP1)
print("Number of false positive FP2:", FP2)
print("Number of false negative:", FN)
print("accuracy:", accuracy)	
print("precision:", precision)
print("recall:", recall)
print('Done......')
