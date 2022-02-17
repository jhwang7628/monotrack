import sys
import getopt
import numpy as np
import os
from glob import glob
import piexif
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import *
import cv2
from os.path import isfile, join
from PIL import Image
import time
from constants import *
from utils import *

try:
    (opts, args) = getopt.getopt(sys.argv[1:], '', [
        'video_name=',
        'load_weights='
    ])
    if len(opts) != 2:
        raise ''
except:
    print('usage: python3 predict-regressor.py --video_name=<videoPath> --load_weights=<weightPath>')
    exit(1)

for (opt, arg) in opts:
    if opt == '--video_name':
        videoName = arg
    elif opt == '--load_weights':
        load_weights = arg
    else:
        print('usage: python3 predict-regressor.py --video_name=<videoPath> --load_weights=<weightPath>')
        exit(1)

model = load_model(load_weights)

print('Beginning predicting......')

start = time.time()

f = open(videoName[:-4]+'_predict.csv', 'w')
f.write('Frame,Visibility,X,Y,Time\n')

cap = cv2.VideoCapture(videoName)

def read_frame():
    flag, image = cap.read()
    timestamp = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
    return flag, image, timestamp

success, images, frame_time = [], [], []
for i in range(NUM_CONSEC):
    s, im, t = read_frame()
    success.append(s)
    images.append(im)
    frame_time.append(t)

ratioy = images[0].shape[0] / HEIGHT
ratiox = images[0].shape[1] / WIDTH

size = (int(WIDTH*ratiox), int(HEIGHT*ratioy))
fps = cap.get(cv2.CAP_PROP_FPS)

if videoName[-3:] == 'avi':
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
elif videoName[-3:] == 'mp4':
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
else:
    print('usage: video type can only be .avi or .mp4')
    exit(1)

out = cv2.VideoWriter(videoName[:-4]+'_predict'+videoName[-4:], fourcc, fps, size)

print('About to begin prediction...')

count = 0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
from tqdm import tqdm
pbar = tqdm(total=total)

while True:
    unit = []
    # Adjust BGR format (cv2) to RGB format (PIL)
    for i in range(NUM_CONSEC):
        image = cv2.resize(images[i], dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        if grayscale:
            xi = np.average(image, axis=-1)
        else:
            xi = image
        unit.append(xi)
        
    unit = np.asarray(unit)	
    if grayscale:
        unit = unit.reshape((1, NUM_CONSEC, HEIGHT, WIDTH))
    else:
        unit = unit.reshape((1, NUM_CONSEC, HEIGHT, WIDTH, 3))
    unit = unit.astype('float32')
    unit /= 255
    
    y_pred = model.predict(unit, batch_size=1)[0][0]
    for i in range(NUM_CONSEC):
        if np.max(y_pred[i]) <= 0.05:
            f.write(str(count)+',0,0,0,'+frame_time[i]+'\n')
            out.write(images[i])
        else:
            cx_pred, cy_pred = int(images[i].shape[1] * y_pred[i][0]), int(images[i].shape[0] * y_pred[i][1])
            f.write(str(count)+',1,'+str(cx_pred)+','+str(cy_pred)+','+frame_time[i]+'\n')
            image_cp = np.copy(images[i])
            cv2.circle(image_cp, (cx_pred, cy_pred), 5, (0,0,255), -1)
            out.write(image_cp)
        count += 1
    
    success, images, frame_time = [], [], []
    for i in range(NUM_CONSEC):
        try:
            s, im, t = read_frame()
            success.append(s)
            images.append(im)
            frame_time.append(t)
        except:
            pass
    
    if len(success) < NUM_CONSEC or not success[-1]:
        break
        
    pbar.n = count
    pbar.last_print_n = count
    pbar.refresh()

f.close()
out.release()
end = time.time()
print('Prediction time:', end-start, 'secs')
print('Done......')
