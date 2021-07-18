import cv2
import csv
import os
import sys
import shutil
from glob import glob
game = 'game1'
p = os.path.join(game, 'rally_video', '*mp4')
video_list = glob(p)
os.makedirs(game + '/frame/')
for videoName in video_list:
	rallyName = videoName[len(os.path.join(game, 'rally_video'))+1:-4]
	outputPath = os.path.join(game, 'frame', rallyName)
	outputPath += '/'
	os.makedirs(outputPath)
	cap = cv2.VideoCapture(videoName)
	success, count = True, 0
	success, image = cap.read()
	while success:
		cv2.imwrite(outputPath + '%d.png' %(count), image)
		count += 1
		success, image = cap.read()

