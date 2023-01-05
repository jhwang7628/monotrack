import numpy as np
import cv2
import sys
import os
import pickle
from collections import defaultdict

dataset_folder = sys.argv[1]
matches = ['match' + str(i) for i in range(1, 27)] + ['test_match' + str(i) for i in range(1,4)]

data = defaultdict(list)
current = None
cap = None

data["info"] = "Court points are labelled from left to right, bottom to top."

def click_and_label(event, x, y, flags, param):
	# grab references to the global variables
	global data, cap, current
	global image

	if event == cv2.EVENT_LBUTTONDOWN:
		data[current].append((x,y))
		image = cv2.circle(image, (x, y), 5, (0, 0, 255), -1)


def save():
	global data
	with open("courts.pkl", "wb") as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_label)
for match in matches:
	rally_dir = f"{dataset_folder}/{match}/rally_video"
	done = False
	for rally in os.listdir(rally_dir):
		try:
			print('Trying', match, rally.split('.')[0])
			cap = cv2.VideoCapture(f"{rally_dir}/{rally}")
			current = (match, rally)
			_, image = cap.read()

			while True:
				cv2.imshow("image", image)
				key = cv2.waitKey(1) & 0xFF
				if key == ord("s"):
					cap.release()
					print(data)
					print('saving...')
					save()
					done = True
					break
				elif key == ord("n"):     #jump next 30 frames
					for i in range(30):
						_, image = cap.read()

			print('Reached here')
			break
		except:
			pass

save()

cv2.destroyAllWindows()
