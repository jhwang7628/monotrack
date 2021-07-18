import numpy as np
import cv2
import sys
import os
import pickle
import pandas as pd
video_name = sys.argv[1]
# name="WS_TAITzuYing_vs_CHENYuFei"
# ext=".mp4"
# filename=name+ext
filename = video_name.split('.',1)[0]
data=dict()
racket = dict()
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global data,cap,current
	global image
	# if event == cv2.EVENT_LBUTTONDOWN:
		# if current in racket:
		# 	if len(racket[current]) < 2:
		# 		racket[current].append((x,y))
		# 		print('\nRacket coordinate ', str(current)+':',str(racket[current]))
		# 	else:
		# 		print('\nYou had clicked two racket points!!\n')
		# else:
		# 	racket[current] = list()
		# 	racket[current].append((x,y))
		# 	print('\nRacket coordinate ',str(current)+':  '+str(x)+', '+str(y))
		#
		# image=toframe(cap,current,total_frame)
	if event == cv2.EVENT_LBUTTONDOWN:
		data[current] = (x,y)
		image=toframe(cap,current,total_frame)
def toframe(cap,n,total_frame):
	print('current frame: ',n)
	cap.set(cv2.CAP_PROP_POS_FRAMES,n); 
	ret, frame = cap.read()
	if not ret:
		return None
	else:
		
		if current in data:
			cv2.circle(frame, data[current], 3, (0,0,255),thickness=-1)
		if current in racket:
			for i in range(len(racket[current])):
				if i ==0:
					cv2.circle(frame, racket[current][0], 3, (255,0,0),thickness=-1)
				else:
					cv2.circle(frame, racket[current][1], 3, (0,255,0),thickness=-1)

		return frame

total_frame=0
cap = cv2.VideoCapture(video_name)
total_frame=cap.get(cv2.CAP_PROP_FRAME_COUNT)
print ("Total frame : "+str(total_frame))


current=0
image=toframe(cap,current,total_frame)
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
saved=False

try:
	data,racket=pickle.load(open(filename+".pkl",'rb'))
	print ("loaded from "+filename+".pkl")
	if max(data.keys()) > max(racket.keys()):
		print ("min frame ", str(min(data.keys())))
		print ("max frame ", str(max(data.keys())))
		print ("jump to max frame")
		current=max(data.keys())
	else:
		print ("min frame ", str(min(racket.keys())))
		print ("max frame ", str(max(racket.keys())))
		print ("jump to max frame")
		current=max(racket.keys())
	image=toframe(cap,current,total_frame)
except Exception as e:
	print ('\nThis is new video! Good Luck!!')
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
	
	if key == ord("w"):		# delete ball point
		if current in data:
			del data[current]
			print('\nYou delete the ball coordinate.')
			image=toframe(cap,current,total_frame)
		else:
			print('\nNo ball coordinate!!')
	elif key == ord("f"):
		current=int(input('Enter your frame:'))
		image=toframe(cap,current,total_frame)
	elif key == ord("n"):     #jump next 30 frames
		check = current+30
		if check < total_frame-1:
			current+=30
			
		else:
			current = total_frame-1
			print('\nThis is last frame.')
		image=toframe(cap,current,total_frame)
	elif key == ord("p"):     #jump last 30 frames
		check = current-30
		if check <= 0:
			print('\nInvaild !!! Jump to first image...')
			current = 0
		else:
			current = check
		image=toframe(cap,current,total_frame)
	elif key == ord("d"):     #jump next frame
		if current < total_frame-1:
			current+=1
			image=toframe(cap,current,total_frame)
		else:
			print('\nCongrats! This is the last frame!!')
	elif key == ord("e"):     #jump last frame
		if current == 0:
			print('\nThis is first images')
		else:
			current-=1
		image=toframe(cap,current,total_frame)
	elif key == ord("s"):     #save as .pkl
		saved = True
		try:
			pickle.dump([data,racket],open(filename+".pkl",'wb'))
			
			print ("saved to "+filename+".pkl")
		except Exception as e:
			print (str(e))
	# if key == ord("l"):     #load .pkl and jump to last label
	# 	try:
	# 		data=pickle.load(open(filename+".pkl",'rb'))
	# 		print ("loaded from "+filename+".pkl")
	# 		print ("min frame ", str(min(data.keys())))
	# 		print ("max frame ", str(max(data.keys())))
	# 		print ("jump to max frame")
	# 		current=max(data.keys())
	# 		image=toframe(cap,current,total_frame)
	# 	except Exception as e:
	# 		print ("ERRORRRR!!!:",str(e))
	# if the 'c' key is pressed, break from the loop
	elif key == ord("q"):
		if saved:
			break
		else:
			print('\nYou DONT save the data!!')
			print('You DONT save the data!!')

matchName = filename.split('/',1)[0]
rallyName = filename.split('/',1)[1]
if not os.path.exists(matchName+'_labels'):
    os.mkdir(matchName+'_labels')
    os.mkdir(matchName+'_labels/Ball')
#    os.mkdir(matchName+'_labels/Racket')

# close all open windows
outputfile_name1 = matchName+'_labels/Ball/'+rallyName+'_ball.csv'
#outputfile_name2 = matchName+'_labels/Racket/'+rallyName+'_racket.csv'

with open(outputfile_name1,'w') as outputfile:
	for i in range(int(total_frame)):
		if i in data:
			outputfile.write(str(i)+","+str(data[i][0])+","+str(data[i][1])+"\n")

#with open(outputfile_name2,'w') as outputfile:
#	for i in range(int(total_frame)):
#		if i in racket:
#			if len(racket[i])==1:
#				outputfile.write(str(i)+","+str(racket[i][0][0])+","+str(racket[i][0][1])+"\n")
#			else:
#				outputfile.write(str(i)+","+str(racket[i][0][0])+","+str(racket[i][0][1])+","+str(racket[i][1][0])+","+str(racket[i][1][1])+"\n")

Frame=[]
X=[]
Y=[]
Cov = pd.read_csv(outputfile_name1,sep=',',names=["index", "x", "y"])
for i in range((Cov['index']).shape[0]):
	Frame.append(int(Cov['index'][i]))
	X.append(int(Cov['x'][i]))
	Y.append(int(Cov['y'][i]))
Visibility=[1 for _ in range(len(Frame))]
df_label = pd.DataFrame(columns=['Frame', 'Visibility', 'X', 'Y'])
df_label['Frame'], df_label['Visibility'], df_label['X'], df_label['Y'] = Frame, Visibility, X, Y
#Compensate the non-labeled frames due to no visibility of badminton
for i in range(0, Frame[-1]+1):
	if i in list(df_label['Frame']):
		pass
	else:
		df_label = df_label.append(pd.DataFrame(data = {'Frame':[i], 'Visibility':[0], 'X':[0], 'Y':[0]}), ignore_index=True)

#Sorting by 'Frame'
df_label = df_label.sort_values(by=['Frame'])
df_label.to_csv(outputfile_name1, encoding='utf-8', index=False)

cv2.destroyAllWindows()
cap.release()
