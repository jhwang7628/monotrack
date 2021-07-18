import pandas as pd
import csv
import os
import sys

try:
	inputFile = sys.argv[1]
	outputFile = sys.argv[2]
	if (not inputFile) or (not outputFile):
		raise ''
except:
	print('usage: python3 Rearrange_Label.py <inputFile> <outputFile>')
	exit(1)
	
#Open previous label data
with open(inputFile) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	frames = []
	x, y = [], []
	list1 = []
	for row in readCSV:
		list1.append(row)
	for i in range(1 , len(list1)):
		frames += [int(list1[i][0])]
		x += [int(list1[i][1])]
		y += [int(list1[i][2])]
visibility = [1 for _ in range(len(frames))]
#Create DataFrame
df_label = pd.DataFrame(columns=['Frame', 'Visibility', 'X', 'Y'])
df_label['Frame'], df_label['Visibility'], df_label['X'], df_label['Y'] = frames, visibility, x, y
#Compensate the non-labeled frames due to no visibility of badminton
for i in range(0, frames[-1]+1):
	if i in list(df_label['Frame']):
		pass
	else:
		df_label = df_label.append(pd.DataFrame(data = {'Frame':[i], 'Visibility':[0], 'X':[0], 'Y':[0]}), ignore_index=True)

#Sorting by 'Frame'
df_label = df_label.sort_values(by=['Frame'])
df_label.to_csv(outputFile, encoding='utf-8', index=False)
