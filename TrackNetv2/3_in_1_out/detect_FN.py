import sys, getopt
import shutil
import numpy as np
import os
from glob import glob
import piexif
import pandas as pd
import math

try:
	(opts, args) = getopt.getopt(sys.argv[1:], '', [
		'labelFile=',
		'predictFile=',
		'tol='
	])
	if len(opts) != 3:
		raise ''
except:
	print('usage: python3 detect_FN.py --labelFile=<labeledFile> --predictFile=<predictedFile> --tol=<toleranceValue>')
	exit(1)

labelFile = ''
predictFile = ''
tol = 10
for (opt, arg) in opts:
	if opt == '--tol':
		tol = int(arg)
	elif opt == '--labelFile':
		labelFile = arg
	elif opt == '--predictFile':
		predictFile = arg
	else:
		print('usage: python3 detect_FN.py --labelFile=<labeledFile> --predictFile=<predictedFile> --tol=<toleranceValue>')
		exit(1)

#Labeled file
data1 = pd.read_csv(labelFile)
no1 = data1['Frame'].values
v1 = data1['Visibility'].values
x1 = data1['X'].values
y1 = data1['Y'].values
#Predicted file
data2 = pd.read_csv(predictFile)
no2 = data2['Frame'].values
v2 = data2['Visibility'].values
x2 = data2['X'].values
y2 = data2['Y'].values

offset = no2[0] - no1[0]
#0 for TP, 1 for TN, 2 for FP1, 3 for FP2, 4 for FN
outcome = []
#no1[i + offset] is equal to no2[i]
n = min(len(no1) - offset, len(no2))
for i in range(n):
	#Negative
	if v2[i] == 0:
		#TN
		if v1[i + offset] == 0:
			outcome.append(1)
		#FN
		elif v1[i + offset] == 1:
			outcome.append(4)
	#Positive
	elif v2[i] == 1:
		if v1[i + offset] == 0:
			outcome.append(3)
		elif v1[i + offset] == 1:
			dist = math.sqrt(pow(x2[i] - x1[i + offset], 2) + pow(y2[i] - y1[i + offset], 2))
			if dist > tol:
				outcome.append(2)
			else:
				outcome.append(0)

#If the size of the predicted data may be larger than labeled data
if len(no1) < len(no2) + offset:
	for i in range(len(no1) - offset, len(no2)):
		if v2[i] == 0:
			outcome.append(1)
		elif v2[i] == 1:
			outcome.append(3)



continuous_FN = []
accumulate = 0
for i in range(len(outcome)):
	if outcome[i] == 4:
		accumulate += 1
	else:
		if accumulate > 0:
			continuous_FN.append(accumulate)
		accumulate = 0

if accumulate > 0:
	continuous_FN.append(accumulate)

continuous_FN.sort()

s = e = 0
while e < len(continuous_FN):
	s = e
	while continuous_FN[e] == continuous_FN[s]:
		e += 1
		if e >= len(continuous_FN):
			break
	print('Number of '+ str(continuous_FN[s]) + ' successive FNs: ' + str(e - s))

print('Outcome of every frame of the predicted labeling csv file (0 for TP, 1 for TN, 2 for FP1, 3 for FP2, 4 for FN):')
print(outcome)
TP = TN = FP1 = FP2 = FN = 0
for i in range(len(outcome)):
	if outcome[i] == 0:
		TP += 1
	elif outcome[i] == 1:
		TN += 1
	elif outcome[i] == 2:
		FP1 += 1
	elif outcome[i] == 3:
		FP2 += 1
	elif outcome[i] == 4:
		FN += 1

print('(TP, TN, FP1, FP2, FN):', (TP, TN, FP1, FP2, FN))
