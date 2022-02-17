#!/usr/bin/env python


import cv2
import os

import json
import argparse
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("match_root", help="Root path of the match")
parser.add_argument("out_path", help="Output path for the output mp4")

args = parser.parse_args()

root = Path(args.match_root)

files = os.listdir(root)
files = set(name.split('.')[0] for name in files)
files = sorted(list(files), key=lambda x: int(x.split('_')[-1]))

image = cv2.imread(str(root.joinpath(files[0] + '.jpg')))
cap = cv2.VideoWriter(
    args.out_path, 
    cv2.VideoWriter_fourcc('m','p','4','v'), 
    30.0, (image.shape[1], image.shape[0])
)

for path in tqdm(files):
    image = cv2.imread(str(root.joinpath(path + '.jpg')))
    image = cv2.putText(image, path, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    labels = json.load(open(str(root.joinpath(path + '.json')), 'r'))
    for point in labels['points']:
        # if '球中心' in point['label']:
        x, y = point['position'][0], point['position'][1]
        image = cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
    cap.write(image)
cap.release()
