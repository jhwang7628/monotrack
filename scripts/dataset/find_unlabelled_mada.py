#!/usr/bin/env python


import cv2
import os

import json
import argparse
import numpy as np 

from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("dataset_root", help="Root path of the mada folder")
# parser.add_argument("video_root", help="Root path of our video folder")

args = parser.parse_args()

root = Path(args.dataset_root)

subdata = ["behind-the-court", "bleacher", "side"]
unlabelled_counts = []
percentage_unlabelled = []
label_per_frame = []
bad_count = 0
for data in subdata:
    subdata_root = root.joinpath(data)
    for match in os.listdir(subdata_root):
        match_root = subdata_root.joinpath(match)
        try:
            for rally in os.listdir(match_root):
                try:
                    rally_root = match_root.joinpath(rally)
                    files = os.listdir(rally_root)
                    files = set(name.split('.')[0] for name in files if name[-4:] == 'json')
                    files = sorted(list(files), key=lambda x: int(x.split('_')[-1]))
                    label_per_frame.append([])
                    unlabelled = 0
                    for file in files:
                        labels = json.load(open(str(rally_root.joinpath(file + '.json')), 'r'))
                        is_unlabelled = 1
                        for point in labels['points']:
                            if '球中心' in point['label']:
                                is_unlabelled = 0
                        unlabelled += is_unlabelled
                        label_per_frame[-1].append(is_unlabelled)
                    unlabelled_counts.append(unlabelled)
                    percentage_unlabelled.append(unlabelled / len(files))
                    if unlabelled > 0 and unlabelled / len(files) > 0.15 and len(files) > 100:
                        bad_count += 1
                        print(data, match, rally, unlabelled, unlabelled / len(files))
                except:
                    pass
        except:
            pass

print(len(unlabelled_counts), np.mean(unlabelled_counts))
print(len(percentage_unlabelled), np.mean(percentage_unlabelled))
print(bad_count)



# image = cv2.imread(str(root.joinpath(files[0] + '.jpg')))
# cap = cv2.VideoWriter(
#     args.out_path, 
#     cv2.VideoWriter_fourcc('m','p','4','v'), 
#     30.0, (image.shape[1], image.shape[0])
# )

# for path in tqdm(files):
#     image = cv2.imread(str(root.joinpath(path + '.jpg')))
#     image = cv2.putText(image, path, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#     labels = json.load(open(str(root.joinpath(path + '.json')), 'r'))
#     for point in labels['points']:
#         # if '球中心' in point['label']:
#         x, y = point['position'][0], point['position'][1]
#         image = cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
#     cap.write(image)
# cap.release()
