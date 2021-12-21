#!/usr/bin/env python

from dataset import *
from media import *

import cv2
from imutils import build_montages
from imutils import paths

import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("dataset_root", help="Root path of the dataset")
parser.add_argument("--out_dir", help="Output path")
parser.add_argument("--tracknet", help="Dataset is TrackNet")

args = parser.parse_args()

root = Path(args.dataset_root)

if args.out_dir is not None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
else:
    out_dir = None

metadata = {}

video_files = get_video_paths(root)

duration_all = 0.0
video_count = 0
frames = []

for file_path in video_files:
    print("  ", file_path)

    cap = cv2.VideoCapture(str(file_path))

    if not cap.isOpened():
        continue

    duration = get_duration(cap)
    fps = get_fps(cap)

    duration_all += duration

    obj = {}
    obj["duration"] = duration
    obj["fps"] = fps

    metadata[str(file_path)] = obj

    idx, frame = extract_representative_frame(cap)
    obj["representative_frame"] = idx
    obj["dimension"] = get_image_dimension(frame)
    frames.append(frame)

    # save to outdir
    if out_dir:
        cv2.imwrite(str(out_dir / f"{video_count}.png"), frame)

    video_count += 1

    cap.release()

montages = build_montages(frames, (256, 144), (8,5))
for idx, montage in enumerate(montages):
    cv2.imshow(f"Montage {idx}", montage)
    if out_dir:
        cv2.imwrite(str(out_dir / f"montage_{idx}.png"), montage)
    cv2.waitKey(0)

metadata["metadata"] = {
    "duration" : duration_all,
    "video_count": video_count
}

print(json.dumps(metadata, indent=2))
