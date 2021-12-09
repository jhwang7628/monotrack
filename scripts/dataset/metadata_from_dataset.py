#!/usr/bin/env python

import json
import argparse
from pathlib import Path
from imutils import build_montages
from imutils import paths

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("dataset_root", help="Root path of the dataset")
parser.add_argument("--out_dir", help="Output path")
parser.add_argument("--tracknet", help="Dataset is TrackNet")

args = parser.parse_args()

root = Path(args.dataset_root)

allowed_extensions = [".mp4", ".webm"]

if args.out_dir is not None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
else:
    out_dir = None

sports_type_paths = [x for x in root.iterdir() if x.is_dir()]

def get_image_dimension(img):
    height, width, channels = img.shape
    return width, height

def get_fps(video_capture):
    return video_capture.get(cv2.CAP_PROP_FPS)

def get_duration(video_capture):
    fps = get_fps(video_capture)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count / fps

def extract_representative_frame(video_capture):
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = frame_count // 2
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = video_capture.read()
    assert ret == True
    return idx, frame

metadata = {}

file_ids = []

for sports_type_path in sports_type_paths:
    camera_type_paths = [x for x in sports_type_path.iterdir() if x.is_dir()]
    sport_key = str(sports_type_path.name)
    metadata[sport_key] = {}
    for camera_type_path in camera_type_paths:
        file_paths = [x for x in camera_type_path.iterdir() if (x.is_file() and x.suffix in
                                                                allowed_extensions)]
        camera_key = str(camera_type_path.name)
        metadata[sport_key][camera_key] = {}
        for file_path in file_paths:
            file_ids.append((sport_key, camera_key, file_path.name, file_path))
            # tracknet stores different rally for the same match. skip here to render one thumbnail
            # per match
            if args.tracknet:
                break

duration_all = 0.0
video_count = 0
frames = []

for file_id in file_ids:
    sport_key, camera_key, file_key, file_path = file_id
    print("file_id = ", file_id)

    cap = cv2.VideoCapture(str(file_path))

    if not cap.isOpened():
        continue

    duration = get_duration(cap)
    fps = get_fps(cap)

    duration_all += duration

    obj = {}
    obj["duration"] = duration
    obj["fps"] = fps

    metadata[sport_key][camera_key][file_key] = obj

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
