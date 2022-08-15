#!/usr/bin/env python

import cv2
import json
import glob
import math
import sys
from pathlib import Path

root = Path("/home/juiwang/data/ai-badminton/dataset/mada/sample_result_20211123")
image = root / "image"
video = root / "video"
outdir = Path("./out")
image_outdir = outdir / "image"
video_outdir = outdir / "video"

outdir.mkdir(exist_ok=True)
image_outdir.mkdir(exist_ok=True)
video_outdir.mkdir(exist_ok=True)

def get_image_dimension(img):
    height, width, channels = img.shape
    return width, height

def draw_point(img, pt=(480, 270), color=(0,0,255), radius=10):
    return cv2.circle(img, pt, radius=radius, color=color, thickness=-1)

def get_all_prefixes(root_path, ext="json"):
    return list(root_path.glob(f"*.{ext}"))

def visualize_keypoint_annotations(image_dir, image_outdir):
    files = get_all_prefixes(image_dir, "json")
    sorted_files = sorted(files, key=lambda x: int(x.stem.split("_")[-1]))

    for f_json in sorted_files:
        stem = f_json.stem
        f_png = image_dir / f"{stem}.jpg"
        print(f_json, f_png)
        if f_json.is_file() and f_png.is_file():
            annotations = json.load(open(f_json))
            img = cv2.imread(str(f_png))
            for point in annotations["points"]:
                pt = point["position"]
                pt[0] = int(pt[0])
                pt[1] = int(pt[1])
                print("pt = ", pt)
                img = draw_point(img, (pt[0], pt[1]), radius=2)
            cv2.imshow("test", img)
            cv2.waitKey(0)
            cv2.imwrite(str(Path(image_outdir / f"{stem}.png")), img)
            cv2.destroyAllWindows()

def get_frame_number_from_timestamp_annotations(data, fps):
    output = []
    output_time = []
    for d in data["annotations"]:
        frame_idx = int(math.floor(float(d["time"]) * fps))
        print(d["time"], frame_idx)
        output.append(frame_idx)
        output_time.append(float(d["time"]))
    print("output_time = ", sorted(output_time))
    return output

def visualize_timestamp_annotations(filename_json=None):
    if filename_json is None:
        files = get_all_prefixes(video, "json")
    else:
        files = [filename_json]
    sorted_files = sorted(files, key=lambda x: int(x.stem.split("_")[-1]))

    for f_json in sorted_files:
        stem = f_json.stem
        f_mp4 = video / f"{stem}.mp4"
        print(f_json, f_mp4)
        if f_json.is_file() and f_mp4.is_file():
            annotations = json.load(open(f_json))
            cap = cv2.VideoCapture(str(f_mp4))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_indices = get_frame_number_from_timestamp_annotations(annotations, fps)
            print("fps = ", fps)
            idx = 0
            while (cap.isOpened()):
                t = idx / float(fps)
                print("here", idx)
                ret, frame = cap.read()
                if ret == True:
                    filename = f"{stem}_{str(idx).zfill(6)}.png"
                    if idx in frame_indices:
                        frame = draw_point(frame, pt=(200, 200), color=(255,0,0), radius=10)
                        cv2.imwrite(str(Path(video_outdir / filename)), frame)
                    else:
                        cv2.imwrite(str(Path(video_outdir / filename)), frame)
                    if cv2.waitKey(25) & 0xFF == ord("q"):
                        break
                else:
                    break
                idx += 1
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":

    input_image_path = Path(sys.argv[1])
    output_image_path = Path(sys.argv[2])

    output_image_path.mkdir(exist_ok=True, parents=True)

    #visualize_timestamp_annotations()
    #visualize_timestamp_annotations(video / "0006.json")
    visualize_keypoint_annotations(input_image_path, output_image_path)
