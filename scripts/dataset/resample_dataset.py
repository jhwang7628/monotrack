from dataset import *
from media import *

import cv2

import argparse
import random
import shutil
import pickle
from collections import defaultdict

# python ~/code/ai-badminton/scripts/dataset/resample_dataset.py ./racket_split ./racket_split_resampled 4800

if __name__ == "__main__":

    random.seed(10)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dataset_root", help="Root path of the input dataset")
    parser.add_argument("output_dataset_root", help="Root path of the output dataset")
    parser.add_argument("new_duration", help="Total video duration of the output dataset")

    args = parser.parse_args()

    input_root = Path(args.input_dataset_root)
    output_root = Path(args.output_dataset_root)
    new_duration = float(args.new_duration)

    output_root.mkdir(exist_ok=True, parents=True)

    # read in all videos and durations
    video_paths = get_video_paths(input_root, allowed_extensions=["mp4"])
    matches = defaultdict(list)
    match_names_map = {}
    for video_path in video_paths:
        print("Reading", video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        duration = get_duration(cap)
        match_name = str(video_path.parts[-2])
        camera_view_type = str(video_path.parts[-3])
        sport_type = str(video_path.parts[-4])
        if match_name in match_names_map:
            match_idx = match_names_map[match_name]
        else:
            match_idx = len(match_names_map)
            match_names_map[match_name] = match_idx

        matches[match_name].append({
            "input_path": video_path,
            "duration": duration,
            "camera_view_type": camera_view_type,
            "sport_type": sport_type,
            "match_idx": match_idx,
        })

    def shuffle_dict(d):
        for k, v in d.items():
            random.shuffle(v)
        l = list(d.items())
        random.shuffle(l)
        return dict(l)
    matches = shuffle_dict(matches)
    num_matches = len(matches)

    print("Resampled matches")
    for m in matches:
        print("=========\n", m, "\n========")
        for mm in matches[m]:
            print(mm["input_path"], mm["match_idx"])

    # go through the matches and pop one rally at a time until reaching target duration
    stats = {}
    total_duration = 0
    num_rallies = 0
    resampled_matches = defaultdict(list)
    done = False
    while total_duration < new_duration:
        empty_matches = 0
        for _, match in matches.items():
            if len(match) == 0:
                empty_matches += 1
                continue
            rally = match.pop()

            if total_duration + rally["duration"] >= new_duration:
                done = True
                break

            match_idx = rally["match_idx"]
            resampled_matches[match_idx].append(rally)
            num_rallies += 1
            total_duration += rally["duration"]

        # we exhaust all matches and still not reach limit
        if empty_matches == len(matches):
            done = True

        if done:
            break

    print(resampled_matches)
    print(total_duration)

    file_map = []
    view_duration = defaultdict(float)
    view_matches = defaultdict(set)
    view_num_rallies = defaultdict(int)
    view_types = set()
    for _, match in resampled_matches.items():
        for rally_idx, rally in enumerate(match):
            match_idx = rally["match_idx"]
            out_dir = output_root / rally["sport_type"] / rally["camera_view_type"] / f"match_{match_idx}"
            out_dir.mkdir(exist_ok=True, parents=True)
            out_path = out_dir / f"rally_{rally_idx}.mp4"
            in_path = rally["input_path"]
            print("copying ", in_path, "to", out_path)
            shutil.copy(in_path, out_path)
            file_map.append((in_path, out_path))
            # stats
            view_type = rally["camera_view_type"]
            view_types.add(view_type)
            view_duration[view_type] += rally["duration"]
            view_matches[view_type].add(match_idx)
            view_num_rallies[view_type] += 1

    # write out all the metadata
    out_dir = output_root / "metadata"
    out_dir.mkdir(exist_ok=True, parents=True)

    stats = []

    stats.append(("total_duration", total_duration))
    stats.append(("total_num_matches", num_matches))
    stats.append(("total_num_rallies", num_rallies))

    for view_type in view_types:
        stats.append((f"{view_type}_duration", view_duration[view_type]))
        stats.append((f"{view_type}_num_matches", len(view_matches[view_type])))
        stats.append((f"{view_type}_num_rallies", view_num_rallies[view_type]))

    with open(out_dir / "file_map.txt", "w") as stream:
        for f in file_map:
            stream.write(f"{str(f[0])} {str(f[1])}\n")

    with open(out_dir / "statistics.txt", "w") as stream:
        for s in stats:
            stream.write(f"{s[0]} {s[1]}\n")
