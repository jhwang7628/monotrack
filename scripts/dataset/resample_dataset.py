from dataset import *
from media import *

import cv2

import argparse
import random
from collections import defaultdict

if __name__ == "__main__":

    random.seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dataset_root", help="Root path of the input dataset")
    parser.add_argument("output_dataset_root", help="Root path of the output dataset")
    parser.add_argument("new_duration", help="Total video duration of the output dataset")

    args = parser.parse_args()

    input_root = Path(args.input_dataset_root)
    output_root = Path(args.output_dataset_root)
    new_duration = float(args.new_duration)

    output_root.mkdir(exist_ok=True, parents=True)

    video_paths = get_video_paths(input_root, allowed_extensions=["mp4"])

    def shuffle_dict(d):
        for k, v in d.items():
            random.shuffle(v)
        l = list(d.items())
        random.shuffle(l)
        return dict(l)

    matches = defaultdict(list)
    for video_path in video_paths:
        print("Reading", video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        duration = get_duration(cap)

        match_name = str(video_path.parts[-2])

        matches[match_name].append({
            "input_path": video_path,
            "duration": duration,
        })

    matches = shuffle_dict(matches)

    total_duration = 0
    resampled_matches = defaultdict(list)
    done = False
    while total_duration < new_duration:
        match_idx = 0
        empty_matches = 0
        for _, match in matches.items():
            if len(match) == 0:
                empty_matches += 1
                continue
            rally = match.pop()

            if total_duration + rally["duration"] >= new_duration:
                done = True
                break

            rally["rally_idx"] = len(resampled_matches[match_idx])
            resampled_matches[match_idx].append(rally)
            match_idx += 1
            total_duration += rally["duration"]

        # we exhaust all matches and still not reach limit
        if empty_matches == len(matches):
            done = True

        if done:
            break

    ## for each match, randomly sample up to num_rallies
    #total_duration = 0
    #num_rallies = 10
    #match_idx = 0
    #for k, v in matches.items():
    #    chosen_matches = random.sample(v, min(num_rallies, len(v)))
    #    for rally_idx, m in enumerate(chosen_matches):
    #        total_duration += m["duration"]
    #        if total_duration >= new_duration:
    #            break
    #    if total_duration >= new_duration:
    #        break

    print(resampled_matches)
    print(total_duration)

    ## sort based on how many matches we have for the match
    #sorted_matches = dict(sorted(matches.items(), key=lambda item: len(item[1])))
    #
    #print("\n\n")
    #for m in sorted_matches:
    #    print(m, len(sorted_matches[m]))

