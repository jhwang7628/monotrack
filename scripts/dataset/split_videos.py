#!/usr/bin/env python
import argparse
import os
import csv
import time
import math
import subprocess

from pathlib import Path
from collections import defaultdict

def to_timestamp(t):
    return time.strftime('%H:%M:%S', time.gmtime(t)) + '.' + str(int(t * 1000) % 1000)

def run_cmd(cmd, exe=True):
    print("\n\n", cmd, "\n\n")
    if exe:
        subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split videos into cuts corresponding to output from a VIA CSV.')
    parser.add_argument('csv_file', metavar='csv_file', help='The name of the CSV file.')
    parser.add_argument("out_dir", help="Output directory")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    csvfile = open(args.csv_file, 'r')
    csvreader = csv.reader(csvfile, delimiter=',')

    cut_dict = defaultdict(list)
    for row in csvreader:
        try:
            filename = eval(row[1])[0]
            cut = eval(row[2]), eval(row[3])
            cut_dict[filename].append(cut)
        except:
            pass

    print(cut_dict)
    for dirname in ['behind-the-court', 'bleacher', 'side']:
        print(f'Processing {dirname}...')
        for file in os.listdir(dirname):
            if '.mp4' in file or '.webm' in file:
                vidname, ext = os.path.splitext(file)
                path = out_dir / dirname / vidname
                path.mkdir(exist_ok=True, parents=True)

                print('    ', file, file in cut_dict)
                for idx, cut in enumerate(cut_dict[file]):
                    start, duration = to_timestamp(cut[0]), to_timestamp(cut[1] - cut[0])
                    warm_start = to_timestamp(max(cut[0] - 15, 0))
                    print(f'Processing cut {idx}..')
                    #cmd = f'ffmpeg -ss {warm_start} -i "{dirname}/{file}" -ss {start} -t {duration} -c copy "{path}/rally_{idx}.mp4"'
                    cmd = f'ffmpeg -ss {start} -i "{dirname}/{file}" -t {duration} -vcodec libx264 -acodec copy -strict -2 "{path}/rally_{idx}.mp4"'
                    run_cmd(cmd, exe=True)
