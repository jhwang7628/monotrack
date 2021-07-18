#!/bin/bash

slugify () {
    echo "$1" | sed -r s/[~\^]+//g | sed -r s/[^a-zA-Z0-9]+/-/g | sed -r s/^-+\|-+$//g | tr A-Z a-z
}

VID_PATH=/home/inutard/remote-disk/badminton-vids
for player in `ls $VID_PATH`
    do
    for video in $VID_PATH/$player/*.mp4
        do
        # Requires slugify-videos to be run first
        basedir=$(basename "$video" .mp4)
        mkdir -p $VID_PATH/$player/$basedir-scenes
        cd $VID_PATH/$player/$basedir-scenes
        scenedetect -i $video detect-content -t 15 split-video
        cd -
    done
done