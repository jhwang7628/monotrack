#!/bin/bash

slugify () {
    echo "$1" | sed -r s/[~\^]+//g | sed -r s/[^a-zA-Z0-9]+/-/g | sed -r s/^-+\|-+$//g | tr A-Z a-z
}

VID_PATH=/home/code-base/user_space/mined_data/videos
for player in `ls $VID_PATH`
    do
    for video in $VID_PATH/$player/*
        do
        prevname=$(basename "$video" .mp4)
        newname=$(slugify "$prevname")
        newpath=$VID_PATH/$player/$newname.mp4
        mv "$video" $newpath
    done
done