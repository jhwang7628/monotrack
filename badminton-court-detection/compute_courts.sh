#!/bin/bash

for id in {23..26}
do
    folder=./data/match$id
    mkdir -p $folder/court/
    echo $folder
    for video in $(ls $folder/rally_video)
    do
        court_file=${video/.mp4/.out}
        ./build/bin/detect $folder/rally_video/$video $folder/court/$court_file
    done
done

