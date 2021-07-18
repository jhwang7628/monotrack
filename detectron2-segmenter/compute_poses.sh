#!/usr/bin/bash

for id in {1..22}
do
    folder=/home/groups/djames/prj-sports-video/data/match$id
    mkdir -p $folder/poses/
    echo $folder
    for video in $(ls $folder/rally_video)
    do
        pose_file=${video/.mp4/.out}
        python3 video_to_pose.py --gpus 2 -i $folder/rally_video/$video -f $folder/poses/$pose_file -p --config-file configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml	
    done
done
