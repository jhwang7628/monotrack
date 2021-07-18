Currently a playground of ideas and code for applying pose estimation to badminton.

# Pose Estimation

1. Install the necessary dependencies for detectron2: [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

2. Run the following commands:
```
cd detectron2-segmenter
python process_video.py -i assets/videos/walk.small.mp4 -p -d --config-file configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml
```

In general, the options for the script are:
```
python process_video.py -i [video_file] -p -d --config-file [model_file]
```
A list of different models can be found in configs (see complete list [here](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)). 

3. (Optional) Use some of our mined badminton videos instead: [Drive Link](https://drive.google.com/file/d/17CcjALOAl51mmlUmV5qw7KFJreWvz4RV/view?usp=sharing).

The full set of videos can be downloaded through:
```
wget -r --no-parent http://35.203.182.12/remote-disk/badminton-vids/
```

4. Output the poses into a text file:
```
python video_to_pose.py -i [video_file] -p --config-file [model_file]
```
The output will be in `poses.out`.

# Shuttle Tracking

See README file in `./TrackNetv2`. TrackNetv2 produces a csv file of the shuttle's pixel coordinates on each frame as a csv file.

# Data Analysis / Match Statistics

1. Create the folder `notebooks/data`.

2. Put the video, pose output, and the shuttle output files into `notebooks/data`. Rename the pose output file to be `[video_name]_poses.out` and the shuttle output file to be `[video_name]_predict.csv`.

3. Create the folder `notebooks/output`. Step through the cells of `notebooks/single-match-analysis.ipynb`. The output will be written to `notebooks/output`.  

# (Optional) Mining the datasets

The datasets were mined with [youtube-dl](https://ytdl-org.github.io/youtube-dl/index.html) from youtube videos containing the players' names and the keywords "nice angle". The videos were then sorted with the scripts in `./scripts/`.
