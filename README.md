This codebase contains the system described in [MonoTrack: Shuttle trajectory reconstruction from monocular badminton video](https://cs.stanford.edu/people/paulliu/files/cvpr-2022.pdf). MonoTrack is an end-to-end system for reconstructing 3D and 2D trajectories from broadcast-style badminton videos.

# Installation 

## 1. Court Detection
See the `court-detection` folder for further instructions.

## 2. Pose Estimation
We use MMPose and MMDet for pose detection. See [here](https://mmpose.readthedocs.io/en/v0.26.0/install.html) for MMPose and [here](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation) for MMDet.

## 3. Shuttle Tracking: Install our modified TrackNet
We use a modified TrackNet for shuttle tracking. See the `modified-tracknet` for further instructions.

## 4. Data Analysis / Match Statistics
Install our Python package.
```
cd python/ai-badminton/
pip install --user -e .
```

## 5. Prepare the dataset for hitnet
```
cd setup
python setup.py
```

## 6. Training HitNet
Run through the cells of `modified-tracket/train-hitnet.ipynb`.

## 7. Running our pipeline
See `ai_badminton.pipeline_clean` (i.e. `pipeline_clean.py` in `python/ai-badminton/src/`).