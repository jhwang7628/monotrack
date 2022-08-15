import os
from tqdm import tqdm
from subprocess import run, DEVNULL, check_output
from multiprocessing import Pool
from ai_badminton.pose import Pose, read_player_poses, process_pose_file
from ai_badminton.court import Court, read_court
from ai_badminton.trajectory import Trajectory
from ai_badminton.video_annotator import annotate_video

court_detection_bin = '/home/work_space/ai-badminton-private/court-detection/build/bin/detect'
data_dir = '/home/work_space/data'
matches = list('match' + str(i) for i in range(1, 27)) + list('test_match' + str(i) for i in range(1, 4))

# Compute the court detections
for match in tqdm(matches):
    os.makedirs(f'{data_dir}/{match}/court/', exist_ok=True)
    args = []
    for video in os.listdir(f'{data_dir}/%s/rally_video/' % match):
        rally, _ = os.path.splitext(video)
        args.append(rally)
    
    def mapper(rally):
        check_output([court_detection_bin, 
                      f'{data_dir}/{match}/rally_video/{rally}.mp4', 
                      f'{data_dir}/{match}/court/{rally}.out'], stderr=DEVNULL)
    
    with Pool(8) as pool:
        pool.map(mapper, args)
        
# Compute the poses
model_path = "/home/work_space/models"
code_path = "/home/work_space/ai-badminton-private"
pose_est_bin = f'python {code_path}/mmpose/run_mmpose.py'

det_config = f"{code_path}/mmpose/configs/faster_rcnn_r50_fpn_coco.py"
pose_config = f"{code_path}/mmpose/configs/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
det_model = f"{model_path}/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
pose_model = f"{model_path}/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"

def create_cmd(video_path, out_path):
    return f"{pose_est_bin} {det_config} {det_model} {pose_config} {pose_model} --video-path {video_path} --out-file {out_path}"

for match in matches:
    os.makedirs(f'{data_dir}/{match}/poses/', exist_ok=True)
    print(match)
    for video in tqdm(os.listdir(f'{data_dir}/%s/rally_video/' % match)):
        rally, _ = os.path.splitext(video)
        run(create_cmd(f'{data_dir}/{match}/rally_video/{rally}.mp4', f'{data_dir}/{match}/poses/{rally}.out'), shell=True)


# Process pose files
for match in matches:
    for video in os.listdir(f'{data_dir}/%s/rally_video/' % match):
        rally, _ = os.path.splitext(video)
        
        court_pts = read_court(f'{data_dir}/%s/court/%s.out' % (match, rally))
        corners = [court_pts[1], court_pts[2], court_pts[0], court_pts[3]]
        court = Court(corners)

        print(match, rally)
        poses = process_pose_file(
            f'{data_dir}/%s/poses/%s.out' % (match, rally), 
            f'{data_dir}/%s/poses/%s' % (match, rally), 
            court,
            fullPose=True
        )

# Predict shuttle positions
# This is set by default to tracknet's released model
# Change this once you've trained one of our improved tracknet models
tracknet_model = f"{model_path}/model906_30"

from ai_badminton.pipeline_clean import tracknet_inference
from pathlib import Path
import shutil

for match in matches:
    for video in os.listdir(f'{data_dir}/%s/rally_video/' % match):
        rally, _ = os.path.splitext(video)
        tracknet_inference(
            Path(f'{data_dir}/{match}/rally_video/{rally}.mp4'), 
            Path(tracknet_model),
            Path(f'{data_dir}/{match}/ball_trajectory/{rally}_ball_predicted.csv')
        )
        