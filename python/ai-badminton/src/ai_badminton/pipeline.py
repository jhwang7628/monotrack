import os
import shutil
import pandas as pd

from .trajectory import *
from .pose import *
from .court import *
from .video_annotator import *
from .rally_reconstructor import *

import multiprocessing
from pathlib import Path

"""
The pipeline takes in an input video and performs several transformations:
    - Split video into cuts, assuming each cut is a rally
    - Run court detection once, assuming all cuts are from a fixed tripod position
    - For each video (can be done in parallel):
        - Run pose detection to figure out poses
        - Run shuttle detection to figure out shuttle positions
    - Filter out the two poses on court based on court position
    - Detect hits
"""

class Pipeline(object):
    def __init__(
        self,
        shotcut_model,
        detector_model,
        pose_model,
        shuttle_model,
        hit_model
    ):
        self.shotcut_model = shotcut_model
        self.detector_model = detector_model
        self.pose_model = pose_model
        self.shuttle_model = shuttle_model
        self.hit_model = hit_model
        
    def split_cuts(self, shotcut_path):
        status = os.system('python3 %s %s %s/rally_video/' % (shotcut_path, self.video_path, self.output_path))
        return status

    def make_output_dirs(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for info in ['ball_trajectory', 'court', 'poses', 'rally_video', 'shot', 'annotated', '3d', 'annotated3d']:
            status = os.makedirs('%s/%s' % (output_dir, info), exist_ok=True)
            if status:
                return status
        return status

    def save_video(self, path=None):
        if path is None:
            shutil.copyfile(self.video_path, '%s/%s' % (self.output_path, self.video_name))
        else:
            shutil.copyfile(self.video_path, path)

    def detect_court(self, detector_path):
        status = os.system('%s %s %s/court/%s.out' % (detector_path, self.video_path, self.output_path, self.video_prefix))
        return status
    
    def detect_poses(self, pose_path, model_path, model_type):
        if model_type == 'detectron2':
            status = os.system('/opt/conda/bin/python3 %s --gpus 4 -i %s -f %s/poses/%s.out -p --config-file %s' % (pose_path, self.video_path, self.output_path, self.video_prefix, model_path))
            self.postprocess_poses()
            return status
        elif model_type == 'mmpose':
            command = "/opt/conda/bin/python3 /home/code-base/user_space/ai-badminton/mmpose/run_mmpose.py /home/code-base/user_space/ai-badminton/mmpose/configs/faster_rcnn_r50_fpn_coco.py     https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth     /home/code-base/user_space/ai-badminton/mmpose/configs/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py     https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth --video-path %s --out-file %s/poses/%s.out" % (self.video_path, self.output_path, self.video_prefix)
            status = os.system(command)
            self.postprocess_poses(fullPose=True)
            return status
        else:
            raise Exception('Unsupported pose model!')
    
    def detect_shuttles(self, tracknet_path, model_path):
        status = os.system('/opt/conda/bin/python3 %s --video_name=%s --load_weights=%s' % (tracknet_path, self.video_path, model_path))
        if status:
            return status
        video = '/'.join(os.path.split(self.video_path)[:-1]) + '/' + self.video_prefix
        shutil.copyfile(video + '_predict.csv', '%s/ball_trajectory/%s_ball.csv' % (self.output_path, self.video_prefix))
    
    def postprocess_poses(self, fullPose=False):
        court_pts = read_court('%s/court/%s.out' % (self.output_path, self.base_prefix))
        corners = [court_pts[1], court_pts[2], court_pts[0], court_pts[3]]
        court = Court(corners)
        process_pose_file('%s/poses/%s.out' % (self.output_path, self.video_prefix), 
                          '%s/poses/%s' % (self.output_path, self.video_prefix), 
                          court,
                          fullPose)
    
    def detect_hits(self, model_path, annotate=False):
        from ai_badminton import hit_detector
        
        cap = cv2.VideoCapture('%s' % self.video_path)
        if cap.isOpened() is False:
            print('Error opening video stream or file')

        fps = cap.get(cv2.CAP_PROP_FPS)
        
        trajectory = Trajectory('%s/ball_trajectory/%s_ball.csv' % (self.output_path, self.video_prefix))
        poses = read_player_poses('%s/poses/%s' % (self.output_path, self.video_prefix))
        
        court_pts = read_court('%s/court/%s.out' % (self.output_path, self.base_prefix))
        corners = [court_pts[1], court_pts[2], court_pts[0], court_pts[3]]
        court = Court(corners)
        
        detector = hit_detector.MLHitDetector(
            court,
            poses,
            trajectory,
            model_path
        )
        result, is_hit = detector.detect_hits(fps)
        
        # Write hit to csv file
        L = len(trajectory.X)
        frames = list(range(L))
        hits = [0] * L
        for fid, pid in zip(result, is_hit):
            hits[fid] = pid
            
        data = {'frame' : frames, 'hit' : hits}
        df = pd.DataFrame(data=data)
        df.to_csv('%s/shot/%s_hit.csv' % (self.output_path, self.video_prefix))
    
        if annotate and len(result) > 3:
            outfile = '%s/annotated/%s_annotated.mp4' % (self.output_path, self.video_prefix)
            annotate_video(
                cap, court, poses, trajectory, result, is_hit, 
                outfile=outfile
            )
            
            # Re-encode with ffmpeg from mp41 to mp42
            os.system("ffmpeg -i %s -c:v libx264 -crf 18 -preset slow -c:a copy tmp.mp4" % outfile)
            os.system("mv tmp.mp4 %s" % outfile)
        
    def reconstruct_trajectory(self, visualize=True):        
        trajectory = Trajectory('%s/ball_trajectory/%s_ball.csv' % (self.output_path, self.video_prefix))
        poses = read_player_poses('%s/poses/%s' % (self.output_path, self.video_prefix))
        hits = read_hits_file('%s/shot/%s_hit.csv' % (self.output_path, self.video_prefix))
        
        court_pts = read_court('%s/court/%s.out' % (self.output_path, self.base_prefix))
        corners = [court_pts[1], court_pts[2], court_pts[0], court_pts[3]]
        pole_tips = [court_pts[4], court_pts[5]]
        court3d = Court3D(corners, pole_tips)
        
        cap = cv2.VideoCapture('%s' % self.video_path)
        if cap.isOpened() is False:
            print('Error opening video stream or file')

        fps = cap.get(cv2.CAP_PROP_FPS)
        
        reconstructor = RallyReconstructor(
            court3d,
            poses,
            trajectory,
            hits
        )
        
        results = reconstructor.reconstruct(fps)
        # Write hit to csv file
        L = min(len(trajectory.X), results.shape[0])
        frames = list(range(L))
        
        data = {
            'frame' : frames, 
            'ball_x' : results[:, 0].tolist(), 
            'ball_y' : results[:, 1].tolist(), 
            'ball_z' : results[:, 2].tolist()
        }
        df = pd.DataFrame(data=data)
        df.to_csv('%s/3d/%s_3d.csv' % (self.output_path, self.video_prefix))
        
        if visualize:
            annotated2d = '%s/annotated/%s_annotated.mp4' % (self.output_path, self.video_prefix)
            cap = cv2.VideoCapture(annotated2d)
            if cap.isOpened() is False:
                cap = cv2.VideoCapture('%s' % self.video_path)
            df = pd.read_csv('%s/3d/%s_3d.csv' % (self.output_path, self.video_prefix))
            outfile = '%s/annotated3d/%s_3d.mp4' % (self.output_path, self.video_prefix)
            annotate_video_3d(
                cap, court3d,
                df,
                outfile=outfile
            )
            
            # Re-encode with ffmpeg from mp41 to mp42
            os.system("ffmpeg -i %s -c:v libx264 -crf 18 -preset slow -c:a copy tmp.mp4" % outfile)
            os.system("mv tmp.mp4 %s" % outfile)
                
    def set_video_variables(self, video_path):
        self.video_path = video_path
        self.video_name = os.path.split(self.video_path)[-1]
        self.video_prefix, _ = os.path.splitext(self.video_name)
        
    def run_pipeline(self, video_path, output_path, annotate=True, split_cuts=False):        
        logfile = open('%s/log.txt' % output_path, 'w')
        self.output_path = output_path
        self.make_output_dirs(output_path)
        self.set_video_variables(video_path)
        
        # For use in court preprocessing step
        self.base_prefix, _ = os.path.splitext(self.video_name)
        
        if split_cuts:
            logfile.write('Splitting file into shots...\n')
            status = self.split_cuts(self.shotcut_model)
            if status:
                return

            self.save_video()
            logfile.write('Done file splitting!\n')
        else:
            self.save_video('%s/rally_video/%s' % (self.output_path, self.video_name))
            logfile.write('Assumed entire video is one rally.\n')
        
        logfile.write('Detecting courts...\n')
        # Assumes the same court persists throughout the video
        status = self.detect_court(self.detector_model)
        if status:
            return
        
        videos = os.listdir('%s/rally_video' % self.output_path)
        print(len(videos))
        if len(videos) > 30 * 2 * 3:
            logfile.write('Too many rallies for a single game: %d\n' % len(videos))
            return 1
        else:
            for video in videos:
                _, ext = os.path.splitext(video)
                if ext != '.mp4' or '_predict' in _:
                    continue
                logfile.write('Processing rally: %s\n' % video)
                try:
                    # Set the next video
                    self.set_video_variables('%s/rally_video/%s' % (self.output_path, video))
                    logfile.write('Detecting poses...\n')
                    status = self.detect_poses(self.pose_model[0], self.pose_model[1], self.pose_model[2])
                    if status:
                        return 1

                    logfile.write('Detecting shuttles...\n')
                    self.detect_shuttles(self.shuttle_model[0], self.shuttle_model[1])
                        
                    # This workaround is because Tensorflow fails to automatically release memory
                    # see https://github.com/tensorflow/tensorflow/issues/36465
                    # The failure to release memory means that the ball detection model will not have
                    # enough memory to run while the hit detector is in use.
                    logfile.write('Detecting hits...\n')
                    detect = lambda: self.detect_hits(self.hit_model, annotate)
                    p = multiprocessing.Process(target=detect)
                    p.start()
                    p.join()
                    
                    self.reconstruct_trajectory()
                except:
                    logfile.write('Something went wrong!\n')
                    pass

        # All done!
        return 0

def run_pose_detection_on_match(match_dir_path):
    print(match_dir_path)



if __name__ == "__main__":

    base_dir = Path("/sensei-fs/users/juiwang/ai-badminton/data/tracknetv2_042022/profession_dataset")
    for match_idx in range(1, 23):
        run_pose_detection_on_match(base_dir / f"match{match_idx}")

