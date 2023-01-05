from ai_badminton.pose import process_pose_file, read_player_poses
from ai_badminton.court import Court, read_court, court_points_to_corners, court_points_to_corners_and_poles
from ai_badminton.hit_detector import MLHitDetector
from ai_badminton.trajectory import Trajectory
from ai_badminton.rally_reconstructor import Court3D, RallyReconstructor

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

assert has_mmdet, 'Please install mmdet to run the demo.'

from tqdm import tqdm
import cv2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Reshape
import tensorflow.keras.backend as K
import pandas as pd

import numpy as np

from pathlib import Path
import subprocess
import time
import copy

# TrackNet dependencies
import sys
sys.path.insert(0, "/home/work_space/ai-badminton-private/modified-tracknet")
from tracknet_improved import custom_loss
from utils import custom_time, get_coordinates
from constants import NUM_CONSEC, HEIGHT, WIDTH, grayscale

####################################################################################################

MODEL_PATH = "/home/juiwang/ai-badminton/data/models"
CODE_BASE_PATH = "/home/juiwang/ai-badminton/code/ai-badminton"

DET_CONFIG = f"{CODE_BASE_PATH}/mmpose/configs/faster_rcnn_r50_fpn_coco.py"
POSE_CONFIG = f"{CODE_BASE_PATH}/mmpose/configs/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
DET_CHECKPOINT = f"{MODEL_PATH}/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
POSE_CHECKPOINT = f"{MODEL_PATH}/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"

SHUTTLE_TRACKING_MODEL = f"{MODEL_PATH}/model906_30"
HIT_DETECTION_MODEL = f"{MODEL_PATH}/hitnet_conv_model_predict_direction-12-6-0.h5"

COURT_DETECTION_BIN = f"{CODE_BASE_PATH}/tennis-court-detection/build/bin/detect"

# optional
DEVICE="cuda:0"
DET_CAT_ID=1
BBOX_THR=0.3

# global
u_det_model = None
u_pose_model = None

def assert_file_exists(file_path):
    p = Path(file_path)
    assert p.is_file()

def run(cmd):
    print(cmd)
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        raise Exception("cmd process returned error code")

def run_mmpose(video_path, output_file, det_model, pose_model):

    dataset = pose_model.cfg.data['test']['type']

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f'Faild to load video file {video_path}'

    out_file = open(output_file, 'w')

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_id in tqdm(range(total_frames)):
        out_file.write('frame %d\n' % frame_id)
        flag, img = cap.read()
        if not flag:
            break
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, DET_CAT_ID)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=BBOX_THR,
            format='xyxy',
            dataset=dataset,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        for i, result in enumerate(pose_results):
            out_file.write('pose %d\n' % i)
            for kp in range(result['keypoints'].shape[0]):
                keypoints = " ".join(str(x) for x in result['keypoints'][kp])
                out_file.write(keypoints + '\n')

def run_pose_detection_on_match(match_dir):

    assert_file_exists(DET_CONFIG)
    assert_file_exists(DET_CHECKPOINT)
    assert_file_exists(POSE_CONFIG)
    assert_file_exists(POSE_CHECKPOINT)

    # build the detector and pose model from a config file and a checkpoint file
    det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device=DEVICE.lower())
    pose_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE.lower())

    pose_output = match_dir / "poses"
    pose_output.mkdir(parents=True, exist_ok=True)

    rally_dir = match_dir / "rally_video"
    assert rally_dir.is_dir()

    for video_path in rally_dir.iterdir():
        print("\n", video_path)
        output_file = pose_output / (video_path.stem + ".out")
        if output_file.is_file():
            continue

        run_mmpose(video_path, str(output_file), det_model, pose_model)

def run_pose_postprocessing(match_dir):

    court_path = match_dir / "court"
    assert court_path.is_dir(), f"Court path {str(court_path)} does not exist."
    for p in court_path.iterdir():
        if p.suffix == ".out":
            print(f"Reading court file: {str(p)}")
            court_pts = read_court(str(p))
            corners = [court_pts[1], court_pts[2], court_pts[0], court_pts[3]]
            court = Court(corners)

            pose_path = match_dir / "poses" / (p.stem + ".out")
            assert pose_path.is_file(), f"Pose path {str(pose_path)} does not exist["

            output_prefix = pose_path.with_suffix("")

            print(f"Processing pose file {str(pose_path)} with output prefix {str(output_prefix)}")
            process_pose_file(str(pose_path),
                              str(output_prefix),
                              court,
                              True)

def run_court_detection_on_match(match_dir):

    assert_file_exists(COURT_DETECTION_BIN)

    court_output = match_dir / "court"
    court_output.mkdir(parents=True, exist_ok=True)

    rally_dir = match_dir / "rally_video"
    assert rally_dir.is_dir()

    for video_path in rally_dir.iterdir():
        print("\n", video_path)
        output_file = court_output / (video_path.stem + ".out")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.is_file():
            continue

        cmd = f"{COURT_DETECTION_BIN} {str(video_path)} {str(output_file)}"
        try:
            run(cmd)
        except Exception:
            print(f"ERROR on computing court for {str(video_path)} in {str(match_dir)}")

def tracknet_inference(video_path, weights_path, output_path):

    print("tracknet_inference")
    model = load_model(str(weights_path), custom_objects={"custom_loss": custom_loss})
    imgs_input = Input(shape=(NUM_CONSEC, HEIGHT, WIDTH, 3))
    x = K.permute_dimensions(imgs_input, (0, 1, 4, 2, 3))
    imgs_output = Reshape(target_shape=(NUM_CONSEC*3, HEIGHT, WIDTH))(x)
    model = Model(imgs_input, model(imgs_output))

    print("Beginning prediction")
    start = time.time()

    stream = open(output_path, "w")
    stream.write("Frame,Visibility,X,Y,Time\n")

    cap = cv2.VideoCapture(str(video_path))

    def read_frame():
        flag, image = cap.read()
        timestamp = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
        return flag, image, timestamp

    success, images, frame_time = [], [], []
    for i in range(NUM_CONSEC):
        s, im, t = read_frame()
        success.append(s)
        images.append(im)
        frame_time.append(t)

    ratioy = images[0].shape[0] / HEIGHT
    ratiox = images[0].shape[1] / WIDTH

    size = (int(WIDTH*ratiox), int(HEIGHT*ratioy))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if video_path.suffix == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    elif video_path.suffix == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        assert False, f"video type can only be .avi or .mp4: {video_path}"

    print('About to begin prediction...')

    count = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total)

    while True:
        unit = []
        # Adjust BGR format (cv2) to RGB format (PIL)
        for i in range(NUM_CONSEC):
            image = cv2.resize(images[i], dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            if grayscale:
                xi = np.average(image, axis=-1)
            else:
                xi = image
            unit.append(xi)

        unit = np.asarray(unit)
        if grayscale:
            unit = unit.reshape((1, NUM_CONSEC, HEIGHT, WIDTH))
        else:
            unit = unit.reshape((1, NUM_CONSEC, HEIGHT, WIDTH, 3))
        unit = unit.astype('float32')
        unit /= 255

        y_pred = model.predict(unit, batch_size=1)[0]
        y_pred = y_pred > 0.5
        y_pred = (y_pred * 255).astype('uint8')
        for i in range(NUM_CONSEC):
            if np.max(y_pred[i]) == 0:
                stream.write(str(count)+',0,0,0,'+frame_time[i]+'\n')
            else:
                x, y = get_coordinates(y_pred[i])
                cx_pred, cy_pred = int(x * ratiox), int(y * ratioy)
                stream.write(str(count)+',1,'+str(cx_pred)+','+str(cy_pred)+','+frame_time[i]+'\n')
            count += 1

        success, images, frame_time = [], [], []
        for i in range(NUM_CONSEC):
            try:
                s, im, t = read_frame()
                success.append(s)
                images.append(im)
                frame_time.append(t)
            except:
                pass

        if len(success) < NUM_CONSEC or not success[-1]:
            break

        pbar.n = count
        pbar.last_print_n = count
        pbar.refresh()

    stream.close()
    end = time.time()
    print('Prediction time:', end-start, 'secs')
    print('Done......')

def run_shuttle_detection(match_dir):

    assert Path(SHUTTLE_TRACKING_MODEL).is_file()

    shuttle_output = match_dir / "ball_trajectory"
    shuttle_output.mkdir(parents=True, exist_ok=True)

    rally_dir = match_dir / "rally_video"
    assert rally_dir.is_dir()

    for video_path in rally_dir.iterdir():
        if video_path.suffix != ".mp4":
            continue

        output_path = shuttle_output / (video_path.stem + "_ball_predict.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tracknet_inference(video_path, SHUTTLE_TRACKING_MODEL, output_path)

def read_poses_court_trajectory(match_path, video_name, read_predicted_trajectory=True):
    poses = read_player_poses(str(match_path / "poses" / video_name))

    court_pts = read_court(str(match_path / "court" / (video_name + ".out")))
    court_corners, pole_tips = court_points_to_corners_and_poles(court_pts)
    court = Court(corners = court_corners)
    court3d = Court3D(court_corners, pole_tips)

    if read_predicted_trajectory:
        trajectory_path = match_path / "ball_trajectory" / (str(video_name) + "_ball_predict.csv")
    else:
        trajectory_path = match_path / "ball_trajectory" / (str(video_name) + "_ball.csv")
    assert trajectory_path.is_file()
    trajectory = Trajectory(str(trajectory_path), interp=True)

    return {
        "poses": poses,
        "court": court,
        "court3d": court3d,
        "trajectory2d": trajectory
    }

def run_hit_detection_inference(video_path, trajectory, poses, court, hit_detection_model,
                                output_path):

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), "Error opening video stream or file"

    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = MLHitDetector(
        court,
        poses,
        trajectory,
        hit_detection_model,
        fps=fps
    )
    result, is_hit = detector.detect_hits()
    print("result = \n", result, len(result))
    print("is_hit = \n", is_hit, len(is_hit))

    # Write hit to csv file
    L = len(trajectory.X)
    frames = list(range(L))
    hits = [0] * L
    for fid, pid in zip(result, is_hit):
        hits[fid] = pid

    data = {'frame' : frames, 'hit' : hits}
    df = pd.DataFrame(data=data)
    print(df)
    df.to_csv(str(output_path), index=False)

def run_hit_detection(match_path, use_predicted_hits_trajectory=True):

    assert Path(HIT_DETECTION_MODEL).is_file()

    hit_output = match_path / "shot"
    hit_output.mkdir(parents=True, exist_ok=True)

    rally_dir = match_path / "rally_video"
    assert rally_dir.is_dir()

    for video_path in rally_dir.iterdir():
        if video_path.suffix != ".mp4":
            continue

        video_name = video_path.stem

        print(f"Processing video for hit detection: {video_name}")

        metadata = read_poses_court_trajectory(match_path, video_name, use_predicted_hits_trajectory)

        if use_predicted_hits_trajectory:
            output_path = hit_output / (video_path.stem + "_hit_predict.csv")
        else:
            output_path = hit_output / (video_path.stem + "_hit_predict_bootstraped.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        run_hit_detection_inference(video_path,
                                    metadata["trajectory2d"],
                                    metadata["poses"],
                                    metadata["court"],
                                    HIT_DETECTION_MODEL,
                                    output_path)

def run_3d_trajectory_reconstruction(match_path, use_predicted_hits_trajectory=True):

    def fix_gt_hits(hits_data, match_name, video_name):
        # 1 is bottom player, 2 is top player, alternate since this will be hand labeled hit frames
        ground_truth_start_hit_result = {
            ("test_match3", "1_02_00") : 2,
            ("test_match3", "1_03_02") : 1,
            ("test_match3", "1_05_02") : 2,
            ("test_match3", "1_05_03") : 2,
            ("test_match3", "1_06_05") : 2,
            ("test_match3", "1_06_06") : 2,
            ("test_match3", "1_08_08") : 2,
            ("test_match3", "1_08_09") : 2,
            #("test_match3", "1_09_12") : 1, # no gt hit
            ("test_match3", "1_09_15") : 1,
            ("test_match3", "1_10_16") : 2,
        }

        key = (match_name, video_name)
        if key in ground_truth_start_hit_result:
            new_hits_data = copy.copy(hits_data)
            start_hit_result = ground_truth_start_hit_result[key]
            count = 0
            for ii in range(new_hits_data.shape[0]):
                print(ii, new_hits_data.at[ii, "hit"])
                if new_hits_data.at[ii, "hit"] == 1:
                    count += 1
                    # if odd, use the result from start_hit_result
                    if count % 2 == 1:
                        new_hits_data.at[ii, "hit"] = start_hit_result
                    else:
                        new_hits_data.at[ii, "hit"] = (start_hit_result % 2) + 1
            return new_hits_data
        else:
            print("**WARNING** ground truth hits label not exist. Using TrackNet data. This results in no player info in hits (and thus trajectories all being on same side")
            return hits_data




    rally_dir = match_path / "rally_video"
    assert rally_dir.is_dir()

    for video_path in rally_dir.iterdir():
        if video_path.suffix != ".mp4":
            continue

        video_name = video_path.stem

        print(f"Processing video for 3d trajectory reconstruction: {video_name}")

        metadata = read_poses_court_trajectory(match_path, video_name, use_predicted_hits_trajectory)

        if use_predicted_hits_trajectory:
            hits_path = match_path / "shot" / (video_name + "_hit_predict.csv")
        else:
            hits_path = match_path / "shot" / (video_name + "_hit.csv")
        assert hits_path.is_file(), f"Hits file does not exist: {hits_path}"
        hits_data = pd.read_csv(str(hits_path))
        if not use_predicted_hits_trajectory:
            hits_data = fix_gt_hits(hits_data, match_path.stem, video_name)

        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"Cannot open video: {video_path}"
        fps = cap.get(cv2.CAP_PROP_FPS)

        reconstructor = RallyReconstructor(
            metadata["court3d"],
            metadata["poses"],
            metadata["trajectory2d"],
            hits_data
        )
        reconstruct_first_shot = True
        results = reconstructor.reconstruct(fps, reconstruct_first_shot)

        # write results
        if use_predicted_hits_trajectory:
            output_path = match_path / "ball_trajectory_3d" / (video_name + "_3d.csv")
        else:
            output_path = match_path / "ball_trajectory_3d_bootstraped" / (video_name + "_3d.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        N_frames = min(len(metadata["trajectory2d"].X), results.shape[0])
        data = {
            'frame' : list(range(N_frames)),
            'ball_x' : results[:,0].tolist(),
            'ball_y' : results[:,1].tolist(),
            'ball_z' : results[:,2].tolist()
        }
        pd.DataFrame(data=data).to_csv(str(output_path), index=False)

if __name__ == "__main__":

    base_dir = Path("/sensei-fs/users/juiwang/ai-badminton/data/tracknetv2_042022/profession_dataset")

    # training data
    #for match_idx in range(1, 23):
    #    print(f"\n\nComputing ML data for match_{match_idx}")
#
    #    match_dir = base_dir / f"match{match_idx}"
#
    #    #print("=== Running pose detection ===")
    #    #run_pose_detection_on_match(match_dir)
#
    #    #print("=== Running court detection ===")
    #    #run_court_detection_on_match(match_dir)
#
    #    #print("=== Running pose postprocessing ===")
    #    #run_pose_postprocessing(match_dir)
#
    #    #print("=== Running shuttle detection ===")
    #    #run_shuttle_detection(match_dir)
#
    #    print("=== Running shot detection ===")
    #    run_hit_detection(match_dir, False)
#
    #    print("=== Running 3D reconstruction ===")
    #    run_3d_trajectory_reconstruction(match_dir, False)

    ### validation data
    for match_idx in range(1, 4):
        print(f"\n\nComputing ML data for test_match_{match_idx}")

        # FIXME debug START
        if match_idx != 3:
            continue
        # FIXME debug END

        match_dir = base_dir / f"test_match{match_idx}"

        #print("=== Running pose detection ===")
        #run_pose_detection_on_match(match_dir)

        #print("=== Running court detection ===")
        #run_court_detection_on_match(match_dir)

        #print("=== Running pose postprocessing ===")
        #run_pose_postprocessing(match_dir)

        #print("=== Running shuttle detection ===")
        #run_shuttle_detection(match_dir)

        #print("=== Running shot detection ===")
        #run_hit_detection(match_dir, False)

        print("=== Running 3D reconstruction ===")
        run_3d_trajectory_reconstruction(match_dir, False)

    # debugging
    #run_shuttle_detection(Path("/home/juiwang/ai-badminton/data/tracknetv2_042022/profession_dataset/match1_cp"))
    #run_hit_detection(Path("/home/juiwang/ai-badminton/data/tracknetv2_042022/profession_dataset/match1_cp"))
    #run_3d_trajectory_reconstruction(Path("/home/juiwang/ai-badminton/data/tracknetv2_042022/profession_dataset/match1"))

