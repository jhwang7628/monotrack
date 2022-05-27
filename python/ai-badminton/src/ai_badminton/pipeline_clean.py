from tqdm import tqdm
import cv2

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

assert has_mmdet, 'Please install mmdet to run the demo.'

from pathlib import Path
import subprocess

####################################################################################################

MODEL_PATH = "/home/juiwang/ai-badminton/data/models"
DET_CONFIG = "/home/juiwang/ai-badminton/code/ai-badminton/mmpose/configs/faster_rcnn_r50_fpn_coco.py"
DET_CHECKPOINT = f"{MODEL_PATH}/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
POSE_CONFIG = "/home/juiwang/ai-badminton/code/ai-badminton/mmpose/configs/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
POSE_CHECKPOINT = f"{MODEL_PATH}/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"

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
    subprocess.check_call(cmd, shell=True)

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
        print("\n\n", video_path)
        output_file = pose_output / (video_path.stem + ".out")
        if output_file.is_file():
            continue

        run_mmpose(video_path, str(output_file), det_model, pose_model)

if __name__ == "__main__":

    base_dir = Path("/sensei-fs/users/juiwang/ai-badminton/data/tracknetv2_042022/profession_dataset")
    for match_idx in range(1, 23):
        match_dir = base_dir / f"match{match_idx}"
        run_pose_detection_on_match(match_dir)

