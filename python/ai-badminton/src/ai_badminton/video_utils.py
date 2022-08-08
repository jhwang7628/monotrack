import cv2
import numpy as np

def read_video_assert(video_path):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Video cannot be opened: {video_path}"
    return cap

def get_total_num_frames(cap):
    return cap.get(cv2.CAP_PROP_FRAME_COUNT)

def get_fps(cap):
    return cap.get(cv2.CAP_PROP_FPS)

def frame_time_to_frame_index(cap, time):
    idx = int(np.floor(time * get_fps(cap)))
    assert idx < get_total_num_frames(cap), "Frame time exceeded total number of frames"
    return idx

def read_frame(frame_idx, cap, BGR2RGB=False):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret and BGR2RGB:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ret, frame