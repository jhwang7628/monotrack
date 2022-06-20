import cv2

def read_video_assert(video_path):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Video cannot be opened: {video_path}"
    return cap

def get_total_num_frames(cap):
    return cap.get(cv2.CAP_PROP_FRAME_COUNT)

def get_fps(cap):
    return cap.get(cv2.CAP_PROP_FPS)
