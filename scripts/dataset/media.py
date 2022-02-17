import cv2

def get_image_dimension(img):
    height, width, channels = img.shape
    return width, height

def get_fps(video_capture):
    return video_capture.get(cv2.CAP_PROP_FPS)

def get_duration(video_capture):
    fps = get_fps(video_capture)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count / fps

def extract_representative_frame(video_capture):
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = frame_count // 2
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = video_capture.read()
    assert ret == True
    return idx, frame
