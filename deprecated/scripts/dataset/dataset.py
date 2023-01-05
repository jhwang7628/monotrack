from pathlib import Path

ALLOWED_EXTENSIONS_DEFAULT = ["mp4", "webm"]

def get_video_paths(root:Path, allowed_extensions:list=ALLOWED_EXTENSIONS_DEFAULT):
    video_files = []
    for ext in allowed_extensions:
        video_files.extend(list(root.glob(f"**/*.{ext}")))
    return video_files
