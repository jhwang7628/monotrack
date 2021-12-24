from dataset import *

import sys
from pathlib import Path

root = Path(sys.argv[1])

pp = get_video_paths(root, ["mp4"])
for p in pp:
    print(str(p))
