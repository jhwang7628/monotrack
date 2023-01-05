import cv2
import torch
import numpy as np

from pipeline.pipeline import Pipeline

class WritePose(Pipeline):
    """Pipeline task for writing poses to file."""

    def __init__(self, dst, outfile='pose.out'):
        self.dst = dst
        self.cpu_device = torch.device("cpu")
        self.output_file = open(outfile, 'w')

        super().__init__()

    def map(self, data):
        self.annotate_poses(data)
        return data

    def annotate_poses(self, data):
        predictions = data["predictions"]
        instances = predictions["instances"]
        keypoints = instances.pred_keypoints.cpu().numpy()
        l_pairs = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (6, 12), (5, 11), (11, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]

        self.output_file.write('frame %d\n' % data['frame_num'])
        for idx in range(keypoints.shape[0]):
            self.output_file.write('pose %d\n' % idx)
            instance_keypoints = keypoints[idx]
            l_points = {}
            p_scores = {}
            # Draw keypoints
            for n in range(instance_keypoints.shape[0]):
                score = instance_keypoints[n, 2]
                if score <= 0.05:
                    continue
                cor_x = int(instance_keypoints[n, 0])
                cor_y = int(instance_keypoints[n, 1])
                l_points[n] = (cor_x, cor_y)
                p_scores[n] = score
                self.output_file.write("%d %d %d %f\n" % (n, cor_x, cor_y, score))
