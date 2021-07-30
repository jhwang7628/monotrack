from collections import defaultdict
import cv2
import os
import numpy as np
import csv
import pandas as pd
import errno

class Pose:
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (6, 12), (5, 11), (11, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
        "left_wrist", "right_wrist", "left_hip", "right_hip", 
        "left_knee", "right_knee", "left_ankle", "right_ankle" ]
    
    def __init__(self, kplines=[]):
        if not kplines:
            return
        
        keypoints = []
        for line in kplines:
            i, px, py, score = [float(x) for x in line.split()]
            keypoints.append((int(i), np.array([px, py])))
        self.init_from_kp(keypoints)
    
    def init_from_kparray(self, kparray):
        kp = np.array(kparray).reshape((17, 2))
        keypoints = []
        for i in range(17):
            keypoints.append((i, kp[i]))
        self.init_from_kp(keypoints)
        
    # Each pose has 17 key points, representing the skeleton
    def init_from_kp(self, keypoints):
        # Keypoints should be tuples of (id, point)
        self.kp = np.empty((17, 2))
        self.kp[:] = np.NaN

        for i, p in keypoints:
            self.kp[i] = p
        
        self.bx = [min(self.kp[:, 0]), max(self.kp[:, 0])]
        self.by = [min(self.kp[:, 1]), max(self.kp[:, 1])]
        
    
    def draw_skeleton(self, img, colour=(0, 128, 0), thickness=5):
        cimg = img.copy()
        for line in self.skeleton:
            X, Y = self.kp[line[0]], self.kp[line[1]]
            if any(np.isnan(X)) or any(np.isnan(Y)):
                continue
            p0, p1 = tuple(X.astype(int)), tuple(Y.astype(int))
            # For the legs, colour them and the ankles separately
            if line == (13, 15) or line == (14, 16):
                cimg = cv2.line(cimg, p0, p1, (0, 128, 128), thickness)
                cimg = cv2.circle(cimg, p1, 3, (128, 128, 0), thickness=-1)
            else:
                cimg = cv2.line(cimg, p0, p1, colour, thickness)
        return cimg
    
    def get_base(self):
        # Returns the midpoint of the two ankle positions
        return (self.kp[15] + self.kp[16]) / 2.
    
    def get_centroid(self):
        n = 0
        p = np.zeros((2,))
        for i in range(17):
            if any(np.isnan(self.kp[i])) or max(self.kp[i]) == 0:
                continue
            
            n += 1
            p += self.kp[i]
        return p / n
    
    def can_reach(self, p, epsx=1.5, epsy=1.5):
        # if within (1+/-eps) of the bounding box then we can reach it
        dx, dy = self.bx[1] - self.bx[0], self.by[1] - self.by[0]
        return self.bx[0] - epsx * dx < p[0] < self.bx[1] + epsx * dx and \
               self.by[0] - epsy * dy < p[1] < self.by[1] + epsy * dy
    
def read_player_poses(input_prefix):
    bottom_player = pd.read_csv(input_prefix + '_player_bottom.csv')
    top_player = pd.read_csv(input_prefix + '_player_top.csv')
    
    bottom_player.drop('frame', axis=1, inplace=True)
    top_player.drop('frame', axis=1, inplace=True)
    bottom_player.fillna(0, inplace=True)
    top_player.fillna(0, inplace=True)
    
    poses = [bottom_player, top_player]
    return poses
        
'''
Processes the raw text pose output from the pose estimation scripts.
'''
def process_pose_file(input_path, output_prefix, court):
    poses = open(input_path)
    filtered_poses = defaultdict(dict)

    frame_id = -1
    pose_lines = []

    def filter_pose(fid, kplines):
        if not kplines:
            return
        pose = Pose(kplines)
        in_court = court.in_court(pose.get_base())# or court.in_court(pose.get_centroid())
        if in_court:
            filtered_poses[fid][in_court] = pose

    def process_poses(fid, lines):
        if not lines:
            return

        kplines = []
        for line in lines:
            if 'pose' in line:
                filter_pose(fid, kplines)
                kplines = []
            else:
                kplines.append(line)
        filter_pose(fid, kplines)

    print('Read in files. Processing poses...')
    
    import tqdm.notebook as tq
    lines = poses.readlines()
    for lid in tq.tqdm(range(len(lines))):
        line = lines[lid]
        if 'frame' in line:
            process_poses(frame_id, pose_lines)
            pose_lines = []
            frame_id += 1
        else:
            pose_lines.append(line)
    
    process_poses(frame_id, pose_lines)
    
    try:
        os.makedirs('output')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    print('Separating top and bottom poses...')
    # We have all the poses now, lets write the player poses to distinct CSV files
    for pid, player in enumerate(['bottom', 'top']):
        filename = output_prefix + '_player_%s.csv' % player
        file = open(filename, 'w')
        player_writer = csv.writer(file)

        row_names = ['frame']
        for joint in Pose.joint_names:
            row_names.append(joint + '_x')
            row_names.append(joint + '_y')

        player_writer.writerow(row_names)
        for fid in range(frame_id):
            row = [fid] + [np.NaN] * 34
            if pid + 1 in filtered_poses[fid]:
                for i, z in enumerate(filtered_poses[fid][pid + 1].kp):
                    row[2*i+1] = z[0]
                    row[2*i+2] = z[1]
            player_writer.writerow([str(s) for s in row])
        file.close()

        # Lets interpolate all the missing values
        player = pd.read_csv(filename)
        player.interpolate(method='slinear', inplace=True)
        player.fillna(method='bfill', inplace=True)
        player.to_csv(filename, index=False)

    bottom_player = pd.read_csv(output_prefix + '_player_bottom.csv')
    top_player = pd.read_csv(output_prefix + '_player_top.csv')
    players = [bottom_player, top_player]
    print('Done!')
    
    return players