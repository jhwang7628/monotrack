from .pose import Pose

import numpy as np

from pathlib import Path

'''
Input:
    - trajectory of the shots
    - poses of both players
    - an array indicating for each frame whether there was a hit or not
'''
def generate_sequence(court, traj, poses, hits):
    Xb, Yb = traj.X, traj.Y
    hit_frame, who_hit = hits[0], hits[1]

    def get_position(pose, x, y):
        cp = np.array([x, y])
        lp, rp = pose.kp[15], pose.kp[16]
        if abs(cp[0] - lp[0]) < abs(cp[0] - rp[0]):
            hit_pos = court.pixel_to_court(lp)
        else:
            hit_pos = court.pixel_to_court(rp)
        return hit_pos

    positions = []
    for fid, hid in zip(hit_frame, who_hit):
        player_poses = []
        xb, yb = Xb[fid], Yb[fid]
        for j in range(2):
            xy = poses[j].iloc[fid].to_list()
            pose = Pose()
            pose.init_from_kparray(xy)
            player_poses.append(pose)

        if hid == 0:
            continue
        if hid <= 2:
            # Get location of where shuttle was hit from foot position
            pos = get_position(player_poses[hid-1], xb, yb)
        else:
            pos = court.pixel_to_court(np.array([xb, yb]))
        positions.append((fid, pos))
    return positions

'''
We'll do six zones per side:
    - Front, Middle, Back divided into halves.
    - We'll arbitrarily assign 25%, 35%, 40% to the Front, Back, Middle
    - Two sides x Six zones per side = 12 possible ids
'''
def zone(pos):
    # Give the bottom player coords 0-5, top gets 6-11
    # y-coord < 0.5 is the bottom of the screen
    side = -1
    x, y = pos[0], pos[1]
    if pos[1] <= 0.5:
        side = 0
    else:
        side = 1
        y = 1-y
        x = 1-x

    y_id = -1
    if 0 <= y < 0.35 / 2:
        y_id = 0
    elif 0.25 / 2 <= y < 0.65 / 2:
        y_id = 1
    else:
        y_id = 2

    x_id = 0 if x <= 0.5 else 1
    zone_id = 2 * y_id + x_id + 6 * side
    return zone_id

'''
Ideally, this should filter should:
    - Be associated with a database of shot sequences
    - Support adding locations to the filter
    - Our application should support progressively adding more shots as well as
      choosing a sequence and constantly updating it.
'''
class ShotFilter(object):
    def __init__(self, sequences):
        self.result = list(sequences)
        self.search_seq = []
    def add_shot(self, pos):
        L = len(self.search_seq)
        filtered = []
        for seq in self.result:
            if len(seq[2]) < L + 1:
                continue
            if zone(pos) == zone(seq[2][L][1]):
                filtered.append(seq)

        self.search_seq.append(pos)
        self.result = filtered

    def get_sequences(self):
        return self.result
