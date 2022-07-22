#!/usr/bin/env python

from ai_badminton.trajectory import read_trajectory_3d
from ai_badminton.hit_detector import read_hits

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np

from pathlib import Path

def draw_box_helper(ax, box = [[0, 0], [6.1, 13.41]]):

    x_low = box[0][0]
    y_low = box[0][1]

    x_top = box[1][0]
    y_top = box[1][1]

    ax.plot( [x_low, x_low], [y_low, y_top], [0, 0], color="k")
    ax.plot( [x_low, x_top], [y_low, y_low], [0, 0], color="k")
    ax.plot( [x_top, x_top], [y_low, y_top], [0, 0], color="k")
    ax.plot( [x_low, x_top], [y_top, y_top], [0, 0], color="k")


def draw_court_lines_3d(ax):
    draw_box_helper(ax, box=[[0, 0], [6.1, 13.41]])
    draw_box_helper(ax, box=[[0, 0], [6.1/2.0, 3.96+0.76]])
    draw_box_helper(ax, box=[[6.1/2.0, 0], [6.1, 3.96+0.76]])
    draw_box_helper(ax, box=[[0, 6.71+1.98], [6.1/2.0, 13.41]])
    draw_box_helper(ax, box=[[6.1/2.0, 6.71+1.98], [6.1, 13.41]])
    draw_box_helper(ax, box=[[0, 0], [0.45, 13.41]])
    draw_box_helper(ax, box=[[6.1-0.45, 0], [6.1-0.45, 13.41]])
    draw_box_helper(ax, box=[[0, 0], [6.1, 0.76]])
    draw_box_helper(ax, box=[[0, 13.41-0.76], [6.1, 13.41]])

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def is_trajectory_below_ground(traj):
    return (traj[:, 2] < 0).any()

def is_valid_trajectory(traj):
    return not is_trajectory_below_ground(traj)

dataset_root = Path("/home/juiwang/data/ai-badminton/dataset/profession_dataset")

match_path = dataset_root / "match1"

rallies = [x.stem for x in sorted((dataset_root / "match1" / "rally_video").glob("*.mp4"))]

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_box_aspect(aspect = (1,1,1))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
axisEqual3D(ax)
for rally in rallies:

    print(f"reading rally {rally}")

    trajectory3d = read_trajectory_3d(str(match_path / "ball_trajectory_3d" / (rally + "_3d.csv"))).values[:, 1:]
    hits = read_hits(match_path / "shot" / (rally + "_hit_predict.csv"))

    ids = hits["player_ids"]
    frames = hits["frames"]
    assert len(frames) == len(ids)

    N = len(frames)

    trajectories = []

    trajectories.append(trajectory3d[:frames[0], :])
    for ii in range(N-1):
        trajectories.append(trajectory3d[frames[ii]:frames[ii+1], :])
    if frames[-1] != len(trajectory3d):
        trajectories.append(trajectory3d[frames[-1]:, :])

    for idx, traj in enumerate(trajectories):
        if is_valid_trajectory(traj):
            ax.plot3D(
                traj[:,0],
                traj[:,1],
                traj[:,2],
                linewidth=2
            )
    draw_court_lines_3d(ax)

plt.show()
