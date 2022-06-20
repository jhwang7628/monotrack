import os
import tqdm.notebook as tq
import errno
import cv2
import numpy as np
from .pose import Pose
from .rally_reconstructor import *

COURT_IMG = "/home/juiwang/ai-badminton/code/ai-badminton/python/ai-badminton/src/ai_badminton/court.jpg"

'''
TODO: Figure out a clean way to encapsulate result, frame_lim, and is_hit.
Also, update trajectory.

Annotates an existing video (cap) with:
    - Player Poses (player_poses)
    - Court lines (court)
    - Shuttle trajectory (trajectory)
'''
def annotate_video(cap, 
                   court, 
                   poses, 
                   trajectory,
                   result=None,
                   is_hit=None,
                   frame_limit=None,
                   outfile='./output/output.mp4'):
    
    try:
        os.makedirs(os.path.dirname(outfile))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if trajectory is not None:
        Xb, Yb = trajectory.X, trajectory.Y
    # Write the hits to video, draw a dot right as the shuttle is struck
    outvid = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (width, height))

    court_img = cv2.imread(COURT_IMG)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    duration = 3
    bid = 0
    import tqdm.auto as tq

    L = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_limit:
        L = frame_limit
    if not frame_limit:
        if poses is not None:
            L = min(L, poses[0].values.shape[0])
        if trajectory is not None:
            L = min(L, len(Xb))
        
    for i in tq.tqdm(range(L)):
        ret, frame = cap.read()
        frame = court.draw_lines(frame)
        if poses is not None:
            player_poses = []
            for j in range(2):
                xy = poses[j].iloc[i].to_list()
                pose = Pose()
                pose.init_from_kparray(xy)
                player_poses.append(pose)
                frame = pose.draw_skeleton(frame, colour=(128, 128 + j * 127, 128))

        if trajectory is not None:
            centre = (int(Xb[i]), int(Yb[i]))
            radius = 5
            colour = (0, 255, 0)
            thickness = -1
            frame = cv2.circle(frame, centre, radius, colour, thickness)

        if result:
            if bid < len(result) and abs(result[bid] - i) < duration:
                radius = 4
                colour = (255, 0, 0)
                frame = cv2.circle(frame, centre, radius, colour, thickness)

                if i == result[bid]:
                    draw_hit = False
                    if 1 <= is_hit[bid] <= 2:
                        pid = is_hit[bid] - 1
                        # Use the athlete's feet position as where its hit
                        cp = np.array([Xb[i], Yb[i]])
                        lp, rp = player_poses[pid].kp[15], player_poses[pid].kp[16]
                        # Find point on segment closest to the hit point
                        du, dv = lp - cp, rp - lp
                        if np.dot(dv, dv) < 1e-2:
                            opt = 0.5
                        else:
                            opt = - np.dot(du, dv) / np.dot(dv, dv)
                            opt = min(1, max(0, opt))
                        proj = (1 - opt) * lp + opt * rp
                        hit_pos = court.pixel_to_court(proj)
                        
                        colour = (255,0,0)
                    else:
                        # Assume it hit the ground
                        hit_pos = court.pixel_to_court(centre)
                        colour = (0,0,255)
                    
                    hit_pos[hit_pos < 0] = 0
                    hit_pos[hit_pos > 1] = 1
                    court_img = court.draw_hit(court_img, hit_pos, colour)

                if i == result[bid] + duration - 1:
                    bid += 1

            frame[-court_img.shape[0]:, -court_img.shape[1]:] = court_img        
        outvid.write(frame)
    outvid.release()
    
'''
Annotatation for 3d reconstruction. Reprojects 3d trajectory back onto video.
'''
def annotate_video_3d(cap, 
                      court3d, 
                      trajectory3d,
                      frame_limit=None,
                      outfile='./output/output.mp4'):
    
    try:
        os.makedirs(os.path.dirname(outfile))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Write the hits to video, draw a dot right as the shuttle is struck
    outvid = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    import tqdm.auto as tq

    L = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), trajectory3d.values.shape[0])
    print(L)
    if frame_limit:
        L = frame_limit
 
    xyz = trajectory3d.values[:, 1:]
    for i in tq.tqdm(range(L)):
        if (xyz[i] == np.array([-1, -1, -1])).all() or xyz[i][2] < 0:
            continue
        ret, frame = cap.read()
        frame = court3d.draw_lines(frame)
        P = court3d.project_uv(xyz[i])
        centre = (int(P[0]), int(P[1]))
        radius = 5
        colour = (0, 255, 0)
        thickness = -1
        frame = cv2.circle(frame, centre, radius, colour, thickness)      
        outvid.write(frame)
    outvid.release()
