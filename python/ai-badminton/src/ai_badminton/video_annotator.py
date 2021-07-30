import os
import tqdm.notebook as tq
import errno
import cv2
import numpy as np
from .pose import Pose

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
        os.makedirs('output')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    Xb, Yb = trajectory.X, trajectory.Y
    # Write the hits to video, draw a dot right as the shuttle is struck
    outvid = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc('M','P','4','V'), 10, (width, height))

    court_img = cv2.imread('court.jpg')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    duration = 3
    bid = 0
    import tqdm.notebook as tq

    L = frame_limit
    if not frame_limit:
        L = min(poses[0].values.shape[0], len(Xb))
        
    for i in tq.tqdm(range(L)):
        ret, frame = cap.read()
        frame = court.draw_lines(frame)
        player_poses = []
        for j in range(2):
            xy = poses[j].iloc[i].to_list()
            pose = Pose()
            pose.init_from_kparray(xy)
            player_poses.append(pose)
            frame = pose.draw_skeleton(frame, colour=(128, 128 + j * 127, 128))

        centre = (int(Xb[i]), int(Yb[i]))
        radius = 5
        colour = (0, 255, 0)
        thickness = -1
        frame = cv2.circle(frame, centre, radius, colour, thickness)

        if result:
            if bid < len(result) and abs(result[bid] - i) < duration:
                radius = 7
                colour = (255, 0, 0)
                frame = cv2.circle(frame, centre, radius, colour, thickness)

                if i == result[bid]:
                    draw_hit = False
                    if is_hit[bid]:
                        pid = is_hit[bid] - 1
                        # Use the athlete's feet position as where its hit
                        # Find the foot closer to the shuttle
                        cp = np.array([Xb[i], Yb[i]])
                        lp, rp = player_poses[pid].kp[15], player_poses[pid].kp[16]
                        if np.linalg.norm(cp - lp) < np.linalg.norm(cp - rp):
                            hit_pos = court.pixel_to_court(lp)
                        else:
                            hit_pos = court.pixel_to_court(rp)
                        colour = (255,0,0)
                        draw_hit = True

                        if min(hit_pos) < 0 or max(hit_pos) > 1:
                            draw_hit = False
                    else:
                        # Assume it hit the ground
                        hit_pos = court.pixel_to_court(centre)
                        colour = (0,0,255)
                        if min(hit_pos) < 0 or max(hit_pos) > 1:
                            draw_hit = False
                    if draw_hit:
                        court_img = court.draw_hit(court_img, hit_pos, colour)

                if i == result[bid] + duration - 1:
                    bid += 1

            frame[-court_img.shape[0]:, -court_img.shape[1]:] = court_img        
        outvid.write(frame)
    outvid.release()