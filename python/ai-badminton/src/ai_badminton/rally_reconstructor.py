from .court import Court
from .pose import *
from .trajectory_filter import *

import numpy as np
import scipy.optimize
import pandas as pd
import tqdm

def read_hits_file(filename):
    hits = pd.read_csv(filename)
    return hits

class Court3D:
    def __init__(self, corners, pole_tips):
        self.court2d = Court(corners)

        # The net
        self.left_post = [[0, 13.4/2, 0], [0, 13.4/2, 1.55]]
        self.right_post = [[6.1, 13.4/2, 0], [6.1, 13.4/2, 1.55]]
        self.net_line_top = [[0, 13.4/2, 1.55], [6.1, 13.4/2, 1.55]]
        self.net_line_bot = [[0, 13.4/2, 1.55 / 2.], [6.1, 13.4/2, 1.55 / 2.]]
        self.boundary = [[0, 0, 0], [6.1, 0, 0], [0, 13.4, 0], [6.1, 13.4, 0]]
        self.net = [self.left_post, self.right_post, self.net_line_top, self.net_line_bot]

        # Last two are poles
        image_points = corners + pole_tips
        object_points = self.boundary + self.net_line_top

        N = len(image_points)
        A = np.zeros((2*N, 11))
        y = np.zeros((2*N, 1))
        for i in range(N):
            for j in range(2):
                st = 4 if j > 0 else 0
                z = np.array(object_points[i])
                A[2*i+j, st:st+3] = z
                A[2*i+j, st+3] = 1
                A[2*i+j, -3:] = -image_points[i][j] * z
                y[2*i+j] = image_points[i][j]

        c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        C = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                if i == 2 and j == 3:
                    C[i,j] = 1
                else:
                    C[i,j] = c[4 * i + j]

        self.camProj = np.array(C)

    def pixel_to_court(self, coord):
        return self.court2d.pixel_to_court(coord)

    def project_uv(self, p3d):
        Z = self.camProj @ np.array([x for x in p3d] + [1])
        return Z[:2] / Z[2]

    def draw_lines(self, image):
        cimg = self.court2d.draw_lines(image)
        colour = (0, 0, 255)
        thickness = 3
        for line in [self.left_post, self.right_post, self.net_line_top]:
            z0, z1 = self.project_uv(np.array(line[0])), self.project_uv(np.array(line[1]))
            p0, p1 = tuple(z0.astype(int)), tuple(z1.astype(int))
            cimg = cv2.line(cimg, p0, p1, colour, thickness)
        return cimg

class RallyReconstructor:
    def __init__(self, court3d, poses, trajectory, hits, filter_trajectory=True):
        self.court3d = court3d
        self.poses = poses
        if filter_trajectory:
            self.trajectory = trajectory
        else:
            traj_filter = TrajectoryFilter()
            self.trajectory = traj_filter.filter_trajectory(trajectory)
        self.hits = hits

    def reconstruct_one_hit(self, s2d, e2d, T, s3d=None, fps=25, fr_adjust=30./25):
        substeps = 30
        g = np.array([0, 0, -9.8])
        height_guess = 1.7

        xg = np.array(s2d.tolist() + [height_guess])
        if s3d is not None:
            xg = np.array(s3d)

        N = T[1] - T[0]
        td = N / float(fps)
        vg = np.array(((e2d - s2d) / td).tolist() + [9.8 * td / 2 - 1. / td])
        Cg = 0.2

        initg = np.concatenate([xg, vg, np.array([Cg])])
        bounds = [(0, 6.1), (0, 13.4), (0.1, 6)] + [(-150, 150)] * 3 +  [(0, 0.4)]

        bounds = [(0, 6.1), (0, 13.4 / 2), (0.1, 6)] + [(-150, 150)] + [(0, 150)] + [(-150, 150)] +  [(0, 0.4)]
        if s2d[1] > 13.4 / 2:
            bounds[1] = (13.4 / 2, 13.4)
            bounds[4] = (-150, 0)

        camProj = self.court3d.camProj
        norm_scale = np.linalg.norm(camProj[:3, :3], ord=2)

        def get_trajectory(p):
            x = np.array(p[:3])
            v = np.array(p[3:-1])
            C = np.array(p[-1])

            dt = fr_adjust * 1. / (fps * substeps)
            res = []

            for t in range(substeps * N + 1):
                v += dt * (g - C * np.linalg.norm(v) * v)
                if t % substeps == 0:
                    res.append(np.array(x))
                x += dt * v
            return res

        def f(p, debug=False):
            x = np.array(p[:3])
            if s3d is not None:
                x = np.array(s3d)
            v = np.array(p[3:-1])
            C = np.array(p[-1])

            loss = 0
            dt = fr_adjust / (fps * substeps)

            pord = 6
            drift_loss, out_loss = 0, 0
            tid = T[0]

            for t in range(1, substeps * N + 1):
                v += dt * (g - C * np.linalg.norm(v) * v)

                if t % substeps == 0:
                    q = camProj[:, :3] @ x + camProj[:, 3]
                    z = q[:2] / q[2]
                    if self.trajectory.X[tid] != 0 and self.trajectory.X[tid] == self.trajectory.X[tid]:
                        z_ = np.array([self.trajectory.X[tid], self.trajectory.Y[tid]])
                        loss += np.linalg.norm(z - z_, ord=pord)**pord

                    tid += 1
                    if tid == T[1]:
                        # Project current trajectory until it lands. If its too far out, add a penalty
                        xc = np.array(x)
                        vc = np.array(v)

                        while xc[2] > 0:
                            xc += dt * vc
                            vc += dt * (g - C * np.linalg.norm(vc) * vc)

                        out_loss += max(0 - xc[0], xc[0] - 6.1, 0)**pord
                        out_loss += max(0 - xc[1], xc[1] - 13.4, 0)**pord
                        out_loss += max(0 - xc[2], xc[2] - 3., 0)**pord
                        drift_loss += np.linalg.norm(x[:2] - e2d)**pord
                x += dt * v

            pinv = 1. / pord
            return loss**pinv + (norm_scale * drift_loss)**pinv

        res = scipy.optimize.minimize(
            f, initg, bounds=bounds,
            method='SLSQP'
        )
        est_traj = np.array(get_trajectory(res.x))
        return est_traj

    def reconstruct(self, fps, recon_first=False):
        fr_adjust = 30. / fps

        def get_location(player_id, frame_id):
            assert 1 <= player_id <= 2
            xy = self.poses[player_id-1].iloc[frame_id].tolist()
            pose = Pose()
            pose.init_from_kparray(xy)

            mid_pt = pose.get_base()
            court_pt = self.court3d.pixel_to_court(mid_pt)
            court_pt = [court_pt[0] * 6.1, court_pt[1] * 13.4]
            return np.array(court_pt)

        hit_frame = [(i, x) for i, x in enumerate(self.hits.hit) if x > 0]
        if hit_frame[0][0] != 0:
            hit_frame = [(0, 3-hit_frame[0][1])] + hit_frame

        s3d = None
        all_traj = []
        for i in tqdm.tqdm(range(len(hit_frame) - 1)):
            st, en = hit_frame[i], hit_frame[i+1]
            s2d = get_location(st[1], st[0])
            e2d = get_location(en[1], en[0])

            traj = self.reconstruct_one_hit(s2d, e2d, [st[0], en[0]], s3d, fps, fr_adjust)
            last_pt = self.court3d.project_uv(traj[-1])
            last_pix = np.array([self.trajectory.X[en[0]], self.trajectory.Y[en[0]]])
            if np.linalg.norm(last_pt - last_pix) < 40 and (recon_first or i != 0):
                s3d = traj[-1]
            else:
                s3d = None
            all_traj.append(traj[:-1])
        all_traj = np.vstack(all_traj)
        if not recon_first:
            all_traj[0:hit_frame[1][0]] = -1
        return all_traj
