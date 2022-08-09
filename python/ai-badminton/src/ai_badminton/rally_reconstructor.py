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

    def reconstruct_one_hit(self, s2d, e2d, T, s3d=None, fps=25, fr_adjust=30./25, reconstructing_last_shot=False):
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
        velocity_bound = 120.0
        bounds = [(0, 6.1), (0, 13.4 / 2), (0.0, 3.0)] + [(-velocity_bound, velocity_bound)] + [(0, velocity_bound)] + [(-velocity_bound, velocity_bound)] +  [(0, 0.4)]
        if xg[1] > 13.4 / 2:
            bounds[1] = (13.4 / 2, 13.4)
            bounds[4] = (-velocity_bound, 0)

        camProj = self.court3d.camProj
        norm_scale = np.linalg.norm(camProj[:3, :3], ord=2)**(-2)
        loss_scale_reprojection = 1.0
        loss_scale_drift = 10.0
        loss_scale_out = 0.2
        loss_scale_net = 100.0 # cannot net unless it is the last shot. If use inf then inf*0.0 = nan so I chose a randomly large number
        if reconstructing_last_shot:
            loss_scale_net = 0.1
        loss_scale_net = 0.0 #FIXME debug. trying the constraint

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

            reprojection_loss = 0
            dt = fr_adjust / (fps * substeps)

            pord = 2
            out_loss = 0
            drift_loss = np.linalg.norm(x[:2] - s2d, ord=pord)**2
            net_loss = 0
            tid = T[0]
            
            count_reprojection = 0
            for t in range(1, substeps * N + 1):
                v += dt * (g - C * np.linalg.norm(v) * v)

                if t % substeps == 0:
                    q = camProj[:, :3] @ x + camProj[:, 3]
                    z = q[:2] / q[2]
                    if self.trajectory.X[tid] != 0 and self.trajectory.X[tid] == self.trajectory.X[tid]:
                        z_ = np.array([self.trajectory.X[tid], self.trajectory.Y[tid]])
                        reprojection_loss += np.linalg.norm(z - z_, ord=pord)**2
                        count_reprojection += 1

                    tid += 1
                    if tid == T[1]:
                        # Project current trajectory until it lands. If its too far out, add a penalty
                        xc = np.array(x)
                        vc = np.array(v)

                        while xc[2] > 0:
                            xc += dt * vc
                            vc += dt * (g - C * np.linalg.norm(vc) * vc)

                        out_amount = np.array([
                            max(0 - xc[0], xc[0] - 6.1, 0),
                            max(0 - xc[1], xc[1] - 13.4, 0),
                            max(0 - xc[2], xc[2] - 3., 0)
                        ])
                            
                        out_loss += np.linalg.norm(out_amount, ord=pord)**2
                        if not reconstructing_last_shot:
                            drift_loss += np.linalg.norm(x[:2] - e2d, ord=pord)**2
                dx = dt * v
                
                def delta_takes_trajectory_pass_net(x, dx):
                    return (x[1] - 13.41/2.0) * ((x[1] + dx[1]) - 13.41/2.0) < 0
                
                def point_on_net(x):
                    return x[2] >= 0 and x[2] <= (1.55 + 0.01)
                
                if delta_takes_trajectory_pass_net(x, dx) and point_on_net(x):
                    net_loss = 1.0
                x += dx
                
            total_loss = {
                "reprojection": norm_scale * reprojection_loss / count_reprojection,
                "drift": drift_loss,
                "out": out_loss,
                "net": net_loss
            }
            return (
                loss_scale_reprojection * total_loss["reprojection"] +
                loss_scale_drift * total_loss["drift"] +
                loss_scale_out * total_loss["out"] +
                loss_scale_net * total_loss["net"]
            )
        
        def constraint_ineq_higher_than_net_when_crossing(p):
            clearance = 0.03
            def delta_takes_trajectory_pass_net(x, new_x):
                return (x[1] - 13.41/2.0) * (new_x[1] - 13.41/2.0) < 0
            traj = np.array(get_trajectory(p))
            for idx, x in enumerate(traj[:-1]):
                old_x = x
                new_x = traj[idx+1]
                if delta_takes_trajectory_pass_net(old_x, new_x):
                    break
            return x[2] - (1.55 + clearance)
    
        def constraint_eq_endpoint_smooth(p):
            return p[:3] - xg
        
        def constraint_ineq_endpoint_in_other_court(p):
            traj = np.array(get_trajectory(p))
            for idx, x in enumerate(traj):
                if x[2] <= 0.0:
                    break
            if xg[1] > 13.41 / 2:
                return 13.41 / 2.0 - x[1]
            else:
                return x[1] - 13.41 / 2.0
            
        def constraint_ineq_above_ground(p):
            traj = np.array(get_trajectory(p))
            return min(traj[:,2]) 
        
        res = scipy.optimize.minimize(
            f,
            initg,
            bounds=bounds,
            constraints=[{
                "type": "ineq",
                "fun": constraint_ineq_higher_than_net_when_crossing
            }, {
                "type": "eq",
                "fun": constraint_eq_endpoint_smooth
            }, {
                "type": "ineq",
                "fun": constraint_ineq_endpoint_in_other_court
            }, {
                "type": "ineq",
                "fun": constraint_ineq_above_ground
            }
            ],
            method='SLSQP'
        )
        if not res.success:
            tqdm.tqdm.write("**WARNING** Optimization was not successful. Result = ", res)
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
            tqdm.tqdm.write(f"  reconstructing shot between frame {st[0]} and frame{en[0]}")
            s2d = get_location(st[1], st[0])
            e2d = get_location(en[1], en[0])

            reconstruct_last_shot = False
            if i == len(hit_frame) - 2:
                reconstruct_last_shot = True
            traj = self.reconstruct_one_hit(s2d, e2d, [st[0], en[0]], s3d, fps, fr_adjust, reconstruct_last_shot)
            last_pt = self.court3d.project_uv(traj[-1])
            last_pix = np.array([self.trajectory.X[en[0]], self.trajectory.Y[en[0]]])
            if (recon_first or i != 0):
                s3d = traj[-1]
            else:
                s3d = None
            all_traj.append(traj[:-1])
        all_traj = np.vstack(all_traj)
        if not recon_first:
            all_traj[0:hit_frame[1][0]] = -1
        return all_traj
