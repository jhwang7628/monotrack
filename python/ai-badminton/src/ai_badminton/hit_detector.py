import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .pose import Pose

class HitDetector(object):
    def __init__(self, court, poses, trajectory):
        self.court = court
        self.poses = poses
        self.trajectory = trajectory
        
    # Returns the hits in the given trajectory as well as who hit it
    # Output is are two lists of values:
    #   - List of frame ids where things are hit
    #   - 0 (no hit), 1 (bottom player hits), 2 (top player hits)
    def detect_hits(self):
        pass
    
class AdhocHitDetector(HitDetector):
    def __init__(self, poses, trajectory):
        super().__init__(None, poses, trajectory)  
    
    def _detect_hits_1d(self, z, thresh=4, window=8):
        # For a hit to be registered, the point must be a local max / min and
        # the slope must exceed thresh on either the left or right side
        # The slope is averaged by the window parameter to remove noise
        z = np.array(z)
        bpts = []
        for i in range(window+1, len(z)-window-1):
            if (z[i]-z[i-1]) * (z[i]-z[i+1]) < 0:
                continue

            # This is a local opt
            left = abs(np.median(z[i-window+1:i+1] - z[i-window:i]))
            right = abs(np.median(z[i+1:i+window+1] - z[i:i+window]))
            if max(left, right) > thresh:
                bpts.append(i)
        return bpts

    def _merge_hits(self, x, y, closeness=2):
        bpt = []
        for t in sorted(x + y):
            if len(bpt) == 0 or bpt[-1] < t - closeness:
                bpt.append(t)
        return bpt

    def _detect_hits(self, x, y, thresh=10, window=7, closeness=15):
        return self._merge_hits(
            self._detect_hits_1d(x, thresh, window),
            self._detect_hits_1d(y, thresh, window),
            closeness
        )
    
    def detect_hits(self, fps=25):
        Xb, Yb = self.trajectory.X, self.trajectory.Y
        result = self._detect_hits(Xb, Yb)
    
        # Filter hits by pose
        is_hit = []
        last_hit = -1
        # Filter hits by velocity
        avg_hit = np.average(np.diff(result))
        for i, fid in enumerate(result):
            if i+1 < len(result) and result[i+1] - fid > 1.6 * avg_hit:
                is_hit.append(0)
                continue
                
            if fid > self.poses[0].values.shape[0]:
                break

            c = np.array([Xb[fid], Yb[fid]])
            reached_by = 0
            dist_reached = 1e99
            for j in range(2):
                xy = self.poses[j].iloc[fid].to_list()
                pose = Pose()
                pose.init_from_kparray(xy)
                if pose.can_reach(c):
                    pdist = np.linalg.norm(c - pose.get_centroid())
                    if not reached_by or reached_by == last_hit or pdist < dist_reached:
                        reached_by = j + 1
                        dist_reached = pdist

            if reached_by:
                last_hit = reached_by
            is_hit.append(reached_by)

        print('Total shots hit by players:', sum(x > 0 for x in is_hit))
        print('Total impacts detected:', len(result))
        print('Distribution of shot times:')
        plt.hist(np.diff(result))
        print('Average time between shots (s):', np.average(np.diff(result)) / fps)
        return result, is_hit

class MLHitDetector(HitDetector):
    def __init__(self, court, poses, trajectory, model_file):
        super().__init__(court, poses, trajectory)
    
        self.model = tf.keras.models.load_model(model_file)
        
        import tensorflow.keras.backend as K

        trainable_count = np.sum([K.count_params(w) for w in self.model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in self.model.non_trainable_weights])

        print('Number of layers:', len(self.model.layers))
        print('Total params: %d' % (trainable_count + non_trainable_count))
        print('Trainable params: %d' % trainable_count)
        print('Non-trainable params: %d' % non_trainable_count)
        
    def detect_hits(self, fps=25):
        Xb, Yb = self.trajectory.X, self.trajectory.Y
        num_consec = int(self.model.input_shape[1] // (2 * (34 + 4 + 1)))
        court_pts = self.court.corners
        corner_coords = np.array([court_pts[1], court_pts[2], court_pts[0], court_pts[3]]).flatten()

        bottom_player = self.poses[0]
        top_player = self.poses[1]

        corners = np.array([court_pts[1], court_pts[2], court_pts[0], court_pts[3]]).flatten()

        x_list = []
        L = min(bottom_player.values.shape[0], len(Xb))
        for i in range(num_consec):
            end = L-num_consec+i+1
            x_bird = np.array(list(zip(Xb[i:end], Yb[i:end])))
            x_pose = np.hstack([bottom_player.values[i:end], top_player.values[i:end]])
            x = np.hstack([x_bird, x_pose, np.array([corners for j in range(i, end)])])
            
            x_list.append(x)
        x_inp = np.hstack(x_list)
        
        y_pred = self.model.predict(x_inp)
        
        detections = np.where(y_pred[:,0] < 0.05)[0]
        result, clusters, who_hit = [], [], []
        min_x, max_x = np.min(court_pts, axis=0)[0], np.max(court_pts, axis=0)[0]
        for t in detections:
            # Filter based on time
            if len(clusters) == 0 or clusters[-1][0] < t - fps / 2.5:
                clusters.append([t])
            else:
                clusters[-1].append(t)

        delta = 0.1 * (max_x - min_x)
        for cluster in clusters:
            # Filter based on whether any part of the cluster is outside
            any_out = False
            votes = np.array([0., 0., 0.])
            for t in cluster:
                if Xb[t] < min_x + delta or Xb[t] > max_x - delta:
                    any_out = True
                    break
                votes += y_pred[t]
            if not any_out:
                votes[0] = 0
                votes[1] /= 3
                result.append(int(np.median(cluster)))
                who_hit.append(int(np.argmax(votes)))

        is_hit = []
        avg_hit = np.average(np.diff(result))
        last_hit, last_time = -1, -1
        to_delete = [0] * len(result)
        for i, fid in enumerate(result):
            if i+1 < len(result) and result[i+1] - fid > 1.6 * avg_hit:
                is_hit.append(0)
                continue
            if i >= len(who_hit):
                break
                
            # Another filter: prevent two hits in a row by the same person within 1s
            if fid - last_time < fps and last_hit == who_hit[i]:
                to_delete[i] = 1
                continue
                
            is_hit.append(who_hit[i])
            last_time = fid
            last_hit = who_hit[i]
            
        result = [r for i, r in enumerate(result) if not to_delete[i]]
        
        print('Total shots hit by players:', sum(x > 0 for x in is_hit))
        print('Total impacts detected:', len(result))
        print('Distribution of shot times:')
        plt.hist(np.diff(result))
        print('Average time between shots (s):', np.average(np.diff(result)) / fps)
        return result, is_hit