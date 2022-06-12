import pandas as pd
import numpy as np

class Trajectory(object):
    def __init__(self, filename, interp=True):
        # Get poses and trajectories
        trajectory = pd.read_csv(filename)

        if interp:
            trajectory[trajectory.X == 0] = float('nan')
            trajectory[trajectory.Y == 0] = float('nan')
            trajectory = trajectory.assign(X_pred=trajectory.X.interpolate(method='slinear'))
            trajectory = trajectory.assign(Y_pred=trajectory.Y.interpolate(method='slinear'))

            trajectory.fillna(method='bfill', inplace=True)
            trajectory.fillna(method='ffill', inplace=True)

            Xb, Yb = trajectory.X_pred.tolist(), trajectory.Y_pred.tolist()
        else:
            Xb, Yb = trajectory.X.tolist(), trajectory.Y.tolist()

        self.X = Xb
        self.Y = Yb

def read_trajectory_3d(file_path):
    return pd.read_csv(str(file_path))
