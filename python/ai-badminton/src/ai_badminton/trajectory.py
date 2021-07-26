import pandas as pd
import numpy as np

class Trajectory(object):
    def __init__(self, filename):
        # Get poses and trajectories
        trajectory = pd.read_csv(filename)
        
        trajectory[trajectory.X == 0] = float('nan')
        trajectory[trajectory.Y == 0] = float('nan')

        trajectory = trajectory.assign(X_pred=trajectory.X.interpolate(method='slinear'))
        trajectory = trajectory.assign(Y_pred=trajectory.Y.interpolate(method='slinear'))

        trajectory.fillna(method='bfill', inplace=True)
        trajectory.fillna(method='ffill', inplace=True)

        Xb, Yb = trajectory.X_pred.tolist(), trajectory.Y_pred.tolist()
        xa, ya = np.average(np.abs(np.diff(Xb))), np.average(np.abs(np.diff(Yb)))
        t_dist = xa**2. + ya**2.

        for t in range(2):
            for i in range(1, len(Xb)-1):
                d0 = (Xb[i+1] - Xb[i])**2. + (Yb[i+1] - Yb[i])**2.
                d1 = (Xb[i-1] - Xb[i])**2. + (Yb[i-1] - Yb[i])**2.
                if (d0 + d1) / 2. > 1.5 * t_dist:
                    Xb[i] = (Xb[i-1] + Xb[i+1]) / 2.
                    Yb[i] = (Yb[i-1] + Yb[i+1]) / 2.
                    
        self.X = Xb
        self.Y = Yb
