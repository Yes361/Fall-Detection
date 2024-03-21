import numpy as np
from mp_solutions import specific_points

class KalmanFilter:

    def __init__(self, dt, point):
        self.dt = dt
        self.E = np.matrix([[point[0]], [point[1]], [0], [0], [0], [0]])
        self.A = np.matrix([[1, 0, self.dt, 0, 0.5*self.dt**2, 0],
                            [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
                            [0, 0, 1, 0, self.dt, 0],
                            [0, 0, 0, 1, 0, self.dt],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])
        self.H = np.matrix([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0]])
        self.Q = np.identity(6)
        self.R = np.identity(2)
        self.P = np.eye(6)

    def predict(self):
        self.E = self.A @ self.E
        self.P = (self.A @ self.P) @ self.A.T + self.Q
        return self.E

    def update(self, z):
        S = self.H @ (self.P @ self.H.T) + self.R
        K = (self.P @ self.H.T) @ np.linalg.inv(S)
        self.E = np.round(self.E + K @ (z - self.H @ self.E))
        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P


KF_points = {}
def initialize():
    for name, val in specific_points.items():
        KF_points[name] = KalmanFilter(0.1, [0,0])

def get_acceleration(frame, landmarks):
    acceleration = {}
    for name, KF in KF_points.items():
      state = KF.predict()
      acceleration[name] = (state[4,0],state[5,0])
      #print(state)
      KF.update(
        np.expand_dims(
        [
            landmarks[specific_points[name]].x * frame.shape[1], 
            landmarks[specific_points[name]].y * frame.shape[0]
        ], axis=-1))
    return acceleration