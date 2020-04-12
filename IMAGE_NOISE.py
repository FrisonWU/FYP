from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.linalg import  block_diag
from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
import cv2
from numpy.random import randn
import sympy

class Kalman():
        def __init__(self, pos=[0, 0], vel=[0, 0], noise_std=1.):
            self.vel = vel
            self.noise_std = noise_std
            self.pos = [pos[0], pos[1]]
            self.pos[0]=0
            self.pos[1]=0
            self.vel[0]=0
            self.vel[1]=0
            self.noise_std = 1

        def Kalman1(self):
            pos = self.pos
            dt = 0.01
            rk = KalmanFilter(dim_x=4, dim_z=2)
            #radar = RadarSim(dt, pos=0., vel=100., alt=1000.)
            # make an imperfect starting guess
            rk.x = np.array([pos[0]-10, self.vel[0]+10, pos[1]+10,self.vel[1]-10]).T
            rk.F = np.eye(4) + np.array([[0, 1, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 1],
                                         [0, 0, 0, 0]]) * dt
            range_std = 5 # 5 pixels
            rk.R = np.eye(2) * 200 ** 2
            q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1 ** 2)
            rk.Q = block_diag(q, q)
            rk.P *= 1000
            rk.H = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
            return rk



'''
for i in range (30):
    z = Kalman().read()
    lk = Kalman().Kalman1()
    lk.update(z)
    print(z)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''