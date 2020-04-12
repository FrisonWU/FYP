import numpy as np
import math
import MAG_TinyEKF as IMG_EKF
#import RealTimeMassCenter as Rt

class Kalman(IMG_EKF.EKF):
    def __init__(self):
        self.n_state = 2  # (a,da, b, db c,dc, q0,dq0, q1,dq1 q2,dq2 q3,dq3 m)X2
        self.m_ob = 2  # (Bx, By, Bz) X 9 slaves
        self.Default_rval = 0.02
        self.Default_qval = 0.01
        #self.scale = 1e2
        self.dt = 0.001
        #self.m = 2000  # moment unit : I*cm^2

        IMG_EKF.EKF.__init__(self, n=self.n_state, m=self.m_ob, rval=self.Default_rval, qval=self.Default_qval, pval=50,
                             EnableAdaptive=0)
        self.x = np.array([10,10])

    def start(self, zx,zy):
        """Magnetometer sensors fusion using EKF"""

        IMG_Data = [zx,zy]
        oldx, oldy = self.x[0:2]

        # pnorm = np.mean(self.P_post)*20
        # if abs(zx-oldx) > pnorm or abs(zy-oldy)>pnorm:
        #     return oldx,oldy

        self.ekf_step(IMG_Data)
        x,y= self.x[0:2]
        #print('velocity x is %d',vx,'velocity y is %d',vy)
        return x,y

    def h(self, x):
        """计算 measurement function & Jacob"""
        return x, np.eye(2)

    def f(self, x):
        """计算state translation function Jacob, 并返回计算结果"""
        self.F_func =np.eye(2)
        return np.dot(self.F_func, x), self.F_func  # 返回格式为 f,F
