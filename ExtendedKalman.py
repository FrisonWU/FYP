import numpy as np
import math
import MAG_TinyEKF as IMG_EKF
from sympy import *
#import RealTimeMassCenter as Rt

class ExtendedKalman(IMG_EKF.EKF):
    def __init__(self):
        self.n_state = 7  # (a,da, b, db c,dc, q0,dq0, q1,dq1 q2,dq2 q3,dq3 m)X2
        self.m_ob = 2  # (Bx, By, Bz) X 9 slaves
        self.Default_rval = 10
        self.Default_qval = 0.01
        #self.scale = 1e2
        self.dt = 0.001
        #self.m = 2000  # moment unit : I*cm^2
        self.h_func = np.zeros((self.m_ob, self.n_state), np.float)  # Measurement function derivative
        self.H_Jacob = np.zeros((self.m_ob, self.n_state), np.float)  # Measurement function derivative
        self.F_func = np.zeros((self.n_state, self.n_state), np.float)

        IMG_EKF.EKF.__init__(self, n=self.n_state, m=self.m_ob, rval=self.Default_rval, qval=self.Default_qval, pval=50,
                             EnableAdaptive=0)
        self.x = np.array([[10],[10],[10],[1],[1],[1],[1]])
        self.ix = 320
        self.iy = 240

    def start(self, zx,zy):
        """Magnetometer sensors fusion using EKF"""

        IMG_Data = [zx,zy]
        self.ekf_step(IMG_Data)
        wx, wy, wz, q1, q2, q3, q4= self.x[0:7]
        #print('velocity x is %d',vx,'velocity y is %d',vy)
        if self.ix >=0 and self.ix <=640 and self.iy>0 and self.iy <480:
            self.ix,self.iy = self.calP(wx,wy,wz)
        return self.ix,self.iy

    def h(self, x):
        wx,wy,wz,q1,q2,q3,q4 = x[0:7]
        hx,hy,H_JacoX,H_JacoY = self.calR(wx,wy,wz,q1,q2,q3,q4)
        self.h_func = np.vstack((hx,hy))
        self.h_func = np.array(self.h_func,dtype=float)
        #self.h_func= hf.reshape(hf.shape[0],1)
        self.H_Jacob = np.vstack((H_JacoX,H_JacoY))
        self.H_Jacob = np.array(self.H_Jacob,dtype=float)
        """计算 measurement function & Jacob"""
        return self.h_func, self.H_Jacob

    def f(self, x):
        """计算state translation function Jacob, 并返回计算结果"""
        self.F_func =np.eye(7)
        return np.dot(self.F_func, x), self.F_func  # 返回格式为 f,F

    def calR(self,wx_,wy_,wz_,q1_,q2_,q3_,q4_):
        pixel_x = 640
        pixel_y = 480
        span_x = 375  # 280mm
        span_y = 199  # 177mm
        D = 265  # Camera distance z = -475mm
        Projection_Matrix = Matrix([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 1 / D],
                                    [0, 0, -D, 1]])
        Pixel_Transform_Matrix = Matrix([[pixel_x / span_x, 0, 0, 0],
                                         [0, pixel_y / span_y, 0, 0],
                                         [0, 0, 1, 0],
                                         [pixel_x / 2, pixel_y / 2, 0, 1]])
        Project_Pixel = Projection_Matrix * Pixel_Transform_Matrix
        wx, wy, wz, q1, q2, q3, q4 = symbols('wx, wy, wz, q1, q2, q3, q4')
        Input_Matrix = Matrix([0, wx, wy, wz])
        qmodel = sqrt(q1 ** 2 + q2 ** 2 + q3 ** 2 + q4 ** 2)
        Rotation_Matrix = Matrix([[1, 0, 0, 0],
                                  [0, q1 ** 2 + q2 ** 2 - q3 ** 2 - q4 ** 2, 2 * q2 * q3 - 2 * q1 * q4,
                                   2 * q2 * q4 + 2 * q1 * q3],
                                  [0, 2 * q2 * q3 + 2 * q1 * q4, q1 ** 2 - q2 ** 2 + q3 ** 2 - q4 ** 2,
                                   2 * q3 * q4 - 2 * q1 * q2],
                                  [0, 2 * q2 * q4 - 2 * q1 * q3, 2 * q3 * q4 + 2 * q1 * q2,
                                   q1 ** 2 - q2 ** 2 - q3 ** 2 + q4 ** 2]])
        Output_Matrix = Rotation_Matrix * Input_Matrix / qmodel
        # wx_ch, wy_ch, wz_ch = Output_Matrix.row(1), Output_Matrix.row(2), Output_Matrix.row(3)
        # Rotated_World = Matrix([wx_ch, wy_ch, wz_ch, 1])
        Output_Matrix.row_del(0)
        Rotated_World = Output_Matrix.row_insert(3,Matrix([1]))
        Rotated_World = Rotated_World
        Rotated_Pixel = Project_Pixel.T*Rotated_World
        Rotated_Pixel_Homo = Rotated_Pixel / Rotated_Pixel.row(3)
        calx = Rotated_Pixel_Homo.row(0)
        caly = Rotated_Pixel_Homo.row(1)
        state = Matrix([wx, wy, wz, q1, q2, q3, q4])
        X_Jaco = calx.jacobian(state)
        Y_Jaco = caly.jacobian(state)

        calx = calx.subs(wx,wx_).subs(wy,wy_).subs(wz,wz_).subs(q1,q1_).subs(q2,q2_).subs(q3,q3_).subs(q4,q4_)
        caly = caly.subs(wx,wx_).subs(wy,wy_).subs(wz,wz_).subs(q1,q1_).subs(q2,q2_).subs(q3,q3_).subs(q4,q4_)
        X_Jaco = X_Jaco.subs(wx,wx_).subs(wy,wy_).subs(wz,wz_).subs(q1,q1_).subs(q2,q2_).subs(q3,q3_).subs(q4,q4_)
        Y_Jaco = Y_Jaco.subs(wx,wx_).subs(wy,wy_).subs(wz,wz_).subs(q1,q1_).subs(q2,q2_).subs(q3,q3_).subs(q4,q4_)

        calx = np.array(calx)
        caly = np.array(caly)
        X_Jaco = np.array(X_Jaco)
        Y_Jaco = np.array(Y_Jaco)
        return calx,caly,X_Jaco,Y_Jaco

    def calP(self,wx,wy,wz):
        pixel_x = 640
        pixel_y = 480
        span_x = 375  # 280mm
        span_y = 199  # 177mm
        D = 265  # Camera distance z = -475mm
        Projection_Matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1 / D], [0, 0, -D, 1]])
        Pixel_Transform_Matrix = np.array([[pixel_x / span_x, 0, 0, 0], [0, pixel_y / span_y, 0, 0], [0, 0, 1, 0], [pixel_x / 2, pixel_y / 2, 0, 1]])
        Whole_Matrix= np.dot(Projection_Matrix,Pixel_Transform_Matrix)
        Input = np.array([[wx,wy,wz,1]]).T
        Output = np.dot(Whole_Matrix.T,Input)
        Output = Output / Output[3,0]
        return Output[0,0],Output[1,0]






