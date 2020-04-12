from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from numpy import eye, array, asarray
import numpy as np
from sympy import*

class EKF:
    def __init__(self):
        self.wx = 100
        self.wy = 0
        self.wz = 0
        self.q1 = 0
        self.q2 = 1
        self.q3 = 0
        self.q4 = 0
        self.rk = ExtendedKalmanFilter(dim_x=7,dim_z=2)
    def start(self,zx,zy):
        #z = [zx],[zy]
        #rk = ExtendedKalmanFilter(dim_x=7,dim_z=2)
        #wx,wy,wz,q1,q2,q3,q4 = symbols('wx,wy,wz,q1,q2,q3,q4')
        #rk.x=Matrix[wx,wy,wz,q1,q2,q3,q4]
        self.rk.x = np.array([[self.wx,self.wy-10,self.wz,self.q1,self.q2+1,self.q3+1,self.q4+1]]).T
        #rk.F = np.eye(7)

        range_std = 10
        self.rk.R *= range_std
        self.rk.Q *= 0.1
        self.rk.P *= 50

        # wx,wy,wz,q1,q2,q3,q4 = rk.x[0:7]
        #h,H_Jacobian_at = self.calR(rk.x,self.wx,self.wy-10,self.wz+0,self.q1,self.q2,self.q3,self.q4)
        #hx = self.hx(rk.x)
        #H_Jacobian_atx=self.Jaco(rk.x)
        self.rk.predict_update(np.array([[zx],[zy]]),self.HJaco,self.Hs)
        fx = self.rk.x.flatten()
        wx_, wy_, wz_ = fx[0:3]
        ix,iy = self.calP(wx_,wy_,wz_)
        print('eps is ',self.rk.eps,"wx is", wx_,'wy is',wy_,'wz_ is',wz_)
        return ix,iy
    def calR(self):

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
        qmodel = q1 ** 2 + q2 ** 2 + q3 ** 2 + q4 **2
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
        # Rotated_Pixel = Project_Pixel * Rotated_World
        Output_Matrix.row_del(0)
        Rotated_World = Output_Matrix.row_insert(3, Matrix([1]))
        Rotated_Pixel = Project_Pixel.T*Rotated_World
        Rotated_Pixel_Homo = Rotated_Pixel / Rotated_Pixel.row(3)
        hx = Rotated_Pixel_Homo.row_del(3)
        hx = hx.row_del(2)
        variable = Matrix([wx, wy, wz, q1, q2, q3, q4])
        H_Jaco = hx.jacobian(variable)
        h = H_Jaco*variable
        return hx,H_Jaco

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


    def HJaco (self,state):
        h,hj = self.calR()
        wx,wy,wz,q1,q2,q3,q4 = symbols('wx,wy,wz,q1,q2,q3,q4')
        state =state.flatten()
        wx_,wy_, wz_, q1_, q2_, q3_, q4_ = state[0:7]
        Jaco = hj.subs(wx,wx_).subs(wy,wy_).subs(wz,wz_).subs(q1,q1_).subs(q2,q2_).subs(q3,q3_).subs(q4,q4_).evalf()
        Jaco = np.array(Jaco,dtype=float)
        return Jaco
    def Hs(self,state):
        h, hj = self.calR()
        wx, wy, wz, q1, q2, q3, q4 = symbols('wx,wy,wz,q1,q2,q3,q4')
        state = state.flatten()
        wx_, wy_, wz_, q1_, q2_, q3_, q4_ = state[0:7]
        hx = h.subs(wx,wx_).subs(wy,wy_).subs(wz,wz_).subs(q1,q1_).subs(q2,q2_).subs(q3,q3_).subs(q4,q4_).evalf()
        hx = np.array(hx,dtype=float)
        return hx



