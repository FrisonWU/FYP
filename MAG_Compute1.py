# -*- coding: UTF-8 -*-
import numpy as np
import math
import datetime
from MAG_tinyQuaternion import Quaternion
import multiprocessing
from threading import Thread
import MAG_TinyEKF as MAG_EKF
import MAG_SerialRead as MSR
import MAG_DataPlot as MAGplt
from filterpy.common import Q_discrete_white_noise
import pyqtgraph as pg
import array


class MagPredictor(MAG_EKF.EKF):
    """Read the data from MAG_SerialRead thread, and then process the data using EKF"""
    def __init__(self):
        """Parameters definition"""
        self.MAG_Num_of_Slaves = 9  # Num of slave boards
        self.MAG_Num_of_Gaussdata = 3  # Bxyz data num
        self.MAGdata = np.zeros(self.MAG_Num_of_Gaussdata * self.MAG_Num_of_Slaves)  # Observation from serial port
        self.n_state = 16  # (a,da, b, db c,dc, q0,dq0, q1,dq1 q2,dq2 q3,dq3 m)X2
        self.m_ob = self.MAG_Num_of_Gaussdata * self.MAG_Num_of_Slaves  # (Bx, By, Bz) X 9 slaves
        self.Default_rval = 550
        self.Default_qval = 0.5
        self.scale = 1e2
        self.MomentXYZ = np.zeros((3, 2))
        self.MagQ = np.zeros((4, 2))

        self.m = 2000  # moment unit : I*cm^2
        self.timecost = 0
        """Inherent the EKF class"""
        MAG_EKF.EKF.__init__(self, n=self.n_state, m=self.m_ob, rval=self.Default_rval, qval=self.Default_qval, pval=50, EnableAdaptive=1)
        self.x = np.array([0.1,0.1,0.1, 1, 0, 0, 0, self.m,
                           0.1,0.1,0.1, 1, 0, 0, 0, self.m])
        #self.Set_NoiseFunc()

        #传感器磁感应强度定义
        self.Bx = np.zeros(self.MAG_Num_of_Slaves)
        self.By = np.zeros(self.MAG_Num_of_Slaves)
        self.Bz = np.zeros(self.MAG_Num_of_Slaves)
        self.h_func = np.zeros(len(self.MAGdata))  # Measurement function
        self.H_Jacob = np.zeros((self.m_ob, self.n_state), np.float)  # Measurement function derivative
        self.F_func = np.zeros((self.n_state, self.n_state), np.float)

        """Sensor location definition"""
        self.Sensor_dis = 6  # sensor distance between each other
        self.Sensor_Loc = [(-self.Sensor_dis, self.Sensor_dis, 0), (0, self.Sensor_dis, 0), (self.Sensor_dis, self.Sensor_dis, 0),  # fist row
                           (-self.Sensor_dis, 0, 0), (0, 0, 0), (self.Sensor_dis, 0, 0),  # second row
                           (-self.Sensor_dis, -self.Sensor_dis, 0), (0, -self.Sensor_dis, 0), (self.Sensor_dis, -self.Sensor_dis, 0)]  # third row
        """Array for display"""
        self.EularAngle = np.zeros((3, 2))

    def start(self, q):
        """Magnetometer sensors fusion using EKF"""
        while 1:
            t1 = datetime.datetime.now()
            self.MAGdata = self.read_fromThread(q)  # read from serial thread
            self.MAGdata[0, :] = -self.MAGdata[0, :]
            self.MAGdata = self.MAGdata.reshape(-1)
            self.ekf_step(self.MAGdata)
            a, b, c, q0, q1, q2, q3, m, \
            a_,b_,c_,q0_,q1_,q2_,q3_,m_ = self.x[0:16]

            t2 = datetime.datetime.now()
            self.timecost = (t2 - t1).total_seconds()
            print "x:{:.3f}  y:{:.3f}  z:{:.3f}   x2:{:.3f}  y2:{:.3f}  z2:{:.3f}".format(a, b, c, a_, b_, c_), \
                 "  Moment: ", self.MomentXYZ[:,0],self.MomentXYZ[:,1],  " Time: ", (t2 - t1).total_seconds(), \
                "  eps:{:.3f}".format(self.eps)

           #self.EularAngle[:, 0] = self.getEularAngle(q0_, q1_, q2_, q3)

            # if t>50:
            #     t = 0
            # t+=1

    def calB(self, x, y, z, q0, q1, q2, q3, m, x_, y_, z_, q0_, q1_, q2_, q3_, m_, loc):
        """计算Bx, By, Bz"""
        # 第一组磁体数据
        a = x-loc[0]
        b = y-loc[1]
        c = z-loc[2]
        r = math.sqrt(a**2+b**2+c**2)
        Mx, My, Mz = 0, 0, 1
        mx = (Mx*(q0**2+q1**2-q2**2-q3**2) + 2*My*(q1*q2+q0*q3) + 2*Mz*(-q0*q2+q1*q3))/(q0**2 + q1**2 + q2**2 + q3**2)
        my = (2*Mx*(-q0*q3+q1*q2) + My*(q0**2-q1**2+q2**2-q3**2) + 2*Mz*(q0*q1+q2*q3))/(q0**2 + q1**2 + q2**2 + q3**2)
        mz = (2*Mx*(q0*q2+q1*q3) + 2*My*(-q0*q1+q2*q3) + Mz*(q0**2-q1**2-q2**2+q3**2))/(q0**2 + q1**2 + q2**2 + q3**2)
        self.MomentXYZ[0, 0], self.MomentXYZ[1, 0], self.MomentXYZ[2, 0] = mx, my, mz  # 记录当前Moment 3维XYZ

        # 第二组磁体数据
        a_ = x_-loc[0]
        b_ = y_-loc[1]
        c_ = z_-loc[2]
        r_ = math.sqrt(a_**2+b_**2+c_**2)
        mx_ = (Mx*(q0_**2+q1_**2-q2_**2-q3_**2) + 2*My*(q1_*q2_+q0_*q3_) + 2*Mz*(-q0_*q2_+q1_*q3_))/(q0_**2 + q1_**2 + q2_**2 + q3_**2)
        my_ = (2*Mx*(-q0_*q3_+q1_*q2_) + My*(q0_**2-q1_**2+q2_**2-q3_**2) + 2*Mz*(q0_*q1_+q2_*q3_))/(q0_**2 + q1_**2 + q2_**2 + q3_**2)
        mz_ = (2*Mx*(q0_*q2_+q1_*q3_) + 2*My*(-q0_*q1_+q2_*q3_) + Mz*(q0_**2-q1_**2-q2_**2+q3_**2))/(q0_**2 + q1_**2 + q2_**2 + q3_**2)
        self.MomentXYZ[0, 1], self.MomentXYZ[1, 1], self.MomentXYZ[2, 1] = mx_, my_, mz_  # 记录当前Moment 3维XYZ
        # 线性叠加
        return self.scale * m * pow(r, -3) * (3 * a * (mx * a + my * b + mz * c) / pow(r, 2) - mx) + self.scale * m_ * pow(r_, -3) * (3 * a_ * (mx_ * a_ + my_ * b_ + mz_ * c_) / pow(r_, 2) - mx_), \
               self.scale * m * pow(r, -3) * (3 * b * (mx * a + my * b + mz * c) / pow(r, 2) - my) + self.scale * m_ * pow(r_, -3) * (3 * b_ * (mx_ * a_ + my_ * b_ + mz_ * c_) / pow(r_, 2) - my_), \
               self.scale * m * pow(r, -3) * (3 * c * (mx * a + my * b + mz * c) / pow(r, 2) - mz) + self.scale * m_ * pow(r_, -3) * (3 * c_ * (mx_ * a_ + my_ * b_ + mz_ * c_) / pow(r_, 2) - mz_)

    def calBxyz_PD(self,  x, y, z, q0, q1, q2, q3, m, x_, y_, z_, q0_, q1_, q2_, q3_, m_, loc):
        """计算dh(x)/dstate 偏微分"""
        a = x - loc[0]
        b = y - loc[1]
        c = z - loc[2]
        a_ = x_-loc[0]
        b_ = y_-loc[1]
        c_ = z_-loc[2]
        mscale = self.scale
        dBxdstate = np.array([3*m*mscale*(a**2*(4*a*(q0*q2 - q1*q3) - 4*b*(q0*q1 + q2*q3) - 2*c*(q0**2 - q1**2 - q2**2 + q3**2)) - a*(3*a*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + 2*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2)) - (a**2 + b**2 + c**2)*(4*a*(q0*q2 - q1*q3) - 2*b*(q0*q1 + q2*q3) - c*(q0**2 - q1**2 - q2**2 + q3**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                             -3*m*mscale*(2*a*(b*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (q0*q1 + q2*q3)*(a**2 + b**2 + c**2)) + b*(3*a*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + 2*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                             -3*m*mscale*(a*(2*c*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2)) + c*(3*a*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + 2*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                             -2*m*mscale*(3*a*(q0*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (-a*q2 + b*q1 + c*q0)*(q0**2 + q1**2 + q2**2 + q3**2)) + 2*q0*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2) - q2*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                             -2*m*mscale*(3*a*(q1*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a*q3 + b*q0 - c*q1)*(q0**2 + q1**2 + q2**2 + q3**2)) + 2*q1*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2) + q3*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                             -2*m*mscale*(3*a*(q2*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a*q0 - b*q3 + c*q2)*(q0**2 + q1**2 + q2**2 + q3**2)) - q0*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2) + 2*q2*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                             -2*m*mscale*(3*a*(q3*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a*q1 + b*q2 + c*q3)*(q0**2 + q1**2 + q2**2 + q3**2)) + q1*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2) + 2*q3*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              mscale*(3*a*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + 2*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                              3*m_*mscale*(a_**2*(4*a_*(q0_*q2_ - q1_*q3_) - 4*b_*(q0_*q1_ + q2_*q3_) - 2*c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - a_*(3*a_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + 2*(q0_*q2_ - q1_*q3_)*(a_**2 + b_**2 + c_**2)) - (a_**2 + b_**2 + c_**2)*(4*a_*(q0_*q2_ - q1_*q3_) - 2*b_*(q0_*q1_ + q2_*q3_) - c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)))/((a_**2 + b_**2 + c_**2)**(7/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)),
                             -3*m_*mscale*(2*a_*(b_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - (q0_*q1_ + q2_*q3_)*(a_**2 + b_**2 + c_**2)) + b_*(3*a_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + 2*(q0_*q2_ - q1_*q3_)*(a_**2 + b_**2 + c_**2)))/((a_**2 + b_**2 + c_**2)**(7/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)),
                             -3*m_*mscale*(a_*(2*c_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - (a_**2 + b_**2 + c_**2)*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + c_*(3*a_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + 2*(q0_*q2_ - q1_*q3_)*(a_**2 + b_**2 + c_**2)))/((a_**2 + b_**2 + c_**2)**(7/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)),
                             -2*m_*mscale*(3*a_*(q0_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - (-a_*q2_ + b_*q1_ + c_*q0_)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)) + 2*q0_*(q0_*q2_ - q1_*q3_)*(a_**2 + b_**2 + c_**2) - q2_*(a_**2 + b_**2 + c_**2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)**2),
                             -2*m_*mscale*(3*a_*(q1_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - (a_*q3_ + b_*q0_ - c_*q1_)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)) + 2*q1_*(q0_*q2_ - q1_*q3_)*(a_**2 + b_**2 + c_**2) + q3_*(a_**2 + b_**2 + c_**2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)**2),
                             -2*m_*mscale*(3*a_*(q2_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + (a_*q0_ - b_*q3_ + c_*q2_)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)) - q0_*(a_**2 + b_**2 + c_**2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2) + 2*q2_*(q0_*q2_ - q1_*q3_)*(a_**2 + b_**2 + c_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)**2),
                             -2*m_*mscale*(3*a_*(q3_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - (a_*q1_ + b_*q2_ + c_*q3_)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)) + q1_*(a_**2 + b_**2 + c_**2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2) + 2*q3_*(q0_*q2_ - q1_*q3_)*(a_**2 + b_**2 + c_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)**2),
                              mscale*(3*a_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + 2*(q0_*q2_ - q1_*q3_)*(a_**2 + b_**2 + c_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2))])

        dBydstate = np.array([-3*m*mscale*(a*(3*b*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - 2*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2)) + 2*b*(a*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (q0*q2 - q1*q3)*(a**2 + b**2 + c**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                               3*m*mscale*(-2*b**2*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - b*(3*b*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - 2*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2)) + (a**2 + b**2 + c**2)*(-2*a*(q0*q2 - q1*q3) + 4*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                              -3*m*mscale*(b*(2*c*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2)) + c*(3*b*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - 2*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                              -2*m*mscale*(3*b*(q0*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (-a*q2 + b*q1 + c*q0)*(q0**2 + q1**2 + q2**2 + q3**2)) - 2*q0*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2) + q1*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              -2*m*mscale*(3*b*(q1*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a*q3 + b*q0 - c*q1)*(q0**2 + q1**2 + q2**2 + q3**2)) + q0*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2) - 2*q1*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              -2*m*mscale*(3*b*(q2*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a*q0 - b*q3 + c*q2)*(q0**2 + q1**2 + q2**2 + q3**2)) - 2*q2*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2) + q3*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                               2*m*mscale*(3*b*(q3*(2*a*(q0*q2 - q1*q3) - 2*b*(q0*q1 + q2*q3) - c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a*q1 + b*q2 + c*q3)*(q0**2 + q1**2 + q2**2 + q3**2)) - q2*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2) + 2*q3*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              mscale*(3*b*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - 2*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                              -3*m_*mscale*(a_*(3*b_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - 2*(q0_*q1_ + q2_*q3_)*(a_**2 + b_**2 + c_**2)) + 2*b_*(a_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + (q0_*q2_ - q1_*q3_)*(a_**2 + b_**2 + c_**2)))/((a_**2 + b_**2 + c_**2)**(7/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)),
                              3*m_*mscale*(-2*b_**2*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - b_*(3*b_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - 2*(q0_*q1_ + q2_*q3_)*(a_**2 + b_**2 + c_**2)) + (a_**2 + b_**2 + c_**2)*(-2*a_*(q0_*q2_ - q1_*q3_) + 4*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)))/((a_**2 + b_**2 + c_**2)**(7/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)),
                             -3*m_*mscale*(b_*(2*c_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - (a_**2 + b_**2 + c_**2)*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + c_*(3*b_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - 2*(q0_*q1_ + q2_*q3_)*(a_**2 + b_**2 + c_**2)))/((a_**2 + b_**2 + c_**2)**(7/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)),
                             -2*m_*mscale*(3*b_*(q0_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - (-a_*q2_ + b_*q1_ + c_*q0_)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)) - 2*q0_*(q0_*q1_ + q2_*q3_)*(a_**2 + b_**2 + c_**2) + q1_*(a_**2 + b_**2 + c_**2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)**2),
                             -2*m_*mscale*(3*b_*(q1_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - (a_*q3_ + b_*q0_ - c_*q1_)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)) + q0_*(a_**2 + b_**2 + c_**2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2) - 2*q1_*(q0_*q1_ + q2_*q3_)*(a_**2 + b_**2 + c_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)**2),
                             -2*m_*mscale*(3*b_*(q2_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + (a_*q0_ - b_*q3_ + c_*q2_)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)) - 2*q2_*(q0_*q1_ + q2_*q3_)*(a_**2 + b_**2 + c_**2) + q3_*(a_**2 + b_**2 + c_**2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)**2),
                              2*m_*mscale*(3*b_*(q3_*(2*a_*(q0_*q2_ - q1_*q3_) - 2*b_*(q0_*q1_ + q2_*q3_) - c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + (a_*q1_ + b_*q2_ + c_*q3_)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)) - q2_*(a_**2 + b_**2 + c_**2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2) + 2*q3_*(q0_*q1_ + q2_*q3_)*(a_**2 + b_**2 + c_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)**2),
                              mscale*(3*b_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - 2*(q0_*q1_ + q2_*q3_)*(a_**2 + b_**2 + c_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2))])

        dBzdstate = np.array([-3*m*mscale*(a*(3*c*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2)) + 2*c*(a*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (q0*q2 - q1*q3)*(a**2 + b**2 + c**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                              -3*m*mscale*(b*(3*c*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2)) + 2*c*(b*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (q0*q1 + q2*q3)*(a**2 + b**2 + c**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                              3*m*mscale*(-2*c**2*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - c*(3*c*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a**2 + b**2 + c**2)*(-q0**2 + q1**2 + q2**2 - q3**2)) + 2*(a**2 + b**2 + c**2)*(-a*(q0*q2 - q1*q3) + b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                              -2*m*mscale*(3*c*(q0*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (-a*q2 + b*q1 + c*q0)*(q0**2 + q1**2 + q2**2 + q3**2)) - q0*(a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2) + q0*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              2*m*mscale*(-3*c*(q1*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (-a*q3 - b*q0 + c*q1)*(q0**2 + q1**2 + q2**2 + q3**2)) + q1*(a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2) + q1*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              2*m*mscale*(-3*c*(q2*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a*q0 - b*q3 + c*q2)*(q0**2 + q1**2 + q2**2 + q3**2)) + q2*(a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2) + q2*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              2*m*mscale*(3*c*(q3*(2*a*(q0*q2 - q1*q3) - 2*b*(q0*q1 + q2*q3) - c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a*q1 + b*q2 + c*q3)*(q0**2 + q1**2 + q2**2 + q3**2)) + q3*(a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2) - q3*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              mscale*(3*c*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a**2 + b**2 + c**2)*(-q0**2 + q1**2 + q2**2 - q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                              -3*m_*mscale*(a_*(3*c_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - (a_**2 + b_**2 + c_**2)*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + 2*c_*(a_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + (q0_*q2_ - q1_*q3_)*(a_**2 + b_**2 + c_**2)))/((a_**2 + b_**2 + c_**2)**(7/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)),
                              -3*m_*mscale*(b_*(3*c_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - (a_**2 + b_**2 + c_**2)*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + 2*c_*(b_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - (q0_*q1_ + q2_*q3_)*(a_**2 + b_**2 + c_**2)))/((a_**2 + b_**2 + c_**2)**(7/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)),
                              3*m_*mscale*(-2*c_**2*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - c_*(3*c_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + (a_**2 + b_**2 + c_**2)*(-q0_**2 + q1_**2 + q2_**2 - q3_**2)) + 2*(a_**2 + b_**2 + c_**2)*(-a_*(q0_*q2_ - q1_*q3_) + b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)))/((a_**2 + b_**2 + c_**2)**(7/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)),
                              -2*m_*mscale*(3*c_*(q0_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) - (-a_*q2_ + b_*q1_ + c_*q0_)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)) - q0_*(a_**2 + b_**2 + c_**2)*(q0_**2 - q1_**2 - q2_**2 + q3_**2) + q0_*(a_**2 + b_**2 + c_**2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)**2),
                              2*m_*mscale*(-3*c_*(q1_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + (-a_*q3_ - b_*q0_ + c_*q1_)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)) + q1_*(a_**2 + b_**2 + c_**2)*(q0_**2 - q1_**2 - q2_**2 + q3_**2) + q1_*(a_**2 + b_**2 + c_**2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)**2),
                              2*m_*mscale*(-3*c_*(q2_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + (a_*q0_ - b_*q3_ + c_*q2_)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)) + q2_*(a_**2 + b_**2 + c_**2)*(q0_**2 - q1_**2 - q2_**2 + q3_**2) + q2_*(a_**2 + b_**2 + c_**2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)**2),
                              2*m_*mscale*(3*c_*(q3_*(2*a_*(q0_*q2_ - q1_*q3_) - 2*b_*(q0_*q1_ + q2_*q3_) - c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + (a_*q1_ + b_*q2_ + c_*q3_)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)) + q3_*(a_**2 + b_**2 + c_**2)*(q0_**2 - q1_**2 - q2_**2 + q3_**2) - q3_*(a_**2 + b_**2 + c_**2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2)**2),
                              mscale*(3*c_*(-2*a_*(q0_*q2_ - q1_*q3_) + 2*b_*(q0_*q1_ + q2_*q3_) + c_*(q0_**2 - q1_**2 - q2_**2 + q3_**2)) + (a_**2 + b_**2 + c_**2)*(-q0_**2 + q1_**2 + q2_**2 - q3_**2))/((a_**2 + b_**2 + c_**2)**(5/2)*(q0_**2 + q1_**2 + q2_**2 + q3_**2))])

        return dBxdstate, dBydstate, dBzdstate

    def h(self, x):
        """计算 measurement function & Jacob"""
        a, b, c, q0, q1, q2, q3, m, \
        a_, b_, c_, q0_, q1_, q2_, q3_, m_ = x[0:16]
        self.MagQ[0, 0], self.MagQ[1, 0], self.MagQ[2, 0], self.MagQ[3, 0],\
        self.MagQ[0, 1], self.MagQ[1, 1], self.MagQ[2, 1], self.MagQ[3, 1] = q0, q1, q2, q3, q0_, q1_, q2_, q3_
        for i in range(9):
            """Element calculation"""
            # obtain the estimated B
            self.Bx[i], self.By[i], self.Bz[i] = self.calB(a, b, c, q0, q1, q2, q3, m, a_, b_, c_, q0_, q1_, q2_, q3_, m_, self.Sensor_Loc[i])

            """计算h的jacob matrix H"""
            self.H_Jacob[i], self.H_Jacob[i + self.MAG_Num_of_Slaves], self.H_Jacob[i + 2 * self.MAG_Num_of_Slaves] = self.calBxyz_PD(a, b, c, q0, q1, q2, q3, m, a_, b_, c_, q0_, q1_, q2_, q3_, m_, self.Sensor_Loc[i])

            # print self.H_Jacob
        """计算h(x)"""
        self.h_func = np.   ((self.Bx, self.By, self.Bz), axis=0)  # measurement function create

        return self.h_func, self.H_Jacob  # 返回格式为:h,H

    def f(self, x):
        """计算state translation function Jacob, 并返回计算结果"""
        self.F_func = np.eye(len(x))
        return np.dot(self.F_func, x), self.F_func  # 返回格式为 f,F

    def Set_NoiseFunc(self):
        """Reset process noise"""
        """第一组磁体"""
        for i in range(0, 3):
            # (x, vx, y, vy, z, vz) states
            self.Q[i*2:i*2+2, i*2:i*2+2] = Q_discrete_white_noise(2, dt=0.3, var=0.1)
            if i == 2:
                self.Q[i * 2:i * 2 + 2, i * 2:i * 2 + 2] = Q_discrete_white_noise(2, dt=0.3, var=0.2)
            self.P_post[i * 2:i * 2 + 2, i * 2:i * 2 + 2] = 100
        for i in range(3, 7):
            # quaternions and v_Quaternions states
            self.Q[i*2:i*2+2, i*2:i*2+2] = Q_discrete_white_noise(2, dt=0.3, var=1)
            self.P_post[i*2:i*2+2, i*2:i*2+2] = 50
        self.Q[14, 14] = 0.0
        # self.P_post[14, 14] =0.0
        """第二组磁体"""
        for i in range(7, 10):
            # (x, vx, y, vy, z, vz) states
            self.Q[i*2:i*2+2, i*2:i*2+2] = Q_discrete_white_noise(2, dt=0.3, var=0.1)
            if i == 9:
                self.Q[i * 2:i * 2 + 2, i * 2:i * 2 + 2] = Q_discrete_white_noise(2, dt=0.3, var=0.2)
            self.P_post[i * 2:i * 2 + 2, i * 2:i * 2 + 2] = 100
        for i in range(10, 14):
            # quaternions and v_Quaternions states
            self.Q[i*2:i*2+2, i*2:i*2+2] = Q_discrete_white_noise(2, dt=0.3, var=1)
            self.P_post[i*2:i*2+2, i*2:i*2+2] = 50
        self.Q[28, 28] = 0.0

    def read_fromThread(self, q):
        """从Serial Read线程中读取数据"""
        while 1:
            data = q.get()
            '''直到序列为空'''
            if q.empty():
                break
        return data

    def getMoment_PolarAngle(self, q0, q1, q2, q3, realize=False):
        """Get polar coordinate angle theta, phi"""
        qt = Quaternion(q=np.array([q0, q1, q2, q3]))
        qt.normalized()
        Point = qt.rotatePoint(np.array([0, 0, 1]))
        theta = np.degrees(np.arccos(Point[2]))  # 先得到theta角
        if theta == 0:
            phi = 0
        else:
            if Point[1] < 0:  # 如果在xy三四象限
                phi = -np.degrees(np.arccos(Point[0]/math.sqrt(Point[0]**2+Point[1]**2)))
            else:  # 如果在xy一二象限
                phi = np.degrees(np.arccos(Point[0]/math.sqrt(Point[0]**2+Point[1]**2)))
        return theta, phi

    def getEularAngle(self, q0, q1, q2, q3):
        qt = Quaternion(q=np.array([q0, q1, q2, q3]))
        qt.normalized()
        yaw = np.arctan2(2*(qt.q[1]*qt.q[2]+qt.q[0]*qt.q[3]), (1-2*(qt.q[2]**2+qt.q[3]**2)))
        pitch = np.arcsin(2*(qt.q[0]*qt.q[2]-qt.q[1]*qt.q[3]))
        raw = np.arctan2(2*(qt.q[2]*qt.q[3]+qt.q[0]*qt.q[1]), (1-2*(qt.q[1]**2+qt.q[2]**2)))
        return np.array([raw, pitch, yaw])

    def getPredictErr(self):
        """Calculate the prediction error"""
        rawdata = self.MAGdata.reshape(self.MAG_Num_of_Gaussdata, self.MAG_Num_of_Slaves)
        err = np.zeros((self.MAG_Num_of_Gaussdata, self.MAG_Num_of_Slaves), np.int)
        r = np.zeros((self.MAG_Num_of_Gaussdata, self.MAG_Num_of_Slaves), np.int)
        for i in range(self.MAG_Num_of_Slaves):
            if abs(rawdata[0, i]) > 10:
                r[0, i] = abs(rawdata[0, i] - self.Bx[i])
                err[0, i] = 100 * r[0, i] / abs(rawdata[0, i])
            else:
                err[0, i] = 0
            if abs(rawdata[1, i]) > 10:
                r[1, i] = abs(rawdata[1, i] - self.By[i])
                err[1, i] = 100 * abs(rawdata[1, i] - self.By[i])/abs(rawdata[1, i])
            else:
                err[1, i] = 0

            if abs(rawdata[2, i]) > 10:
                r[2, i] = abs(rawdata[2, i] - self.Bz[i])
                err[2, i] = 100 * abs(rawdata[2, i] - self.Bz[i])/abs(rawdata[2, i])
            else:
                err[2, i] = 0
        return err[2,:]


if __name__ == '__main__':
    MAG_Serial_Ins = MSR.MAGSerial()
    myMAG_Pre = MagPredictor()
    myMAG_plt = MAGplt.Plotter()

    Thread_list = []
    """Create Queue for thread data transfer"""
    q = multiprocessing.Queue()
    p = multiprocessing.Queue()
    """Create threads"""
    Thread_MAGDataRead = Thread(target=MAG_Serial_Ins.data_read, args=(q,))
    #Thread_MAGDataProcess = Thread(target=myMAG_Pre.test, args=("func", q,))
    Thread_MAGDataProcess = Thread(target=myMAG_Pre.start, args=(q,))
    Thread_MAGDataPlot = Thread(target=myMAG_plt.dataplt_ion, args=(p,))
    """Add the thread to thread list"""
    Thread_list.append(Thread_MAGDataRead)
    Thread_list.append(Thread_MAGDataProcess)
    Thread_list.append(Thread_MAGDataPlot)
    """Start threads"""
    for t in Thread_list:
        t.setDaemon(True)  # 守护线程, 当主线程结束后子线程将立即结束, 不管是否运行完成
        t.start()
    """Join the threads (主线程将在子线程结束后再结束)"""
    for t in Thread_list:
        t.join()



