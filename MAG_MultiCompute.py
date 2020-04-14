# -*- coding: UTF-8 -*-
import numpy as np
import math
import datetime
import time
from MAG_tinyQuaternion import Quaternion
import multiprocessing
from multiprocessing import Process
from threading import Thread
import MAG_TinyEKF as MAG_EKF
import MAG_SerialRead as MSR
import MAG_DataPlot as MAGplt
from filterpy.common import Q_discrete_white_noise
import pyqtgraph as pg
import array


class MultiMagDetector:
    """To determine how many magnets existed and then do predictor"""
    def __init__(self):
        """先创建一个最多能检测10个mag的predictor"""
        self.myPredictor = []
        self.disList = []
        self.isStop = 0
        for i in range(10):
            Predictor = MagPredictor(i+1, 1)
            self.myPredictor.append(Predictor)

    def start(self, q):
        for i in range(10):
            self.dis = np.zeros(i+1)
            for j in range(200):
                self.myPredictor[i].predict(q)
            for k in range(i+1):
                self.dis[k] = (self.myPredictor[i].MagPosXYZ[0, k]**2 + self.myPredictor[i].MagPosXYZ[1, k]**2 + self.myPredictor[i].MagPosXYZ[2, k]**2)**0.5
                print (self.dis)
                if self.dis[k] > 20 or self.dis[k] < 3 or self.myPredictor[i].MagPosXYZ[2, k] < -1:
                    self.dis[k] = 0
                    self.isStop = 1
                    break
            self.disList.append(self.dis)
            if self.isStop == 1:
                break
        #     # print str(i+1) + "test"
        #     # print self.myPredictor[i]
        print self.disList

    def read_fromThread(self, q):
        """从Serial Read线程中读取数据"""
        while 1:
            data = q.get()
            '''直到序列为空'''
            if q.empty():
                break
        return data


class MagPredictor(MAG_EKF.EKF):
    """Read the data from MAG_SerialRead thread, and then process the data using EKF"""
    def __init__(self, nMAG, EnableAdaptive):
        """Parameters definition"""
        self.MAG_Num_of_Slaves = 9  # Num of slave boards
        self.MAG_Num_of_Gaussdata = 3  # Bxyz data num
        self.MAGdata = np.zeros(self.MAG_Num_of_Gaussdata * self.MAG_Num_of_Slaves)  # Observation from serial port
        self.n_state = 8  # (a,da, b, db c,dc, q0,dq0, q1,dq1 q2,dq2 q3,dq3 m)X2
        self.MagMaxNum = 10  # 最多可检测磁体数量
        self.MagNum = nMAG  # 磁体的个数
        self.m_ob = self.MAG_Num_of_Gaussdata * self.MAG_Num_of_Slaves  # (Bx, By, Bz) X 9 slaves

        if nMAG == 3:
            self.Default_rval = 1000
            self.Default_qval = 0.6
        elif nMAG==2:
            self.Default_rval = 550
            self.Default_qval = 0.5
        else:
            self.Default_rval = 500
            self.Default_qval = 0.5

        self.scale = 1e2
        self.MagPosXYZ = np.zeros((3, self.MagNum))
        self.m = 2000  # moment unit : I*cm^2
        self.timecost = 0
        """Inherent the EKF class"""
        MAG_EKF.EKF.__init__(self, n=self.n_state*self.MagNum, m=self.m_ob, rval=self.Default_rval, qval=self.Default_qval, pval=50, EnableAdaptive=EnableAdaptive)
        self.x = np.array([6,6,6, 1, 0, 0, 0, self.m])
        #self.Q[7, 7] = 50.0
        for i in range(self.MagNum - 1):
            self.x = np.hstack((self.x, np.array([6,6,6, 1, 0, 0, 0, self.m])))
            #self.Q[7+7*i, 7+7*i] = 50.0

        #传感器磁感应强度定义
        self.Bx = np.zeros(self.MAG_Num_of_Slaves)
        self.By = np.zeros(self.MAG_Num_of_Slaves)
        self.Bz = np.zeros(self.MAG_Num_of_Slaves)
        self.h_func = np.zeros(len(self.MAGdata))  # Measurement function
        self.H_Jacob = np.zeros((self.m_ob, self.n_state*self.MagNum), np.float)  # Measurement function derivative
        self.F_func = np.zeros((self.n_state*self.MagNum, self.n_state*self.MagNum), np.float)

        """Sensor location definition"""
        self.Sensor_dis = 6  # sensor distance between each other
        self.Sensor_Loc = [(-self.Sensor_dis, self.Sensor_dis, 0), (0, self.Sensor_dis, 0), (self.Sensor_dis, self.Sensor_dis, 0),  # fist row
                           (-self.Sensor_dis, 0, 0), (0, 0, 0), (self.Sensor_dis, 0, 0),  # second row
                           (-self.Sensor_dis, -self.Sensor_dis, 0), (0, -self.Sensor_dis, 0), (self.Sensor_dis, -self.Sensor_dis, 0)]  # third row
        """Array for display"""
        self.EularAngle = np.zeros((3, self.MagNum))
        """str to print"""
        self.str = []

    def start(self, q, p):
        """Magnetometer sensors fusion using EKF"""
        while 1:
            self.predict(q, p)

    def predict(self, q, p):
        """Do predict for one time"""
        self.str = []
        t1 = datetime.datetime.now()
        self.MAGdata = self.read_fromThread(q)  # read from serial thread
        self.MAGdata[0, :] = -self.MAGdata[0, :]
        self.MAGdata = self.MAGdata.reshape(-1)
        self.ekf_step(self.MAGdata)
        t2 = datetime.datetime.now()
        self.timecost = (t2 - t1).total_seconds()
        p[:-1] = self.x[:]
        p[-1] = self.timecost
        for i in range(self.MagNum):
            self.MagPosXYZ[0, i], self.MagPosXYZ[1, i], self.MagPosXYZ[2, i] = self.x[8 * i:8 * i + 3]
            self.str.append("  |x{:d}:{:.3f}  y{:d}:{:.3f}  z{:d}:{:.3f} m{:d}:{:.3f}".format(i + 1, self.MagPosXYZ[0, i], i + 1,
                                                                        self.MagPosXYZ[1, i], i + 1,
                                                                        self.MagPosXYZ[2, i], i+1, self.x[i*8+7]))
        print self.str, " Time: ", (t2 - t1).total_seconds(), "  eps:{:.3f}".format(self.eps)

    def calB(self, state, loc, n):
        """计算Bx, By, Bz"""
        Bx, By, Bz = 0, 0, 0
        for i in range(n):
            x, y, z, q0, q1, q2, q3, m = state[i*8:i*8+8]
            a = x-loc[0]
            b = y-loc[1]
            c = z-loc[2]
            r = math.sqrt(a**2+b**2+c**2)
            Mx, My, Mz = 0, 0, 1
            mx = (Mx*(q0**2+q1**2-q2**2-q3**2) + 2*My*(q1*q2+q0*q3) + 2*Mz*(-q0*q2+q1*q3))/(q0**2 + q1**2 + q2**2 + q3**2)
            my = (2*Mx*(-q0*q3+q1*q2) + My*(q0**2-q1**2+q2**2-q3**2) + 2*Mz*(q0*q1+q2*q3))/(q0**2 + q1**2 + q2**2 + q3**2)
            mz = (2*Mx*(q0*q2+q1*q3) + 2*My*(-q0*q1+q2*q3) + Mz*(q0**2-q1**2-q2**2+q3**2))/(q0**2 + q1**2 + q2**2 + q3**2)
            # 线性叠加
            Bx += self.scale * m * pow(r, -3) * (3 * a * (mx * a + my * b + mz * c) / pow(r, 2) - mx)
            By += self.scale * m * pow(r, -3) * (3 * b * (mx * a + my * b + mz * c) / pow(r, 2) - my)
            Bz += self.scale * m * pow(r, -3) * (3 * c * (mx * a + my * b + mz * c) / pow(r, 2) - mz)
        return Bx, By, Bz

    def calBxyz_PD(self, state, loc, n):
        """计算dh(x)/dstate 偏微分"""
        dBxdstate, dBydstate, dBzdstate = np.zeros(8*n), np.zeros(8*n), np.zeros(8*n)

        for i in range(n):
            x, y, z, q0, q1, q2, q3, m = state[i * 8:i * 8 + 8]
            a = x - loc[0]
            b = y - loc[1]
            c = z - loc[2]
            mscale = self.scale
            BxJacob = np.array([3*m*mscale*(a**2*(4*a*(q0*q2 - q1*q3) - 4*b*(q0*q1 + q2*q3) - 2*c*(q0**2 - q1**2 - q2**2 + q3**2)) - a*(3*a*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + 2*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2)) - (a**2 + b**2 + c**2)*(4*a*(q0*q2 - q1*q3) - 2*b*(q0*q1 + q2*q3) - c*(q0**2 - q1**2 - q2**2 + q3**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                                 -3*m*mscale*(2*a*(b*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (q0*q1 + q2*q3)*(a**2 + b**2 + c**2)) + b*(3*a*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + 2*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                                 -3*m*mscale*(a*(2*c*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2)) + c*(3*a*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + 2*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                                 -2*m*mscale*(3*a*(q0*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (-a*q2 + b*q1 + c*q0)*(q0**2 + q1**2 + q2**2 + q3**2)) + 2*q0*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2) - q2*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                                 -2*m*mscale*(3*a*(q1*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a*q3 + b*q0 - c*q1)*(q0**2 + q1**2 + q2**2 + q3**2)) + 2*q1*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2) + q3*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                                 -2*m*mscale*(3*a*(q2*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a*q0 - b*q3 + c*q2)*(q0**2 + q1**2 + q2**2 + q3**2)) - q0*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2) + 2*q2*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                                 -2*m*mscale*(3*a*(q3*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a*q1 + b*q2 + c*q3)*(q0**2 + q1**2 + q2**2 + q3**2)) + q1*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2) + 2*q3*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                                  mscale*(3*a*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + 2*(q0*q2 - q1*q3)*(a**2 + b**2 + c**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2))])
            ByJacob = np.array([-3*m*mscale*(a*(3*b*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - 2*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2)) + 2*b*(a*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (q0*q2 - q1*q3)*(a**2 + b**2 + c**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                               3*m*mscale*(-2*b**2*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - b*(3*b*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - 2*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2)) + (a**2 + b**2 + c**2)*(-2*a*(q0*q2 - q1*q3) + 4*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                              -3*m*mscale*(b*(2*c*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2)) + c*(3*b*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - 2*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                              -2*m*mscale*(3*b*(q0*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (-a*q2 + b*q1 + c*q0)*(q0**2 + q1**2 + q2**2 + q3**2)) - 2*q0*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2) + q1*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              -2*m*mscale*(3*b*(q1*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a*q3 + b*q0 - c*q1)*(q0**2 + q1**2 + q2**2 + q3**2)) + q0*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2) - 2*q1*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              -2*m*mscale*(3*b*(q2*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a*q0 - b*q3 + c*q2)*(q0**2 + q1**2 + q2**2 + q3**2)) - 2*q2*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2) + q3*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                               2*m*mscale*(3*b*(q3*(2*a*(q0*q2 - q1*q3) - 2*b*(q0*q1 + q2*q3) - c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a*q1 + b*q2 + c*q3)*(q0**2 + q1**2 + q2**2 + q3**2)) - q2*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2) + 2*q3*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              mscale*(3*b*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - 2*(q0*q1 + q2*q3)*(a**2 + b**2 + c**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2))])
            BzJacob = np.array([-3*m*mscale*(a*(3*c*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2)) + 2*c*(a*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (q0*q2 - q1*q3)*(a**2 + b**2 + c**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                              -3*m*mscale*(b*(3*c*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2)) + 2*c*(b*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (q0*q1 + q2*q3)*(a**2 + b**2 + c**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                              3*m*mscale*(-2*c**2*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - c*(3*c*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a**2 + b**2 + c**2)*(-q0**2 + q1**2 + q2**2 - q3**2)) + 2*(a**2 + b**2 + c**2)*(-a*(q0*q2 - q1*q3) + b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)))/((a**2 + b**2 + c**2)**(7/2)*(q0**2 + q1**2 + q2**2 + q3**2)),
                              -2*m*mscale*(3*c*(q0*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) - (-a*q2 + b*q1 + c*q0)*(q0**2 + q1**2 + q2**2 + q3**2)) - q0*(a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2) + q0*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              2*m*mscale*(-3*c*(q1*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (-a*q3 - b*q0 + c*q1)*(q0**2 + q1**2 + q2**2 + q3**2)) + q1*(a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2) + q1*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              2*m*mscale*(-3*c*(q2*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a*q0 - b*q3 + c*q2)*(q0**2 + q1**2 + q2**2 + q3**2)) + q2*(a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2) + q2*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              2*m*mscale*(3*c*(q3*(2*a*(q0*q2 - q1*q3) - 2*b*(q0*q1 + q2*q3) - c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a*q1 + b*q2 + c*q3)*(q0**2 + q1**2 + q2**2 + q3**2)) + q3*(a**2 + b**2 + c**2)*(q0**2 - q1**2 - q2**2 + q3**2) - q3*(a**2 + b**2 + c**2)*(q0**2 + q1**2 + q2**2 + q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2)**2),
                              mscale*(3*c*(-2*a*(q0*q2 - q1*q3) + 2*b*(q0*q1 + q2*q3) + c*(q0**2 - q1**2 - q2**2 + q3**2)) + (a**2 + b**2 + c**2)*(-q0**2 + q1**2 + q2**2 - q3**2))/((a**2 + b**2 + c**2)**(5/2)*(q0**2 + q1**2 + q2**2 + q3**2))])
            if i == 0:
                dBxdstate = BxJacob
                dBydstate = ByJacob
                dBzdstate = BzJacob
            else:
                dBxdstate = np.hstack((dBxdstate, BxJacob))
                dBydstate = np.hstack((dBydstate, ByJacob))
                dBzdstate = np.hstack((dBzdstate, BzJacob))

        return dBxdstate, dBydstate, dBzdstate

    def h(self, x):
        """计算 measurement function & Jacob"""
        for i in range(9):
            """Element calculation"""
            # obtain the estimated B
            self.Bx[i], self.By[i], self.Bz[i] = self.calB(x, self.Sensor_Loc[i], self.MagNum)
            """计算h的jacob matrix H"""
            self.H_Jacob[i], self.H_Jacob[i + self.MAG_Num_of_Slaves], self.H_Jacob[i + 2 * self.MAG_Num_of_Slaves] = self.calBxyz_PD(x, self.Sensor_Loc[i], self.MagNum)
            # print self.H_Jacob
        """计算h(x)"""
        self.h_func = np.concatenate((self.Bx, self.By, self.Bz), axis=0)  # measurement function create
        return self.h_func, self.H_Jacob  # 返回格式为:h,H

    def f(self, x):
        """计算state translation function Jacob, 并返回计算结果"""
        self.F_func = np.eye(len(x))
        return np.dot(self.F_func, x), self.F_func  # 返回格式为 f,F

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
        x = (2 * q1 * q3 - 2 * q0 * q2) / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        y = (2 * q2 * q3 + 2 * q0 * q1) / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        z = (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        theta = np.degrees(np.arccos(z))  # 先得到theta角
        if theta == 0:
            phi = 0
        else:
            if y < 0:  # 如果在xy三四象限
                phi = -np.degrees(np.arccos(x/math.sqrt(x**2+y**2)))
            else:  # 如果在xy一二象限
                phi = np.degrees(np.arccos(x/math.sqrt(x**2+y**2)))
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

    def __str__(self):
        return str(self.str)


if __name__ == '__main__':
    MAG_Serial_Ins = MSR.MAGSerial()
    myMAG_Pre = MagPredictor(2, 1)
    myMAG_plt = MAGplt.Plotter()
    myDectector = MultiMagDetector()
    Thread_list = []
    """Create Queue for thread data transfer"""
    q = multiprocessing.Queue()
    # p = multiprocessing.Queue()
    # """Create threads"""
    # Thread_MAGDataRead = Thread(target=MAG_Serial_Ins.data_read, args=(q,))
    # #Thread_MAGDataProcess = Thread(target=myMAG_Pre.test, args=("func", q,))
    # Thread_MAGDataProcess = Thread(target=myMAG_Pre.start, args=(q,))
    # #Thread_MAGDataProcess = Thread(target=myDectector.start, args=(q,))
    # #Thread_MAGDataPlot = Thread(target=myMAG_plt.dataplt_ion, args=(p,))
    # """Add the thread to thread list"""
    # Thread_list.append(Thread_MAGDataRead)
    # Thread_list.append(Thread_MAGDataProcess)
    # #Thread_list.append(Thread_MAGDataPlot)
    # """Start threads"""
    # for t in Thread_list:
    #     t.setDaemon(True)  # 守护线程, 当主线程结束后子线程将立即结束, 不管是否运行完成
    #     t.start()
    # """Join the threads (主线程将在子线程结束后再结束)"""
    # for t in Thread_list:
    #     t.join()
    """多进程"""
    Process_list = []
    Process_MAGDataRead = Process(target=MAG_Serial_Ins.data_read, args=(q,))
    Process_MAGDataProcess = Process(target=myMAG_Pre.start, args=(q,))
    Process_list.append(Process_MAGDataRead)
    Process_list.append(Process_MAGDataProcess)
    for p in Process_list:
        p.daemon = True
        p.start()
    for p in Process_list:
        p.join()

