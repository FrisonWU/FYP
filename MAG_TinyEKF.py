# -*- coding: UTF-8 -*-
'''
Extended Kalman Filter in Python
Copyright (C) 2016 Simon D. Levy
MIT License
'''

import numpy as np
import math
import sys
from abc import ABCMeta, abstractmethod
from filterpy.stats import logpdf
from filterpy.common import Q_discrete_white_noise


class EKF(object):
    '''
    A abstrat class for the Extended Kalman Filter, based on the tutorial in
    http://home.wlu.edu/~levys/kalman_tutorial.
    '''
    __metaclass__ = ABCMeta

    def __init__(self,n , m, rval, qval, pval, EnableAdaptive):
        '''
        Creates a KF object with n states, m observables, and specified values for
        prediction noise covariance pval, process noise covariance qval, and
        measurement noise covariance rval.
        '''

        # Enable adaptive Q & R
        self.EnableAdaptive = EnableAdaptive

        # No previous prediction noise covariance
        self.P_pre = None

        # Current state is zero, with diagonal noise covariance matrix
        self.x = np.ones(n)
        self.P_post = np.eye(n) * pval

        # Set up covariance matrices for process noise and measurement noise
        self.Q = np.eye(n) * qval
        self.R = np.eye(m) * rval
        self.S = np.zeros((m, m))  # system uncertainty
        self.SI = np.zeros((m, m))  # inverse system uncertainty
        self.y = np.zeros((m, 1))
        self.eps = 0  # residual

        # set to None to force recompute
        self._log_likelihood = math.log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.eps_pre = 0
        self.count = 0
        self.logFactor = 0
        # Identity matrix will be usefel later
        self.I = np.eye(n)

    def ekf_step(self, z):
        '''
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        '''

        # Predict ----------------------------------------------------

        # $\hat{x}_k = f(\hat{x}_{k-1})$
        """F为f的微分"""
        self.x, F = self.f(self.x)

        # $P_k = F_{k-1} P_{k-1} F^T_{k-1} + Q_{k-1}$
        self.P_pre = F * self.P_post * F.T + self.Q

        # Update -----------------------------------------------------
        """H为h的微分"""
        h, H = self.h(self.x)
        # $G_k = P_k H^T_k (H_k P_k H^T_k + R)^{-1}$
        G = np.dot(self.P_pre.dot(H.T), np.linalg.inv(H.dot(self.P_pre).dot(H.T) + self.R))

        if self.EnableAdaptive == 1:
            """measurement covariance calculation"""
            self.S = H.dot(self.P_pre).dot(H.T) + self.R
            self.SI = np.linalg.inv(self.S)
            self.y = (np.array(z) - h.T).T
            self.eps = np.dot(self.y.T, np.linalg.inv(self.S)).dot(self.y)

            self.Adaptive_Q(100/2, 10)

        # $\hat{x}_k = \hat{x_k} + G_k(z_k - h(\hat{x}_k))$

        zz=np.array(z)
        hh=h.T
        m = np.array(z)-h.T
        m = m.T
        n = np.dot(G, (np.array(z) - h.T).T)
        self.x += np.dot(G, (np.array(z) - h.T).T)

        # $P_k = (I - G_k H_k) P_k$
        self.P_post = np.dot(self.I - np.dot(G, H), self.P_pre)

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None
        self.eps_pre = self.eps

        # return self.x.asarray()
        return self.x

    def Adaptive_Q(self, Q_scale_factor, eps_max):
        if self.eps > eps_max:
            self.Q += Q_scale_factor
            #self.R += 5*abs(self.eps-self.eps_pre)
            #print abs(self.eps-self.eps_pre)
            self.count += 1
        elif self.count > 0:
            self.Q -= Q_scale_factor
            #self.R = 550*np.eye(27)
            self.count -= 1



    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """

        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = math.exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """
        Mahalanobis distance of innovation. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = math.sqrt(float(np.dot(np.dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis


    @abstractmethod
    def f(self, x):
        '''
        Your implementing class should define this method for the state-transition function f(x).
        Your state-transition fucntion should return a NumPy array of n elements representing the
        new state, and a nXn NumPy array of elements representing the the Jacobian of the function
        with respect to the new state.  Typically this is just the identity
        function np.copy(x), so the Jacobian is just np.eye(len(x)).  '''
        raise NotImplementedError()

    @abstractmethod
    def h(self, x):
        '''
        Your implementing class should define this method for the observation function h(x), returning
        a NumPy array of m elements, and a NumPy array of m x n elements representing the Jacobian matrix
        H of the observation function with respect to the observation. For
        example, your function might include a component that turns barometric
        pressure into altitude in meters.
        '''
        raise NotImplementedError()