# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:09:23 2017

@author: Harry
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import optimize

class garchModel:
    """Abstract class for garch volatility models"""
    
    __metaclass__ = ABCMeta
    
    def __init__(self, ret, useJIT=None):
        """
        Initializer for garch model
        
        param ret ndarray: vector of log-returns
        """
        self.T = ret.shape[0]
        if useJIT is None:
            useJIT = True if self.T >1e5 else False
        self.useJIT = useJIT
        self.ret = ret
        self.sigma2 = np.zeros(self.T)
        self.mu = None
        self.stdRet = ret - self.getMean()
        
    def getMean(self):
        """Gets the mean"""
        if self.mu is None:
            self.mu = self.ret.sum() / self.T
        return self.mu           

    def getBackcast(self,N=1):
        """return the backcast for first periods"""
        return self.ret[0:N].dot(self.ret[0:N])

    def loglikelihood(self,param):
        """Likelihood function"""
        sigma2 = self.volFilter(param)
        zt = self.stdRet / np.sqrt(sigma2)
        LL = sum( -0.5*(np.log(2*np.pi)+np.log(sigma2)+ zt**2 ))
        if np.isnan(LL):
            return -np.inf
        else:
            return LL
    
    def negloglikelihood(self,param):
        """returns negative likelihood"""
        return self.loglikelihood(param)
    
    @abstractmethod    
    def volFilter(self,param):
        """Method returns the volatility series"""
        raise Exception("Volatility filter should be implemented in leaf class")

    @abstractmethod    
    def getStartingVal(self):
        """Method returns starting points for estimation"""
        raise Exception("Starting val should be implemented in leaf class")

    def estimate(self):
        """Method runs ML estimation"""
        p0 = self.getStartingVal()
        param = optimize.fmin_bfgs(self.negloglikelihood, p0)
        return param

