# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 22:39:00 2017

@author: Harry
"""
from volatility_model import garchModel
from numba import jit
import numpy as np


class garchpq(garchModel):
    """Class for garch(p,q)"""
    
    def __init__(self, ret, p, q):
        super(garchpq, self).__init__(ret)
        self._p = p
        self._q = q
    
    def volFilter(self, param):
        """Method returns the volatility series"""
        if self.useJIT:
            return jit('double[:](double[:],double[:],double[:],int32,int32,double)')(garch_pq_filter)(
                                                                                param,
                                                                                self.stdRet,
                                                                                self.sigma2,
                                                                                self._p,
                                                                                self._q,
                                                                                self.getBackcast())
        else:
            return garch_pq_filter(param,self.stdRet,self.sigma2,self._p,self._q,self.getBackcast())


    def getStartingVal(self):
        """Method returns starting points for estimation"""
        return 0.5 * np.random.rand(1+self._p+self._q)/(1+self._p+self._q)

    
def garch_pq_filter(param, ret, sigma2, p, q, backcast):
    T = np.size(ret, 0)
    for i in xrange(T):
        sigma2[i] = param[0]
        for j in xrange(1, p + 1):
            if (i - j) < 0:
                sigma2[i] += param[j] * backcast
            else:
                sigma2[i] += param[j] * (ret[i - j] * ret[i - j])
        for j in xrange(1, q + 1):
            if (i - j) < 0:
                sigma2[i] += param[p + j] * backcast
            else:
                sigma2[i] += param[p + j] * sigma2[i - j]
    return sigma2
        
