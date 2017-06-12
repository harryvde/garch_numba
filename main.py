# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 22:41:08 2017

@author: Harry
"""

from garch_pq import garchpq
import numpy as np

N = 1000
param = np.array([.1, .1 , .8])
ret = np.random.randn(N)
p,q=1,1

gg = garchpq(ret,p,q)
pp = gg.estimate()
print pp

