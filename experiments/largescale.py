# -*- coding: utf-8 -*-
"""
Quick speed comparison of LSTD and gradient TD approaches
on random MDPs with larger number of states

@author Christoph Dann<cdann@cdann.de>
"""

import td
import examples
import numpy as np
from task import LinearDiscreteValuePredictionTask

for n in [100, 500, 1000, 2500, 5000, 10000]:
    mdp = examples.RandomMDP(n)
    phi = mdp.tabular_phi
    n_iter = 100
    n_indep = 1
    methods = []
    
    
    task = LinearDiscreteValuePredictionTask(mdp, 0.9, phi, 
                                           np.zeros(n))
    
    alpha, color = 0.75, "orange"
    m = td.LinearTDLambda(alpha=alpha, phi=phi, lam=0)
    m.name = r"TD(0) $\alpha$={}".format(alpha)    
    m.color = color                                
    methods.append(m)   
    
    alpha, mu, color = 0.75, 0.1, "orange"
    m = td.GTD(alpha=alpha, beta=mu*alpha, phi=phi)
    m.name = r"GTD $\alpha$={}".format(alpha)    
    m.color = color                                
    methods.append(m)  
    
    alpha, mu, color = 0.75, 0.1, "orange"
    m = td.TDC(alpha=alpha, beta=mu*alpha, phi=phi)
    m.name = r"TDC $\alpha$={}".format(alpha)    
    m.color = color                                
    methods.append(m)  
        
    eps=100
    lstd = td.LSTDLambda(lam=0, eps=eps, phi=phi)
    lstd.name = r"LSTD({}) $\epsilon$={}".format(0, eps)    
    lstd.color = "b"        
    methods.append(lstd)
    
    task.ergodic_error_traces(methods, n_samples=n_iter, criterion="RMSPBE")

    print 20*"-"
    print "n=", n
    for i, m in enumerate(methods):
        print m.name, m.time
    