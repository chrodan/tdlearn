# -*- coding: utf-8 -*-
"""
Dynamic programming value function estimation
Created on Sun Jan 22 00:51:55 2012

@author: Christoph Dann <cdann@cdann.de>
"""
import numpy as np

def estimate(mdp, n_iter=100000, policy="uniform", gamma=1):
    if policy =="uniform":
        policy =mdp.uniform_policy()
        
    P = mdp.P * policy[:,:,np.newaxis]
    P = P.sum(axis=1)
    P /= P.sum(axis=1)[:, np.newaxis]

    r = mdp.r * policy[:,:,np.newaxis]
    r = r.sum(axis=1)
    
    V = np.zeros(len(mdp.states))
    for i in xrange(n_iter):
        V_n = (P * (gamma * V + r)).sum(axis=1)
        if np.linalg.norm(V - V_n) < 1e-22:
            V = V_n            
            break
        V = V_n
    return V