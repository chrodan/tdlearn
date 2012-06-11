# -*- coding: utf-8 -*-
"""
Dynamic programming value function estimation
Created on Sun Jan 22 00:51:55 2012

@author: Christoph Dann <cdann@cdann.de>
"""
import numpy as np
import logging
from measures import bellman_operator_LQR
import policies
def estimate_V_discrete(mdp, n_iter=100000, policy="uniform", gamma=1.):
    if policy =="uniform":
        policy =mdp.uniform_policy()
        
    P = mdp.P * policy.P[:,:,np.newaxis]
    P = P.sum(axis=1)
    P /= P.sum(axis=1)[:, np.newaxis]

    r = mdp.r * policy.P[:,:,np.newaxis]
    r = r.sum(axis=1)
    V = np.zeros(len(mdp.states))
    for i in xrange(n_iter):
        V_n = (P * (gamma * V + r)).sum(axis=1)
        if np.linalg.norm(V - V_n) < 1e-22:
            V = V_n
            logging.info("Convergence after {} iterations".format(i + 1))            
            break
        V = V_n
    return V
    
def estimate_V_LQR(lqmdp, policy, n_iter=100000, gamma=1., eps=1e-14):
    """ Evaluate the value function exactly fora given Linear-quadratic MDP
        the value function has the form
        V = s^T P s
        
        for the policy
        
        as_fun: returns V as a python function instead of P"""
        
    T = bellman_operator_LQR(lqmdp, gamma, policy)
    P = np.matrix(np.zeros((lqmdp.dim_S,lqmdp.dim_S)))
    b = 0.
    for i in xrange(n_iter):
        P_n, b_n = T(P,b) #Q + theta.T * R * theta + gamma * (A+ B * theta).T * P * (A + B * theta)     
        if np.linalg.norm(P - P_n) < eps and np.abs(b - b_n) < eps:
            print "Converged estimating V after ",i,"iterations"
            break
        P = P_n
        b = b_n
    return np.array(P), b
        
def solve_LQR(lqmdp, n_iter=100000, gamma=1., eps=1e-14):
    """ Solves exactly the Linear-quadratic MDP with 
        the value function has the form
        V* = s^T P* s and policy a = theta* s
        
        returns (theta*, P*)"""
        
    P = np.matrix(np.zeros((lqmdp.dim_S,lqmdp.dim_S)))
    Q = np.matrix(lqmdp.Q)
    R = np.matrix(lqmdp.R)
    b = 0.
    theta = np.matrix(np.zeros((lqmdp.dim_A, lqmdp.dim_S)))
    A = np.matrix(lqmdp.A)
    B = np.matrix(lqmdp.B)
    for i in xrange(n_iter):
        theta_n = - gamma * np.linalg.pinv(R + gamma * B.T * P * B) * B.T * P * A 
        T = bellman_operator_LQR(lqmdp, gamma, policies.LinearContinuous(theta=theta_n))
        P_n, b_n = T(P,b) #Q + theta.T * R * theta + gamma * (A+ B * theta).T * P * (A + B * theta)     
        if np.linalg.norm(P - P_n) < eps and np.abs(b - b_n) < eps and np.linalg.norm(theta - theta_n) < eps:
            print "Converged estimating V after ",i,"iterations"
            break
        P = P_n
        b = b_n
        theta = theta_n
    return np.asarray(theta), P,b
