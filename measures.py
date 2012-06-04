# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 18:42:16 2012

@author: Christoph Dann <cdann@cdann.de>
"""
import numpy as np
import mdp as mdpm

def prepare_MSE(mu, mdp, phi, V_true):
    if isinstance(mdp, mdpm.LQRMDP):
        return prepare_LQR_MSE(mu, mdp, phi, V_true)
    else:
        return prepare_discrete_MSE(mu, mdp, phi, V_true)
 
def prepare_MSPBE(mu, mdp, phi, gamma, policy):
    if isinstance(mdp, mdpm.LQRMDP):
        return prepare_LQR_MSPBE(mu, mdp, phi, gamma, policy)
    else:
        return prepare_discrete_MSPBE(mu, mdp, phi, gamma, policy)
      
def prepare_MSBE(mu, mdp, phi, gamma, policy):
    if isinstance(mdp, mdpm.LQRMDP):
        return prepare_LQR_MSBE(mu, mdp, phi, gamma, policy)
    else:
        return prepare_discrete_MSBE(mu, mdp, phi, gamma, policy)
    
def prepare_LQR_MSE(mu_full, mdp, phi, V_true):
    def a(theta): 
        p = phi.retransform(theta).flatten()-V_true.flatten()
        return np.mean((p * mu_full).sum(axis=1)**2)
    return a
    
    
def prepare_discrete_MSE(mu, mdp, phi, V_true):
    Phi = Phi_matrix(mdp, phi)
    return lambda theta: np.sum(((theta * np.asarray(Phi)).sum(axis=1) - V_true)**2 * mu)

def prepare_SMSPBE(mu, mdp, phi, gamma=1, policy="uniform"):
    """ statewise Mean Squared Projected Bellman Error"""
    Phi = Phi_matrix(mdp, phi)
    D = np.diag(mu)
    Pi = Phi * np.linalg.pinv(Phi.T * D * Phi) * Phi.T * D
    T = bellman_operator(mdp, gamma, policy)
        
    def _sMSPBE(theta):
        V = (theta * np.asarray(Phi)).sum(axis=1)
        v = np.asarray(V - np.dot(Pi, T(V))).flatten()
        return v**2
    
    return _sMSPBE

def prepare_discrete_MSPBE(mu, mdp, phi, gamma=1, policy="uniform"):
    """ Mean Squared Projected Bellman Error """
    Phi = Phi_matrix(mdp, phi)
    D = np.diag(mu)
    Pi = Phi * np.linalg.pinv(Phi.T * D * Phi) * Phi.T * D
    T = bellman_operator(mdp, gamma, policy)
        
    def _MSPBE(theta):
        V = (theta * np.asarray(Phi)).sum(axis=1)
        v = np.asarray(V - np.dot(Pi, T(V)))
        return np.sum(v**2 * mu)
    
    return _MSPBE
    
def prepare_LQR_MSPBE(mu_samples, mdp, phi, gamma=1, policy="uniform"):
    """ Mean Squared Projected Bellman Error """
    mu, mu_phi = mu_samples 
    Phi = np.matrix(mu_phi)
    Pi = np.linalg.pinv(Phi.T * Phi) * Phi.T
    T = bellman_operator_LQR(mdp, gamma, policy)
    #print "Compute MSPBE from",mu.shape[0],"samples"
    def _MSPBE(theta):
        V = np.matrix((theta * np.asarray(Phi)).sum(axis=1)).T
        #import ipdb; ipdb.set_trace()
        v = np.asarray(V - Phi * Pi * mu * T(phi.retransform(theta)).flatten().T)
        
        return np.mean(v**2)
    
    return _MSPBE
    
def prepare_LQR_MSBE(mu_samples, mdp, phi, gamma=1, policy="uniform"):
    """ Mean Squared Bellman Error """
    mu_phi_full, mu_phi = mu_samples
    T = bellman_operator_LQR(mdp, gamma, policy)
    #print "Compute MSBE from",mu.shape[0],"samples"
    def _MSBE(theta):
        V = np.array((theta * mu_phi).sum(axis=1))
        proj_V = np.array(T(phi.retransform(theta))).flatten()
        V2 = np.array((proj_V * mu_phi_full).sum(axis=1))
        
        #import ipdb; ipdb.set_trace()
        return np.mean((V-V2)**2)
    
    return _MSBE
    
def prepare_discrete_MSBE(mu, mdp, phi, gamma=1, policy="uniform"):
    
    Phi = Phi_matrix(mdp, phi)
    T = bellman_operator(mdp, gamma, policy)
        
    def _MSBE(theta):
        V = (theta * np.asarray(Phi)).sum(axis=1)
        #import ipdb; ipdb.set_trace()
        v = np.asarray(V - T(V))
        return np.sum(v**2 * mu)
    
    return _MSBE
    
def MSPBE(theta, mu, mdp, phi, gamma=1, policy="uniform", Pi=None, T=None):
    """
    compute the Mean Squared Projected Bellman Error
    for a given Markov Decision Process mdp and a parameter theta assuming:
    a linear value function approximation
    
    Inputs:
        theta:  parameter vector to evaluate
        mu:     stationary distribution of mdp
        mdp:    Markov Decision Process, instance of mdp.MDP
        phi:    feature function: S -> R^d given as python function
        policy: S x A -> R
                numpy array of shape (n_s, n_a)
                *_pi(s,a) is the probability of taking action a in state s
        
        to speed up computation the following parameters can be provided:
            projection opterator Pi and bellman operator T
    """


    Phi = Phi_matrix(mdp, phi)
    V = (theta * np.asarray(Phi)).sum(axis=1)
    if Pi is None:
        D = np.diag(mu)
        Pi = Phi * np.linalg.pinv(Phi.T * D * Phi) * Phi.T * D
    if T is None:
        T = bellman_operator(mdp, gamma, policy)
    v = np.asarray(V - np.dot(Pi, T(V)))
    return np.sum(v**2 * mu)

def MSBE(theta, mu, mdp, phi, gamma=1, policy="uniform", T=None):
    """
    compute the Mean Squared Bellman Error
    for a given Markov Decision Process mdp and a parameter theta assuming:
    a linear value function approximation
    
    Inputs:
        theta:  parameter vector to evaluate
        mu:     stationary distribution of mdp
        mdp:    Markov Decision Process, instance of mdp.MDP
        phi:    feature function: S -> R^d given as python function
        policy: S x A -> R
                numpy array of shape (n_s, n_a)
                *_pi(s,a) is the probability of taking action a in state s
    """

    Phi = Phi_matrix(mdp, phi)
    V = (theta * np.asarray(Phi)).sum(axis=1)
    if T is None:
        T = bellman_operator(mdp, gamma, policy)
    
    v = np.asarray(V - T(V))
    return np.sum(v**2 * mu)

def Phi_matrix(mdp, phi):
    """
    produce the feature representation of all states of mdp as a vertically
    stacked matrix,
        mdp:    Markov Decision Process, instance of mdp.MDP
        phi:    feature function: S -> R^d given as python function
        
        returns: numpy matrix of shape (n_s, dim(phi)) where
                Phi[i,:] = phi(S[i])
    """
    if phi in mdp.Phi:
        Phi = mdp.Phi[phi]
    else:
        Phil = []
        for s in mdp.states:
            f = phi(s)
            Phil.append(f)
        Phi = np.matrix(np.vstack(Phil))
        mdp.Phi[phi] = Phi
    return Phi

def projection_operator(mdp, mu, phi, policy="uniform"):
    """
    compute the projection operator Pi
    for a given Markov Decision Process mdp for a linear :
    value function approximation
    
    Inputs:
        theta:  parameter vector to evaluate
        mu:     stationary distribution of mdp
        mdp:    Markov Decision Process, instance of mdp.MDP
        phi:    feature function: S -> R^d given as python function
        policy: S x A -> R
                numpy array of shape (n_s, n_a)
                *_pi(s,a) is the probability of taking action a in state s
        
        to speed up computation the following parameters can be provided:
            projection opterator Pi and bellman operator T
    """

    Phil = []
    for s in mdp.states:
        f = phi(s)
        Phil.append(f)
    Phi = np.matrix(np.vstack(Phil))
    D = np.diag(mu)
    Pi = Phi * np.linalg.pinv(Phi.T * D * Phi) * Phi.T * D
    return Pi
    
def bellman_operator(mdp, gamma, policy="uniform"):    
    """
    returns a the bellman operator of a given MRP (MDP with policy)
    as a python function which takes the value function represented as a numpy
    array
        T(V) = R + gamma * P * V
    
    details see Chapter 3 of 
    Sutton, R. S., Maei, H. R., Precup, D., Bhatnagar, S., Silver, D., 
    Szepesvari, C., & Wiewiora, E. (2009).: Fast gradient-descent methods for 
    temporal-difference learning with linear function approximation.
    """

    if policy is "uniform":
        policy = mdp.uniform_policy()    
    

    R = mdp.P * mdp.r * policy[:, :, np.newaxis]
    R = np.sum(R, axis=1) # sum over all A
    R = np.sum(R, axis=1) # sum over all S'
    
    P = mdp.P * policy[:, :, np.newaxis]
    P = np.sum(P, axis=1) # sum over all A => p(s' | s)
    
    return lambda V: R + gamma * np.dot(P, V)
    
def bellman_operator_LQR(lqmdp, gamma, policy="uniform"):    
    """
    returns a the bellman operator of a given LQR MRP (MDP with policy)
    as a python function which takes the value function s^T P s represented as a numpy
    aquared array P
        T(P) = R + theta_p^T Q theta_p + gamma * (A + B theta_p)^T P (A + B theta_p)
    
    """

    if policy is "normal":
        policy = lqmdp.normal_policy()    
    
    Q = np.matrix(lqmdp.Q)
    R = np.matrix(lqmdp.R)
    theta = np.matrix(policy.theta)
    A = np.matrix(lqmdp.A)
    B = np.matrix(lqmdp.B)
    Sigma = np.matrix(lqmdp.Sigma)
    
    S = A+ B * theta
    C = Q + theta.T * R * theta
    #import ipdb; ipdb.set_trace()
    return lambda V: C + gamma * (S.T * np.matrix(V) * S + np.trace(np.matrix(V)*Sigma))
    
