# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 18:42:16 2012

@author: Christoph Dann <cdann@cdann.de>
"""
import numpy as np


def prepare_MSE(mu, mdp, phi, V_true):
    
    Phi = Phi_matrix(mdp, phi)
    return lambda theta: np.sum(((theta * np.asarray(Phi)).sum(axis=1) - V_true)**2 * mu)

def prepare_MSPBE(mu, mdp, phi, gamma=1, policy="uniform"):
    
    Phi = Phi_matrix(mdp, phi)
    D = np.diag(mu)
    Pi = Phi * np.linalg.inv(Phi.T * D * Phi) * Phi.T * D
    T = bellman_operator(mdp, gamma, policy)
        
    def _MSPBE(theta):
        V = (theta * np.asarray(Phi)).sum(axis=1)
        v = np.asarray(V - np.dot(Pi, T(V)))
        return np.sum(v**2 * mu)
    
    return _MSPBE
    
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
        Pi = Phi * np.linalg.inv(Phi.T * D * Phi) * Phi.T * D
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
    Pi = Phi * np.linalg.inv(Phi.T * D * Phi) * Phi.T * D
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
    
    for s in mdp.states:
        
        R = mdp.P * mdp.r * policy[:, :, np.newaxis]
        R = np.sum(R, axis=1) # sum over all A
        R = np.sum(R, axis=1) # sum over all S'
        
        P = mdp.P * policy[:, :, np.newaxis]
        P = np.sum(P, axis=1) # sum over all A => p(s' | s)
    
    return lambda V: R + gamma * np.dot(P, V)
    