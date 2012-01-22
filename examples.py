# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 21:04:24 2011

@author: christoph
"""
import mdp
import td
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import measures
from collections import defaultdict
from util.progressbar import ProgressBar


class RandomMDP(mdp.MDP):
    
    def __init__(self, n_states, seed=None):
        if seed is not None:
            np.random.seed(seed)
        n_s = n_states        
        actions = [0, ]
        states = range(n_s)
        
        d0 = np.random.rand(n_s) + 1e-5
        d0 = d0 / d0.sum()
        
        r = np.random.rand(n_s, 1, n_s)
        r[:, :, n_s - 1] = 1
        P = np.random.rand(n_s, 1, n_s) + 1e-5
        P /= P.sum(axis=2)[:,:,np.newaxis]

        mdp.MDP.__init__(self, states, actions, r, P, d0)

class RandomWalkChain(mdp.MDP):
    """ Random Walk chain example MDP """
    # TODO: explain + reference

    def __init__(self, n_states, p_minus=0.5, p_plus=0.5):
        """
            n_states: number of states including terminal ones
            p_minus: probability of going left
            p_plus: probability of going right
        """
        n_s = n_states
        states = range(n_s)
        actions = [0, ]
        d0 = np.zeros(n_s)
        d0[n_s / 2] = 1
        r = np.zeros((n_s, 1, n_s))
        r[:, :, n_s - 1] = 1
        P = np.zeros((n_s, 1, n_s))
        P[0, :, 0] = 1
        P[n_s - 1, :, n_s - 1] = 1
        for s in np.arange(1, n_s - 1):
            P[s, :, s + 1] = p_plus
            P[s, :, s - 1] = p_minus

        mdp.MDP.__init__(self, states, actions, r, P, d0)

    def tabular_phi(self, state):
        """
        feature function that makes linear approximation equivalent to
        tabular algorithms
        """
        result = np.zeros(len(self.states))
        result[state] = 1.
        return result
        
    def dependent_phi(self, state):
        """
        feature function that produces linear dependent features
        the feature of the middle state is an average over all positions
        """
        n = len(self.states)
        l = n / 2 + 1
        #print l
        if state >= l:
            res = state - l < np.arange(l)
        else:
            res = state >= np.arange(l)
        res = res.astype("float")
        res /= np.sqrt(np.sum(res))
        return res
        
class BoyanChain(mdp.MDP):
    """
    Boyan Chain example. All states form a chain with ongoing arcs and a
    single terminal state. From a state one transition with equal probability
    to either the direct or the second successor state. All transitions have 
    a reward of -3 except going from second to last to last state (-2)    
    """        

    def __init__(self, n_states, n_feat):
        """
            n_states: number of states including terminal ones
            n_feat: number of features used to represent the states
                    n_feat <= n_states
        """
        assert n_states >= n_feat
        #assert (n_states - 1) % (n_feat - 1) == 0
        n_s = n_states
        self.n_feat = n_feat 
        states = range(n_s)
        actions = [0, ]
        d0 = np.zeros(n_s)
        d0[0] = 1
        r = np.ones((n_s, 1, n_s))*(-3)
        r[-2, :, -1] = -2
        P = np.zeros((n_s, 1, n_s))
        P[-1, :, -1] = 1
        P[-2, :, -1] = 1
        for s in np.arange(n_s - 2):
            P[s, :, s + 1] = 0.5
            P[s, :, s + 2] = 0.5

        mdp.MDP.__init__(self, states, actions, r, P, d0)
        
    def phi(self, state):       
        n = len(self.states)
        a = (n - 1.) / (self.n_feat - 1)
        r = 1 - abs((state - np.linspace(1,n,self.n_feat)) / a)
        r[r < 0] = 0
        return r
#        i = int(state / a)
#        f1 = max(1 - float(state - (1 + i * a)) / a, 0)
#        #f1 = float(state % a) / a # fraction second position
#        res = np.zeros(self.n_feat)
#        res[i] = f1
#        if state < n - 1:
#            res[i + 1] = 1 - f1
#        return res

class BairdStarExample(mdp.MDP):
    """
    Baird's star shaped example for off-policy divergence of TD(\lambda)
    contains of states ordered in a star shape with 2 possible actions:
        - a deterministic one always transitioning in the star center and
        - a probabilistic one going to one of the star ends with uniform
            probability
    for details see Braid (1995): Residual Algorithms: Reinforcement Learning
        with Function Approximation
    """

    def __init__(self, n_corners):
        """
            n_corners: number of ends of the star
        """
        n_s = n_corners + 1
        actions = ["dotted", "solid"]
        r = np.zeros((n_s, 2, n_s))
        P = np.zeros((n_s, 2, n_s))
        # solid action always go to star center
        P[:, 1, n_corners] = 1
        # dotted action goes to star corners uniformly
        P[:, 0, :n_corners] = 1. / n_corners
        # start uniformly
        d0 = np.ones((n_s), dtype="double") / n_s

        mdp.MDP.__init__(self, range(1, n_corners + 1) + ["center", ],
                     actions, r, P, d0)

    def phi(self, state):
        """
        official approximation function for this example

        taken from: Maei, H. R. (2011). Gradient Temporal-Difference Learning
                Algorithms. Machine Learning. University of Alberta.
                p. 17
        """
        n_corners = len(self.states) - 1
        result = np.zeros(n_corners + 2)
        if state == n_corners:
            result[-1] = 2
            result[-2] = 1
        else:
            result[-1] = 1
            result[state] = 2
        return result

