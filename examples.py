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


class RMalpha(object):
    """
    step size generator of the form
        alpha = c*t^{-mu}
    """
    def __init__(self, c,  mu):
        self.mu = mu
        self.c = c
        self.t = 0.

    def __iter__(self):
        return self

    def next(self):
        self.t += 1.
        return self.c * self.t ** (-self.mu)


def baird_star():
    n = 3
    n_iter = 5000
    beh_pi = np.ones((n + 1, 2))
    beh_pi[:, 0] = float(n) / (n + 1)
    beh_pi[:, 1] = float(1) / (n + 1)

    target_pi = np.zeros((n + 1, 2))
    target_pi[:, 0] = 0
    target_pi[:, 1] = 1
    

    target_pi = beh_pi

    star_mdp = BairdStarExample(n)


    theta_td0 = np.asarray(n * [1] + [10, 1])
    theta_tdc = theta_td0.copy()    
    theta_gtd2 = theta_td0.copy()

    td0 = td.LinearTD0(0.1, star_mdp.phi, gamma=0.99)
    gtd2 = td.GTD2(0.05, 0.5, star_mdp.phi, gamma=0.99)
    tdc = td.TDC(0.1, 0.1, star_mdp.phi, gamma=0.99)

    #theta_hist = np.nan * np.ones((n_iter, n + 2))
    i = 0
    for s, a, s_n, r in star_mdp.sample_transition(max_n=n_iter,
                                                    policy=beh_pi):
        print s,"-",a,"->",s_n
#        theta_gtd2 = gtd2.update_V_offpolicy(s, s_n, r, a, theta_gtd2,
#                           beh_pi, target_pi)
                           
        theta_tdc = tdc.update_V_offpolicy(s, s_n, r, a, theta_tdc,
                           beh_pi, target_pi)

        theta_td0 = td0.update_V_offpolicy(s, s_n, r, a, theta_td0,
                           beh_pi, target_pi)
        #theta_hist[i, :] = theta
        i += 1
        if i % 1 == 0:
            print "TD(0)", theta_td0
#            print "GTD2", theta_gtd2            
            print "TDC", theta_tdc


def random_walk_lin():
    n = 5
    n_iter = 6000
    n_indep = 20
    rw_mdp = RandomWalkChain(n)
    mu = rw_mdp.stationary_distrubution(seed=5, iterations=10000)
    print mu
    phi = rw_mdp.tabular_phi
    T = measures.bellman_operator(rw_mdp, gamma=1)
    Pi = measures.projection_operator(rw_mdp, mu, phi)

    # define the methods to examine          
    gtd2 = td.GTD2(0.1, 0.05, phi)
    gtd2.name = "GTD2"
    gtd2.color = "#0F6E08"
    
    gtd = td.GTD(0.1, 0.05, phi)
    gtd.name = "GTD"
    gtd.color = "#6E086D"
    
    td0 = td.LinearTD0(0.1, phi)
    td0.name = "TD(0)"    
    td0.color = "k"
    
    
    methods = [td0, gtd, gtd2]

    for i in [1, 2, 0.5, 0.25]:
        tdc = td.TDC(0.1, i*0.1, phi)
        tdc.name = "TDC alpha=0.1 mu={}".format(i)   
        tdc.color = "r"        
        methods.append(tdc)
        
    for i in np.linspace(0,1,5):
        lstd = td.LSTDLambda(i, phi)
        lstd.name = "LSTD({})".format(i)    
        lstd.color = "b"        
        methods.append(lstd)
    
    # define the evaluation measures
    mspbe = defaultdict(lambda: np.zeros(n_iter))    
    msbe = defaultdict(lambda: np.zeros(n_iter))
    with ProgressBar() as p:
        for seed in range(n_indep):
            p.update(seed, n_indep, "{} of {} seeds".format(seed, n_indep))
            i = 0   
            for m in methods:
                m.reset()
                theta = defaultdict(lambda: np.zeros(len(phi(0))))

            for s, a, s_n, r in rw_mdp.sample_transition(max_n=n_iter, seed=seed):
                for m in methods:
                    theta[m.name] = m.update_V(s, s_n, r, theta[m.name])
                
                
                if i % 1 == 0:  
                    for k, v in theta.items():
                        u =  np.sqrt(measures.MSPBE(v, mu, rw_mdp, phi, T=T, Pi=Pi))
                        mspbe[k][i] += u
                        u =  np.sqrt(measures.MSBE(v, mu, rw_mdp, phi, T=T))
                        msbe[k][i] += u
                i += 1
            
  
    print "Theta:", theta

    plt.figure()
    plt.ylabel("RMSPBE")
    plt.xlabel("Timesteps")    
    for m in methods:    
        plt.plot(mspbe[m.name]/n_indep, label=m.name, color=m.color)
    plt.legend()
    
    
    plt.figure()
    plt.ylabel("RMSBE")
    plt.xlabel("Timesteps")    
    for m in methods:    
        plt.plot(msbe[m.name]/n_indep, label=m.name, color=m.color)
    plt.legend()
    
    plt.show()


random_walk_lin()

def random_walk():
    rw_mdp = RandomWalkChain(7)

    V = np.zeros(7)
    
    tdl = td.TabularTDLambda(0.9, 0.4)
    for eps in rw_mdp.sample_episodes(5000):
        for s, a, s_n, r in rw_mdp.extract_transitions(eps):
            V = tdl.update_V(s, s_n, r, V)

    print V
    
