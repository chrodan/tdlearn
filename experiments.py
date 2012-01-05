# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 15:50:45 2012

@author: Christoph Dann <cdann@cdann.de
"""
from examples import BairdStarExample, RandomWalkChain, BoyanChain
import measures
from collections import defaultdict
from util.progressbar import ProgressBar
import td
import numpy as np
import matplotlib.pyplot as plt

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
    n = 7
    n_iter = 5000
    beh_pi = np.ones((n + 1, 2))
    beh_pi[:, 0] = float(n) / (n + 1)
    beh_pi[:, 1] = float(1) / (n + 1)

    target_pi = np.zeros((n + 1, 2))
    target_pi[:, 0] = 0
    target_pi[:, 1] = 1
    

    #target_pi = beh_pi

    star_mdp = BairdStarExample(n)


    theta_td0 = np.asarray(n * [1] + [10, 1])
    theta_tdc = theta_td0.copy()    
    theta_gtd2 = theta_td0.copy()

    td0 = td.LinearTD0(0.1, star_mdp.phi, gamma=0.99)
    gtd2 = td.GTD2(0.05, 0.5, star_mdp.phi, gamma=0.99)
    tdc = td.TDC(0.05, 0.5, star_mdp.phi, gamma=0.99)

    #theta_hist = np.nan * np.ones((n_iter, n + 2))
    i = 0
    for i in xrange(5):
        dtheta_gtd2 = 0        
        dtheta_tdc = 0
        dtheta_td0 = 0
        
        for s, a, s_n, r in star_mdp.synchronous_sweep(policy=beh_pi, seed=i):
            print s,"-",a,"->",s_n
            dtheta_gtd2 += gtd2.update_V_offpolicy(s, s_n, r, a, theta_gtd2,
                               beh_pi, target_pi) - theta_gtd2
                               
            dtheta_tdc += tdc.update_V_offpolicy(s, s_n, r, a, theta_tdc,
                               beh_pi, target_pi) - theta_tdc
    
            dtheta_td0 += td0.update_V_offpolicy(s, s_n, r, a, theta_td0,
                               beh_pi, target_pi) - theta_td0
            #theta_hist[i, :] = theta
        i += 1
        theta_td0 += dtheta_td0
        theta_tdc += theta_tdc
        theta_gtd2 += theta_gtd2
        print "TD(0)", theta_td0
        print "GTD2", theta_gtd2            
        print "TDC", theta_tdc


def boyan_chain():
    n = 13
    n_feat = 4
    n_iter = 300
    n_indep = 20
    rw_mdp = BoyanChain(n, n_feat)
    mu = rw_mdp.stationary_distrubution(seed=50, iterations=10000)
    print mu
    phi = rw_mdp.phi
    T = measures.bellman_operator(rw_mdp, gamma=1)
    Pi = measures.projection_operator(rw_mdp, mu, phi)

    # define the methods to examine          
    gtd2 = td.GTD2(alpha=0.5, beta=0.5, phi=phi)
    gtd2.name = "GTD2"
    gtd2.color = "#0F6E08"
    
    gtd = td.GTD(alpha=0.5, beta=0.5, phi=phi)
    gtd.name = "GTD"
    gtd.color = "#6E086D"
    
    td0 = td.LinearTD0(alpha=0.5, phi=phi)
    td0.name = "TD(0)"    
    td0.color = "k"
    
    
    methods = [td0, gtd, gtd2]

    for alpha in [0.06, 0.5, 1]:
        tdc = td.TDC(alpha=alpha, beta=alpha*0.5, phi=phi)
        tdc.name = "TDC alpha={} mu={}".format(alpha, 0.5)   
        tdc.color = "r"        
        methods.append(tdc)
        
    for i in np.linspace(0,1,5):
        lstd = td.LSTDLambda(lam=i, phi=phi)
        lstd.name = "LSTD({})".format(i)    
        lstd.color = "b"        
        methods.append(lstd)
    
    # define the evaluation measures
    mspbe = defaultdict(lambda: np.zeros(n_iter))  
    with ProgressBar() as p:
        for seed in range(n_indep):
            p.update(seed, n_indep, "{} of {} seeds".format(seed, n_indep))
 
            for m in methods:
                m.reset()
                theta = defaultdict(lambda: np.zeros(len(phi(0))))
            
            for i in xrange(n_iter):
                for m in methods:
                    m.reset_trace()
                for s, a, s_n, r in rw_mdp.sample_transition(1000, with_restart=False, seed=i+n_iter*seed):
                    for m in methods:
                        theta[m.name] = m.update_V(s, s_n, r)
                    for k, v in theta.items():
                        u =  np.sqrt(measures.MSPBE(v, mu, rw_mdp, phi, T=T, Pi=Pi))
                        mspbe[k][i] += u
            
    plt.figure()
    plt.ylabel("RMSPBE")
    plt.xlabel("Timesteps")    
    for m in methods:    
        plt.plot(mspbe[m.name]/n_indep, label=m.name, color=m.color)
    plt.legend()
    
    

    
    plt.show()


def random_walk_lin():
    n = 5
    n_iter = 200
    n_indep = 100
    rw_mdp = RandomWalkChain(n)
    mu = rw_mdp.stationary_distrubution(seed=5, iterations=10000)
    print mu
    phi = rw_mdp.dependent_phi
    T = measures.bellman_operator(rw_mdp, gamma=1)
    Pi = measures.projection_operator(rw_mdp, mu, phi)

    # define the methods to examine            
    gtd2 = td.GTD2(alpha=0.06, beta=0.06, phi=phi)
    gtd2.name = "GTD2"
    gtd2.color = "#0F6E08"
    
    gtd = td.GTD(alpha=0.06, beta=0.06, phi=phi)
    gtd.name = "GTD"
    gtd.color = "#6E086D"
    
    td0 = td.LinearTD0(alpha=0.06, phi=phi)
    td0.name = "TD(0)"    
    td0.color = "k"
    
    
    methods = [gtd, td0, gtd2 ]

    for alpha in [0.06, 0.12]:
        tdc = td.TDC(alpha=alpha, beta=alpha*0.5, phi=phi)
        tdc.name = "TDC alpha={} mu={}".format(alpha, 0.5)   
        tdc.color = "r"        
        methods.append(tdc)
        
    i=0
    for eps in [0.01, 0.1]:
        lstd = td.LSTDLambda(eps=eps, lam=i, phi=phi)
        lstd.name = "LSTD({}), eps={}".format(i, eps)    
        lstd.color = "b"        
        methods.append(lstd)
    
    # define the evaluation measures
    mspbe = defaultdict(lambda: np.zeros(n_iter))    
    msbe = defaultdict(lambda: np.zeros(n_iter))
    with ProgressBar() as p:
        for seed in range(n_indep):
            p.update(seed, n_indep, "{} of {} seeds".format(seed, n_indep))
 
            for m in methods:
                m.reset()
                theta = defaultdict(lambda: np.zeros(len(phi(0))))
            
            for i in xrange(n_iter):
                for m in methods:
                    m.reset_trace()
                for s, a, s_n, r in rw_mdp.sample_transition(1000, with_restart=False, seed=i+n_iter*seed):
                    for m in methods:
                        theta[m.name] = m.update_V(s, s_n, r)
                    for k, v in theta.items():
                        u =  np.sqrt(measures.MSPBE(v, mu, rw_mdp, phi, T=T, Pi=Pi))
                        mspbe[k][i] += u
                        u =  np.sqrt(measures.MSBE(v, mu, rw_mdp, phi, T=T))
                        msbe[k][i] += u
            
  
    print "Theta:", theta

    plt.figure()
    plt.ylabel("RMSPBE")
    plt.xlabel("Episodes")    
    for m in methods:    
        plt.plot(mspbe[m.name]/n_indep, label=m.name, color=m.color)
    plt.legend()
    
    
    plt.figure()
    plt.ylabel("RMSBE")
    plt.xlabel("Episodes")    
    for m in methods:    
        plt.plot(msbe[m.name]/n_indep, label=m.name, color=m.color)
    plt.legend()
    
    plt.show()


#random_walk_lin()

def random_walk():
    rw_mdp = RandomWalkChain(7)

    V = np.zeros(7)
    
    tdl = td.TabularTDLambda(0.9, 0.4)
    for eps in rw_mdp.sample_episodes(5000):
        for s, a, s_n, r in rw_mdp.extract_transitions(eps):
            V = tdl.update_V(s, s_n, r, V)

    print V
    

