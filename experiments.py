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

def baird_star_gridsearch():
    mdp = BairdStarExample(7)
    methods = []
    alpha_range = [0.001,0.005, 0.01, 0.03, 0.1]
    mu_range = [0.1, 1, 3, 10]
    grid = np.zeros((len(alpha_range),len(mu_range)))
    for a, alpha in enumerate(alpha_range):
        for j, mu in enumerate(mu_range):
            m = td.TDC(alpha=alpha, beta=mu*alpha, phi=mdp.phi, gamma=0.99, 
                                   theta0=np.asarray(7 * [1] + [10, 1]))        
            m.name = "TDC alpha={} mu={}".format(alpha, mu) 
            m.gridpos = (a,j)
            methods.append(m)
    theta = baird_star_run(methods)
    size = {}
    for m in methods:
    #for k,v in theta.items():
        size[m.name] = np.sqrt((theta[m.name]**2).sum(axis=1))        
        grid[m.gridpos] = size[m.name][-1]
        print m.name, ":", size[m.name][-1]
    plt.imshow(grid, interpolation="nearest")
    plt.ylabel(r"$\alpha$")
    plt.xlabel(r"$\mu$")
    plt.yticks(np.arange(len(alpha_range)), alpha_range)    
    plt.xticks(np.arange(len(mu_range)), mu_range)
    plt.colorbar()
    plt.show()

def baird_star_run(methods, n=7, n_iter=1000, seed=5):
    beh_pi = np.ones((n + 1, 2))
    beh_pi[:, 0] = float(n) / (n + 1)
    beh_pi[:, 1] = float(1) / (n + 1)

    target_pi = np.zeros((n + 1, 2))
    target_pi[:, 0] = 0
    target_pi[:, 1] = 1
    
    star_mdp = BairdStarExample(n)

    theta = {}
    for m in methods:
        theta[m.name] = np.nan * np.ones((n_iter + 1, len(m.phi(0))))
        m.reset()
        theta[m.name][0,:] = m.theta
    i=1
    for s, a, s_n, r in star_mdp.sample_transition(n_iter, policy=beh_pi, seed=seed):
        for m in methods:
            theta[m.name][i,:] = m.update_V_offpolicy(s, s_n, r, a, beh_pi, target_pi)
        i += 1 
    return theta
    
def baird_indistinct():
    b=7
    star_mdp = BairdStarExample(n)
    n_iter = 3000
    n_indep = 50
    methods = []
    
    size = defaultdict(lambda: np.zeros(n_iter+1))  
    
    for alpha, mu, color in [(0.01, 10, "red"), (0.03, 0.1, "orange")]:
        
        #for mu in [0.1, 10, 20]:
            m = td.TDC(alpha=alpha, beta=mu*alpha, phi=star_mdp.phi, gamma=0.99, 
                                   theta0=np.asarray(7 * [1] + [10, 1]))        
            m.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)        
            m.color = color            
            methods.append(m)   
    for alpha, beta, c, color in [(0.52, 0.51, 0.07, "brown"), (0.52, 0.51, 0.7, "cyan")]:
        
            m = td.TDC(alpha=RMalpha(c, alpha), beta=RMalpha(c, beta), phi=star_mdp.phi, gamma=0.99, 
                                   theta0=np.asarray(7 * [1] + [10, 1]))        
            m.name = r"TDC $\alpha={c}t^{{{}}}$ $\beta={c}t^{{{}}}$".format(alpha, beta, c=c)        
            m.color = color            
            methods.append(m)
            
    for alpha, color in [(0.01, "#00AA00"), (0.03, "#00AAAA")]:
        m = td.LinearTD0(alpha=alpha, phi=star_mdp.phi, theta0=np.asarray(7 * [1] + [10, 1]), gamma=0.99)
        m.name = r"TD(0) $\alpha$={}".format(alpha)    
        m.color = color                                
        methods.append(m)   
    
    
    with ProgressBar() as p:
        for seed in range(n_indep):
            p.update(seed, n_indep, "{} of {} seeds".format(seed, n_indep))
            theta = baird_star_run(methods, n_iter=n_iter, n=n, seed=seed)
            for k,v in theta.items():
                size[k] += np.sqrt((v**2).sum(axis=1)) / n_indep
 
            for m in methods:
                m.reset()
                        
    plt.figure()
    plt.ylabel(r"$\|\theta\|_2$")
    plt.xlabel("Timesteps")    
    for m in methods: 
        plt.semilogy(size[m.name], label=m.name, color=m.color)
    plt.ylim(0,10**7)
    plt.legend(loc="lower right")
    

    
    plt.show()

def baird_star():
    n = 7
    n_iter = 100
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



def boyan_chain():
    n = 13
    n_feat = 4
    n_iter = 300
    n_indep = 5
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
        
    for i in np.linspace(0,1,3):
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

    for m in methods:
        print m.name, m.time        
        
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
    phi = rw_mdp.tabular_phi
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
    

