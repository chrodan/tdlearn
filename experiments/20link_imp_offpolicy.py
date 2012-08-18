import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
import util
import features
import policies
from joblib import Parallel, delayed
from task import LinearLQRValuePredictionTask
import itertools

gamma=0.9

dt = 0.1
dim = 20
#sigma = np.zeros((2*dim,2*dim))
sigma = np.ones(2*dim)*0.01


mdp = examples.NLinkPendulumMDP(np.ones(dim), np.ones(dim)*5, sigma=sigma, dt=dt)
phi = features.squared_diag()


n_feat = len(phi(np.zeros(mdp.dim_S)))
theta_p,_,_ = dp.solve_LQR(mdp, gamma=gamma)
#print theta_p
theta_p = np.array(theta_p)

policy = policies.LinearContinuous(theta=theta_p, noise=np.eye(dim)*0.1)
target_policy = policies.LinearContinuous(theta=theta_p, noise=np.eye(dim)*0.05)
#theta0 =  10*np.ones(n_feat)
theta0 =  0.*np.ones(n_feat)

task = LinearLQRValuePredictionTask(mdp, gamma, phi, theta0, policy=policy, target_policy=target_policy, normalize_phi=True)


methods = []

alpha = 0.0005
mu = 2 #optimal
gtd = td.GTD(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
#methods.append(gtd)


alpha, mu = 0.0005, 2 #optimal
gtd = td.GTD2(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
#methods.append(gtd)

alpha = .005
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha$={}".format(alpha)
td0.color = "k"
methods.append(td0)

alpha, mu = (.001,0.5)
tdc = td.TDC(alpha=alpha, beta=alpha*mu, phi=phi, gamma=gamma)
tdc.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)
tdc.color = "b"
#methods.append(tdc)

lstd = td.LSTDLambda(lam=0, phi=phi, gamma=gamma, eps=1000)
lstd.name = r"LSTD({})".format(0)
lstd.color = "g"
methods.append(lstd)

lstd = td.LSTDLambdaJP(lam=0, phi=phi, gamma=gamma)
lstd.name = r"LSTD-JP({})".format(0)
lstd.color = "g"
methods.append(lstd)

alpha=.008
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)

l=30000
error_every=200
n_indep=50
criterion="RMSPBE"
title="Impoverished 20-Link Pole Balancing Off-policy"
name="20link_imp_offpolicy"
if __name__ =="__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=1, **globals())
    save_results(**globals())
    #plot_errorbar(**globals())