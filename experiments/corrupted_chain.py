# -*- coding: utf-8 -*-
"""
Convergence speed comparison of TD methods on the Boyan chain example

Created on Mon Jan 30 21:06:12 2012

@author: Christoph Dann <cdann@cdann.de>
"""

import td
import regtd
import examples
from task import LinearDiscreteValuePredictionTask
import numpy as np
import matplotlib.pyplot as plt
import features
import policies

n = 20
n_random = 800
mdp = examples.CorruptedChain(n_states=n)
phi = features.corrupted_rbfs(n_S=n, n_rbfs=5, n_random=n_random)
gamma = .9
n_feat = phi.dim
p0 = np.zeros(n_feat)
pol = np.zeros((n,2))
pol[:10, 0] = 1
pol[10:, 1] = 1
policy = policies.Discrete(prop_table=pol)
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, p0, policy=policy)

# define the methods to examine
gtd2 = td.GTD2(alpha=0.5, beta=0.5, phi=phi)
gtd2.name = "GTD2"
gtd2.color = "#0F6E08"

gtd = td.GTD(alpha=0.5, beta=0.5, phi=phi)
gtd.name = "GTD"
gtd.color = "#6E086D"

methods = []  # [td0, gtd, gtd2]

#for alpha in [0.5,0.7]:
#    for mu in [0.01, 0.005, 0.05]:
alpha = 1
mu = 0.01
gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "#6E086D"
methods.append(gtd)

#for alpha in [0.5,0.9, 1]:
#    for mu in [0.5, 0.3]:
alpha, mu = 0.9, 0.3
gtd = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "#6E086D"
methods.append(gtd)


for alpha in [0.3]:
    td0 = td.LinearTD0(alpha=alpha, phi=phi)
    td0.name = r"TD(0) $\alpha$={}".format(alpha)
    td0.color = "k"
    methods.append(td0)

#for alpha in [0.3,0.5,0.7]:
#    for mu in [0.01, 0.005, 0.05]:
for alpha, mu in[(0.7, 0.01)]:
    tdc = td.TDC(alpha=alpha, beta=alpha * mu, phi=phi)
    tdc.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)
    tdc.color = "r"
    methods.append(tdc)

methods = []

lstd = td.RecursiveLSTDLambdaJP(lam=0, eps=1000, phi=phi)
lstd.name = r"LSTD({}) $\ell_2 \tau={}$".format(0, 0)
lstd.color = "b"
methods.append(lstd)
#for eps in np.power(10,np.arange(-1,4)):
lstd = td.LSTDLambdaJP(lam=0, tau=0.8, phi=phi)
lstd.name = r"LSTD({}) $\ell_2 \tau={}$".format(0,.8)
lstd.color = "b"
methods.append(lstd)

tau = 5e-8
lstd = regtd.DLSTD(lam=0, phi=phi, tau=tau)
lstd.name = r"D-LSTD({}) $\tau={}$".format(0, tau)
lstd.color = "b"
methods.append(lstd)

tau = 0.0275
lstd = regtd.LarsTD(lam=0, phi=phi, tau=tau)
lstd.name = r"LarsTD({}) $\tau={}$".format(0, tau)
lstd.color = "b"
methods.append(lstd)

beta = 0.135
tau = 2e-6
lstd = regtd.LSTDl21(lam=0, phi=phi, beta=beta, tau=tau, lars=False)
lstd.name = r"LSTD({})-$\ell_{{2,1}}$ $\tau={}, \beta={}$".format(0, lstd.tau, lstd.beta)
lstd.color = "b"
methods.append(lstd)

beta = 0.135
tau = 2e-6
lstd = regtd.LSTDl21(lam=0, phi=phi, beta=beta, tau=tau, lars=True)
lstd.name = r"LSTD({})-$\ell_{{2,1}}$ $\tau={}, \beta={}$ lars".format(0, lstd.tau, lstd.beta)
lstd.color = "b"
methods.append(lstd)

tau = 2e-6
lstd = regtd.LSTDl1(lam=0, phi=phi, tau=tau, lars=True)
lstd.name = r"LSTD({})-$\ell_1$ $\tau={} lars$".format(0, lstd.tau)
lstd.color = "b"
methods.append(lstd)

tau = 2e-6
lstd = regtd.LSTDl1(lam=0, phi=phi, tau=tau, lars=False)
lstd.name = r"LSTD({})-$\ell_1$ $\tau={} block-desc$".format(0, lstd.tau)
lstd.color = "b"
methods.append(lstd)


lstd = regtd.LSTDRP(lam=0, phi=phi, dim_lower=300, seed=2)
lstd.name = r"LSTD-RP({}) d={}".format(0, 0)
lstd.color = "b"
methods.append(lstd)

l = 20
n_eps = 30
episodic = False
error_every = 50
name = "corrupted"
n_indep = 2
title = "{}-State Corrupted Chain ({} trials)".format(n, n_indep)
criterion = "RMSE"
criteria = ["RMSPBE", "RMSBE", "RMSE"]

if __name__ == "__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=2, **globals())
    #save_results(**globals())
    plot_errorbar(**globals())
