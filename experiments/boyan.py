# -*- coding: utf-8 -*-
"""
Convergence speed comparison of TD methods on the Boyan chain example

Created on Mon Jan 30 21:06:12 2012

@author: Christoph Dann <cdann@cdann.de>
"""

import td
import examples
from task import LinearDiscreteValuePredictionTask
import numpy as np
import matplotlib.pyplot as plt
import features

n = 14
n_feat = 4
mdp = examples.BoyanChain(n, n_feat)
phi = features.spikes(n_feat, n)
gamma = 1.
p0 = np.zeros(n_feat)
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, p0)

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

#methods = []
#for eps in np.power(10,np.arange(-1,4)):
lstd = td.LSTDLambda(lam=0, phi=phi)
lstd.name = r"LSTD({})".format(0)
lstd.color = "b"
methods.append(lstd)

brm = td.BRM(phi=phi)
brm.name = "BRM"
brm.color = "b"
methods.append(brm)

#methods = []
#for alpha in np.linspace(0.01,1,10):
alpha = 0.9
rg = td.ResidualGradient(alpha=alpha, phi=phi)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)

eta = 0.001
reward_noise = 0.001
P_init = 1.
ktd = td.KTD(phi=phi, gamma=1., P_init=P_init, theta_noise=None, eta=eta,
             reward_noise=reward_noise)
ktd.name = r"KTD $\eta$={}, $\sigma^2$={} $P_0$={}".format(
    eta, reward_noise, P_init)
methods.append(ktd)



sigma = 1.
gptdp = td.GPTDP(phi=phi, sigma=sigma)
gptdp.name = r"GPTDP $\sigma$={}".format(sigma)
methods.append(gptdp)


l = 20
n_eps = 150
episodic = True
error_every = 1
name = "boyan"
n_indep = 4
title = "{}-State Boyan Chain ({} trials)".format(n, n_indep)
criterion = "RMSPBE"
criteria = ["RMSPBE", "RMSBE", "RMSE"]

if __name__ == "__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=-1, **globals())
    save_results(**globals())
    plot_errorbar(**globals())
