# -*- coding: utf-8 -*-
"""
Experiment that shows arbitrary off-policy behavior of TDC and TD

Created on Tue Jan 31 12:13:51 2012

@author: Christoph Dann <cdann@cdann.de>
"""
import td
import examples
import numpy as np
import features
import matplotlib.pyplot as plt
from task import LinearDiscreteValuePredictionTask
import policies
n = 7
beh_pi = np.ones((n + 1, 2))
beh_pi[:, 0] = float(n) / (n + 1)
beh_pi[:, 1] = float(1) / (n + 1)
beh_pol = policies.Discrete(prop_table=beh_pi)
target_pi = np.zeros((n + 1, 2))
target_pi[:, 0] = 0
target_pi[:, 1] = 1
target_pol = policies.Discrete(prop_table=target_pi)

mdp = examples.BairdStarExample(n)
phi = features.linear_blended(n + 1)

methods = []

gamma = 0.99
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi,
                                         np.asarray(n * [1.] + [10., 1.]),
                                         policy=beh_pol,
                                         target_policy=target_pol)

alpha = 0.004
mu = 4
gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)

alpha = 0.005
mu = 4.
gtd = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)


alpha = .00002
lam = 1.
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
methods.append(td0)


alpha = 0.006
mu = 16
tdc = td.TDC(alpha=alpha, beta=alpha * mu, phi=phi, gamma=gamma)
tdc.name = r"TDC(0) $\alpha$={} $\mu$={}".format(alpha, mu)
tdc.color = "b"
methods.append(tdc)


alpha = 0.005
mu = 8
tdc = td.GeriTDC(alpha=alpha, beta=alpha * mu, phi=phi, gamma=gamma)
tdc.name = r"GeriTDC $\alpha$={} $\mu$={}".format(alpha, mu)
tdc.color = "c"
methods.append(tdc)

alpha = .02
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)

lam = 1.
lstd = td.RecursiveLSTDLambda(lam=lam, phi=phi, gamma=gamma)
lstd.name = r"LSTD({})".format(lam)
lstd.color = "k"
methods.append(lstd)

lam = 1.
alpha = 1.
lspe = td.RecursiveLSPELambda(lam=lam, alpha=alpha, phi=phi)
lstd.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
methods.append(lstd)

lam = 1.
alpha = 1.
lstd = td.FPKF(lam=lam, alpha=alpha, phi=phi)
lstd.name = r"FPKF({}) $\alpha={}$".format(lam, alpha)
methods.append(lstd)

brm = td.BRM(phi=phi)
brm.name = "BRM"
brm.color = "b"
methods.append(brm)

lam = 1.
lstd = td.RecursiveLSTDLambdaJP(lam=lam, phi=phi, gamma=gamma)
lstd.name = r"LSTD-JP({})".format(0)
lstd.color = "k"
methods.append(lstd)


l = 1000
error_every = 1
n_indep = 200
n_eps = 1
episodic=False
name = "baird"
title = "Baird Star"
criterion = "RMSPBE"
criteria = ["RMSPBE", "RMSBE", "RMSE"]
gs_errorevery=20

if __name__ == "__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=-1, **globals())
    save_results(**globals())
    plot_errorbar(**globals())
