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

target_pi = np.zeros((n + 1, 2))
target_pi[:, 0] = 0
target_pi[:, 1] = 1

mdp = examples.BairdStarExample(n)
phi = features.linear_blended(n+1)

methods = []

gamma=0.99
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, 
                                       np.asarray(n * [1.] + [10., 1.]),
                                       policy=policies.Discrete(beh_pi), 
                                       target_policy=policies.Discrete(target_pi))

alpha = 0.003
mu = 0.5 #optimal
gtd = td.GTD(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)

#for alpha in [.005,0.01,0.02]:
#    for mu in [0.01, 0.1]:
alpha, mu = 0.007, 0.5 #optimal
gtd = td.GTD2(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)


alpha = .005
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha$={}".format(alpha)
td0.color = "m"
methods.append(td0)


alpha = 0.005
mu = 0.01
tdc = td.TDC(alpha=alpha, beta=alpha*mu, phi=phi, gamma=gamma)
tdc.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)
tdc.color = "b"
methods.append(tdc)


alpha = 0.005
mu = 0.01
tdc = td.GeriTDC(alpha=alpha, beta=alpha*mu, phi=phi, gamma=gamma)
tdc.name = r"GeriTDC $\alpha$={} $\mu$={}".format(alpha, mu)
tdc.color = "c"
methods.append(tdc)

lstd = td.LSTDLambda(lam=0, phi=phi, gamma=gamma)
lstd.name = r"LSTD({})".format(0)
lstd.color = "g"
methods.append(lstd)
#
#methods = []
#for alpha in [0.01, 0.02, 0.03]:
#alpha = .2
alpha=.02
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)


lstd = td.LSTDLambdaJP(lam=0, phi=phi, gamma=gamma)
lstd.name = r"LSTD-JP({})".format(0)
lstd.color = "k"
methods.append(lstd)


                    
l=700
error_every=1
n_indep=50
name="baird"
title="Baird Star"
criterion="RMSPBE"

if __name__ =="__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=-1, **globals())
    save_results(**globals())
    plot_errorbar(**globals())
