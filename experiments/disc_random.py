# -*- coding: utf-8 -*-
"""
Convergence speed comparison of TD methods on a (uniformly) random MDP
@author: Christoph Dann <cdann@cdann.de>
"""

import td
import examples
from task import LinearDiscreteValuePredictionTask
import numpy as np
import features
import policies
import regtd
n = 400
n_a = 10
n_feat = 200
mdp = examples.RandomMDP(n, n_a)
phi = features.lin_random(n_feat, n, constant=True)
gamma = .95
np.random.seed(3)
beh_pol = policies.Discrete(np.random.rand(n, n_a))
tar_pol = policies.Discrete(np.random.rand(n, n_a))
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, np.zeros(phi.dim),
                                         policy=beh_pol, target_policy=tar_pol)


methods = []
alpha = 0.007
mu = .0001
gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)

alpha, mu = 0.002, 2
gtd = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)

alpha = td.RMalpha(0.01, 0.1)
lam = .0
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
td0.color = "k"
methods.append(td0)

alpha = .004
lam = .2
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
td0.color = "k"
methods.append(td0)

lam = 0.
alpha = 0.002
mu = 0.05
tdc = td.TDCLambda(alpha=alpha, mu = mu, lam=lam, phi=phi, gamma=gamma)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
methods.append(tdc)

lam = 0.
alpha = 0.003
mu = 0.05
tdc = td.GeriTDCLambda(alpha=alpha, mu = mu, lam=lam, phi=phi, gamma=gamma)
tdc.name = r"TDC({})-CO $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
methods.append(tdc)

alpha = .001
lam = .0
lstd = td.RecursiveLSPELambda(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
lstd.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
methods.append(lstd)

alpha = .005
lam = .0
lstd = td.RecursiveLSPELambdaCO(lam=lam, alpha=alpha, eps=10., phi=phi, gamma=gamma)
lstd.name = r"LSPE({})-CO $\alpha$={}".format(lam, alpha)
lstd.color = "g"
methods.append(lstd)

lam = 0.
eps = 100
lstd = td.RecursiveLSTDLambdaJP(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD-CO({}) $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)

lam = 0.
eps = 100
lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)
#
alpha = 0.1
lam = .6
beta = 1.
mins = 1000
lstd = td.FPKF(lam=lam, alpha = alpha, mins=mins, beta=beta, phi=phi, gamma=gamma)
lstd.name = r"FPKF({}) $\alpha={}$ $\beta={}$ m={}".format(lam, alpha, beta, mins)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)

alpha = .005
rg = td.ResidualGradientDS(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG DS $\alpha$={}".format(alpha)
rg.color = "brown"
rg.ls = "--"
methods.append(rg)

alpha = .005
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)


brm = td.RecursiveBRMDS(phi=phi)
brm.name = "BRMDS"
brm.color = "b"
brm.ls = "--"
methods.append(brm)

brm = td.RecursiveBRM(phi=phi)
brm.name = "BRM"
brm.color = "b"
methods.append(brm)


tau=0.0001
lstd = regtd.LSTDl1(tau=tau, lam=0, phi=phi)
lstd.name = r"LSTD-l1({}) $\tau={}$".format(0,tau)
lstd.color = "b"
methods.append(lstd)

tau=0.05
lstd = regtd.LarsTD(tau=tau, lam=0, phi=phi)
lstd.name = r"LarsTD({}) $\tau={}$".format(0,tau)
lstd.color = "b"
methods.append(lstd)

tau=0.05
beta=0.05
lstd = regtd.LSTDl21(tau=tau,beta=beta, lam=0, phi=phi)
lstd.name = r"LSTD({}) $\ell_{{21}}$ $\tau={}$, $\beta={}$".format(0,tau, beta)
lstd.color = "b"
#methods.append(lstd)


l = 8000
n_eps = 1
n_indep = 200

episodic = False
error_every = 80
name = "disc_random"
title = "{}-State Random MDP ({} trials)".format(n, n_indep)
criterion = "RMSPBE"
criteria = ["RMSPBE", "RMSBE", "RMSE"]
if __name__ == "__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=-1, **globals())
    save_results(**globals())
    #plot_errorbar(**globals())
