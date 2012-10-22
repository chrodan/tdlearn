# -*- coding: utf-8 -*-
"""
Convergence speed comparison of TD methods on a (uniformly) random MDP
@author: Christoph Dann <cdann@cdann.de>
"""

import td
import examples
from task import LinearDiscreteValuePredictionTask
import numpy as np
import matplotlib.pyplot as plt
import features
import policies
import regtd
n = 50
n_a = 1
n_feat = 100
mdp = examples.RandomMDP(n, n_a)
phi = features.lin_random(n_feat, n)
#phi = features.eye(n)
gamma = .95
np.random.seed(3)
beh_pol = policies.Discrete(np.random.rand(n, n_a))
tar_pol = policies.Discrete(np.random.rand(n, n_a))
tar_pol=beh_pol
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, np.zeros(phi.dim),
                                         policy=beh_pol, target_policy=tar_pol)

methods = []
lstd = td.LSTDLambda(lam=0, phi=phi)
lstd.name = r"LSTD({})".format(0)
lstd.color = "b"
#methods.append(lstd)

lstd = td.RecursiveLSTDLambdaJP(lam=0, phi=phi)
lstd.name = r"LSTD-JP({})".format(0)
lstd.color = "b"
methods.append(lstd)


tau=0.1
lstd = regtd.DLSTD(tau=tau, lam=0, phi=phi)
lstd.name = r"D-LSTD({}) $\tau={}$".format(0,tau)
lstd.color = "b"
#methods.append(lstd)

tau=0.0001
lstd = regtd.LSTDl1(tau=tau, lam=0, phi=phi)
lstd.name = r"LLSTD-l1({}) $\tau={}$".format(0,tau)
lstd.color = "b"
methods.append(lstd)

tau=0.1
lstd = regtd.LarsTD(tau=tau, lam=0, phi=phi)
lstd.name = r"LarsTD({}) $\tau={}$".format(0,tau)
lstd.color = "b"
methods.append(lstd)

l = 600
n_eps = 1
n_indep = 1

error_every = 3
name = "boyan"
title = "{}-State Random MDP ({} trials)".format(n, n_indep)
criterion = "RMSPBE"
criteria = ["RMSPBE", "RMSE", "RMSBE"]
if __name__ == "__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=1, **globals())
    save_results(**globals())
    plot_errorbar(**globals())
