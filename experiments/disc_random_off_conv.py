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
lam = 0.
eps = 1e4
lstd = td.RecursiveLSTDLambdaJP(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD-CO({}) $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)

brm = td.RecursiveBRMDS(phi=phi, eps=1e4)
brm.name = "BRMDS"
brm.color = "b"
brm.ls = "--"
methods.append(brm)

brm = td.RecursiveBRM(phi=phi, eps=1e4)
brm.name = "BRM"
brm.color = "b"
methods.append(brm)


l = 500010
n_eps = 1
n_indep = 20

episodic = False
error_every = 100000
name = "disc_random_off_conv"
title = "4. {}-State Random MDP Off-policy".format(n, n_indep)
criterion = "RMSPBE"
criteria = ["RMSPBE", "RMSBE", "RMSE"]
if __name__ == "__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=-2, **globals())
    save_results(**globals())
    #plot_errorbar(**globals())
