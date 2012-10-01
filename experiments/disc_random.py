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
import policies

n = 100
n_a = 20
n_feat = 30
mdp = examples.RandomMDP(n, n_a)
phi = features.lin_random(n_feat, n)
#phi = features.eye(n)
gamma = .95
np.random.seed(3)
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, np.zeros(phi.dim),
                                         policy=policies.Discrete(
                                             np.random.rand(n, n_a)),
                                         target_policy=policies.Discrete(np.random.rand(n, n_a)))

methods = []
lstd = td.LSTDLambda(lam=0, phi=phi)
lstd.name = r"LSTD({})".format(0)
lstd.color = "b"
methods.append(lstd)

lstd = td.LSTDLambdaJP(lam=0, phi=phi)
lstd.name = r"LSTD-JP({})".format(0)
lstd.color = "b"
methods.append(lstd)


l = 3000
n_eps = 1
n_indep = 1

error_every = 1
name = "boyan"
title = "{}-State Random MDP ({} trials)".format(n, n_indep)
criterion = "RMSPBE"

if __name__ == "__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=1, **globals())
    save_results(**globals())
    plot_errorbar(**globals())
