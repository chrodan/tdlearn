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
n = 15**2
n_a = 4
n_feat = 10
mdp = examples.GridWorld()
#phi = features.lin_random(n_feat, n, constant=True)
phi = features.eye(n)
gamma = .99
np.random.seed(3)
beh_pol = policies.Discrete(mdp.pol)
tar_pol=beh_pol
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, np.zeros(phi.dim),
                                         policy=beh_pol, target_policy=tar_pol)

methods = []
lstd = td.LSTDLambda(lam=0, phi=phi)
lstd.name = r"LSTD({})".format(0)
lstd.color = "b"
methods.append(lstd)

lstd = td.LSTDLambdaJP(lam=0, phi=phi)
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
lstd.name = r"LSTD-l1({}) $\tau={}$".format(0,tau)
lstd.color = "b"
methods.append(lstd)

tau=0.1
lstd = regtd.LarsTD(tau=tau, lam=0, phi=phi)
lstd.name = r"LarsTD({}) $\tau={}$".format(0,tau)
lstd.color = "b"
#methods.append(lstd)

l = 500
n_eps = 1
n_indep = 1

error_every = 1
name = "disc_random"
title = "{}-State Random MDP ({} trials)".format(n, n_indep)
criterion = "RMSPBE"
criteria = ["RMSPBE", "RMSE", "RMSBE"]
if __name__ == "__main__":
    from experiments import *
    #mean, std, raw = run_experiment(n_jobs=1, **globals())
    #for m in methods:
    #    print m.name, m.theta[-1]
    #save_results(**globals())
    #plot_errorbar(**globals())
    plt.ion()
    plt.imshow(task.V_true.reshape(15,15), interpolation="None",
               cmap=plt.cm.Spectral_r)
    for i in range(15):
        for j in range(15):
            if mdp.world[i][j] == "_":
                continue
            c = mdp.world[i][j]
            if c == "W":
                c = u"≈"
#            if c == "A":
#                c = u"😁"
            plt.text(j-0.3,i+.3,c,fontsize=20)
    plt.colorbar()
