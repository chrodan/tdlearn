# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 20:11:42 2012

Experiment: search for the optimal lambda value of linear TD learning

@author: Christoph Dannn <cdann@cdann.de>
"""


import td
import examples
from task import LinearDiscreteValuePredictionTask
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.transforms as transforms

n = 19
mdp =examples.BoyanChain(n,4)
phi=mdp.phi
task = LinearDiscreteValuePredictionTask(mdp, 1, mdp.phi, np.zeros(4))


fig = plt.figure()
for lam in np.linspace(0,1,5):
    
    methods = []
    for eps in [0.01, 0.001]:
        lstd = td.LSTDLambda(eps=eps, lam=lam, phi=phi)
        lstd.name = "LSTD({}), eps={}".format(lam, eps)    
        lstd.color = "r"        
        #methods.append(lstd)
    for alpha in np.linspace(0.1, 1, 10):
        tdl = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi)
        tdl.name = "TD({}) alpha={}".format(lam, alpha)    
        tdl.color = "k"
        methods.append(tdl)
    dx, dy = 2/72.*lam*10, 0.
    offset = transforms.ScaledTranslation(dx, dy,
      fig.dpi_scale_trans)
    shadow_transform = fig.gca().transData + offset

    mean, std, raw = task.avg_error_traces(methods, 50, n_eps=20, criterion="RMSE")
    r = raw.reshape(raw.shape[0],-1)

    plt.errorbar(np.linspace(0.1, 1, 10),np.mean(r,axis=1), transform=shadow_transform,
                 yerr=np.std(r, axis=1), label=r"linear TD({})".format(lam))
plt.legend()

plt.ylabel(r"avg. $\sqrt{\operatorname{MSE}$ of first 20 episodes")
plt.xlabel(r"$\alpha$") 
plt.show()