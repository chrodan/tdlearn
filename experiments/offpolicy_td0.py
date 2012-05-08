# -*- coding: utf-8 -*-
"""
Experiment that shows arbitrary off-policy behavior of TDC and TD

Created on Tue Jan 31 12:13:51 2012

@author: Christoph Dann <cdann@cdann.de>
"""
import td
import examples
import numpy as np
import matplotlib.pyplot as plt
from task import LinearDiscreteValuePredictionTask

n = 7
beh_pi = np.ones((n + 1, 2))
beh_pi[:, 0] = float(n) / (n + 1)
beh_pi[:, 1] = float(1) / (n + 1)

target_pi = np.zeros((n + 1, 2))
target_pi[:, 0] = 0
target_pi[:, 1] = 1

mdp = examples.BairdStarExample(n)
phi = mdp.phi
n_iter = 700
n_indep = 50
methods = []


task = LinearDiscreteValuePredictionTask(mdp, 0.99, mdp.phi, 
                                       np.asarray(n * [1.] + [10., 1.]),
                                       policy=beh_pi, target_policy=target_pi)

for alpha, color in [(0.01, "#00AA00"), 
                     (0.03, "#0000ff"),
                     (0.1, "#ff0000"),
                     (0.0003, "#ff00ff")]:
    m = td.LinearTDLambda(alpha=alpha, phi=phi, lam=0)
    m.name = r"TD(0) $\alpha$={}".format(alpha)    
    m.color = color                                
    methods.append(m)   

mean, std, raw = task.avg_error_traces(methods, n_indep, n_samples=n_iter, criterion="RMSE")

                    
plt.figure()
plt.ylabel(r"$\sqrt{MSE}$")
plt.xlabel("Timesteps")    
for i, m in enumerate(methods): 
    plt.semilogy(mean[i,:], markevery=50, label=m.name, color=m.color)
plt.ylim(0,10**4)
plt.legend(loc="lower right")

plt.show()
