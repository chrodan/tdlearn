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
import policies

n = 7
beh_pi = np.ones((n + 1, 2))
beh_pi[:, 0] = float(n) / (n + 1)
beh_pi[:, 1] = float(1) / (n + 1)

target_pi = np.zeros((n + 1, 2))
target_pi[:, 0] = 0
target_pi[:, 1] = 1

mdp = examples.BairdStarExample(n)
phi = mdp.phi
n_iter = 300
n_indep = 30
methods = []
target_pi = policies.Discrete(target_pi)
beh_pi = policies.Discrete(beh_pi)

task = LinearDiscreteValuePredictionTask(mdp, 0.99, mdp.phi, 
                                       np.asarray(n * [1.] + [10., 1.]),
                                       policy=beh_pi, target_policy=target_pi)

for alpha, mu, color in [(0.01, 10, "red"), (0.04, 0.5, "orange")]:

    m = td.TDC(alpha=alpha, beta=mu*alpha, phi=phi)        
    m.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)        
    m.color = color            
    methods.append(m)   

for alpha, beta, c, color in [(0.52, 0.51, 0.07, "black"), (0.6, 0.51, 0.35, "cyan")]:
    
    m = td.TDC(alpha=td.RMalpha(c, alpha), beta=td.RMalpha(c, beta), phi=phi)        
    m.name = r"TDC $\alpha={c}t^{{{}}}$ $\beta={c}t^{{{}}}$".format(alpha, beta, c=c)        
    m.color = color            
    methods.append(m)
        
for alpha, color in [(0.01, "#00AA00")]: #, (0.03, "#0000ff")]:
    m = td.LinearTDLambda(alpha=alpha, phi=phi, lam=0)
    m.name = r"TD(0) $\alpha$={}".format(alpha)    
    m.color = color                                
    methods.append(m)

lstd = td.LSTDLambda(lam=0, phi=phi, init_theta=1.)
lstd.name = r"LSTD({})".format(0)
lstd.color = ""
methods.append(lstd)

lstd = td.LSTDLambda(lam=0, phi=phi, init_theta=2.)
lstd.name = r"LSTD({})".format(0)
lstd.color = ""
methods.append(lstd)

mean, std, raw = task.avg_error_traces(methods, n_indep, n_samples=n_iter, criterion="RMSPBE", verbose=11)

                    
plt.figure()
plt.ylabel(r"$\sqrt{MSE}$")
plt.xlabel("Timesteps")    
for i, m in enumerate(methods): 
    plt.errorbar(range(0,n_iter),mean[i,:], markevery=50, errorevery=50, label=m.name)#, color=m.color)
plt.ylim(0,10**4)
plt.legend(loc="lower right")
plt.yscale("log")

plt.show()