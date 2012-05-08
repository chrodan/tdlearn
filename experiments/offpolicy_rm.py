# -*- coding: utf-8 -*-
"""
Experiments on off-policy behavior of TDC


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
n_iter = 80000
n_indep = 20
methods = []


task = LinearDiscreteValuePredictionTask(mdp, 0.99, mdp.phi, 
                                       np.asarray(n * [1.] + [10., 1.]),
                                       policy=beh_pi, target_policy=target_pi)

for alpha, mu, color in [(0.01, 10, "red"), (0.04, 0.5, "orange")]:

    m = td.TDC(alpha=alpha, beta=mu*alpha, phi=phi)        
    m.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)        
    m.color = color            
    #methods.append(m)   
    
for alpha, beta, c, color in [(0.52, 0.51, 0.07, "black"), 
                              (0.6, 0.51, 0.35, "cyan"),
                              (0.9, 0.51, 0.35, "cyan"),
                            (0.52, 0.51, 0.1, "cyan")]:
    
    m = td.TDC(alpha=td.RMalpha(c, alpha), beta=td.RMalpha(c, beta), phi=phi)        
    m.name = r"TDC $\alpha={c}t^{{{}}}$ $\beta={c}t^{{{}}}$".format(alpha, beta, c=c)        
    m.color = color            
    methods.append(m)
        
#for alpha, c, color in [(0.52, 0.07, "#00AA00"), (0.6, 0.35, "#0000ff")]:
for c in np.arange(0.01, 0.1, 0.2):
    for alpha in np.arange(0.5, 1, 0.1):
        m = td.LinearTDLambda(alpha=td.RMalpha(c,alpha), phi=phi, lam=0)
        m.name = r"TD(0) $\alpha={c}t^{{{}}}$".format(alpha, c=c)    
        m.color = color                                
#        methods.append(m)   

mean, std, raw = task.avg_error_traces(methods, n_indep, n_samples=n_iter, criterion="RMSE")

                    
plt.figure(figsize=(8, 8))
plt.ylabel(r"$\sqrt{MSE}$")
plt.xlabel("Timesteps")    
for i, m in enumerate(methods): 
    plt.semilogy(mean[i,:], markevery=50, label=m.name) #, color=m.color)
plt.ylim(0,10**4)
plt.legend()#loc="lower right")

plt.show()
