# -*- coding: utf-8 -*-
"""
Speed comparison of TD methods on the random walk chain.

Created on Mon Jan 30 12:42:50 2012

@author: christoph
"""

import td
import examples
from task import LinearDiscreteValuePredictionTask
import numpy as np
import matplotlib.pyplot as plt


mdp =examples.RandomWalkChain(7)
task = LinearDiscreteValuePredictionTask(mdp, 1, mdp.tabular_phi, np.zeros(7))
phi=mdp.tabular_phi

methods = []
    
alpha = 0.5
mu = 0.1
#for alpha in [0.5, 0.7]:
#    for mu in [0.1, 0.25]:
gtd = td.GTD(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "#6E086D"
methods.append(gtd)

#methods = []
#for alpha in [0.1]:
#    for mu in [0.1, 0.25, 1, 2]:
alpha, mu = 0.1, 1
gtd = td.GTD2(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "#6E086D"
methods.append(gtd)

alpha=0.1
#for alpha in [0.1, 0.03]:
tdc = td.TDC(alpha=alpha, beta=alpha*0.5, phi=phi)
tdc.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, 0.5)   
tdc.color = "r"        
methods.append(tdc)

#methods = []
#for eps in np.power(10,np.arange(-1,4)):
eps=100
lstd = td.LSTDLambda(lam=0, eps=eps, phi=phi)
lstd.name = r"LSTD({}) $\epsilon$={}".format(0, eps)    
lstd.color = "b"        
methods.append(lstd)
#
#methods = []    
#for alpha in np.linspace(0.01,1,10):
alpha=0.3
rg = td.ResidualGradient(alpha=alpha, phi=phi)
rg.name = r"RG $\alpha$={}".format(alpha)    
rg.color = "brown"
methods.append(rg)    
#methods = []
alpha=0.1
#for alpha in [0.3, 0.2, 0.1, 0.06, 0.03]:

tdl = td.LinearTDLambda(alpha=alpha, lam=0, phi=phi)
tdl.name = r"TD({}) $\alpha$={}".format(0, alpha)    
tdl.color = "k"
methods.append(tdl)

n_indep = 200
n_iter = 200

mean, std, raw = task.avg_error_traces(methods, n_indep, n_eps=n_iter, criterion="RMSPBE")

plt.figure()
plt.ylabel(r"$\sqrt{MSPBE}$")
plt.xlabel("Episodes")    
plt.ylim(0,0.12)
for i, m in enumerate(methods): 
    plt.errorbar(range(len(mean[i,:])), mean[i,:], yerr=std[i,:], errorevery=25, label=m.name)
plt.legend()
plt.show()
